# ---- Install if needed ----
# !pip install importnb

# Then, in your importing notebook:
import importnb

# ---- CELL [2] ----
# ==========================================
# 🏎️  F1 Feature Integration for Jupyter
# ==========================================
# Builds a single feature table (2014+) from Kaggle CSVs.
# Paste this in ONE Jupyter cell and run.

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------
# ⚙️ Configuration — set your CSV folder path
# ------------------------------------------
DATA_DIR = Path("/Users/anooplashiyal/PycharmProjects/F1project/data")   # change if your CSVs are elsewhere

# ------------------------------------------
# 📦 Utility functions
# ------------------------------------------
def _safe_read(csv_name: str) -> pd.DataFrame:
    p = DATA_DIR / csv_name
    if not p.exists():
        print(f"⚠️ Missing: {csv_name}")
        return pd.DataFrame()
    return pd.read_csv(p)

def _to_datetime_safe(s):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series(s), errors="coerce")

# ------------------------------------------
# 🧩 Main feature integration
# ------------------------------------------
def build_feature_table(data_dir: Path = DATA_DIR, min_year: int = 2014) -> pd.DataFrame:
    raw = {
        "races": _safe_read("races.csv"),
        "results": _safe_read("results.csv"),
        "qualifying": _safe_read("qualifying.csv"),
        "driver_standings": _safe_read("driver_standings.csv"),
        "constructor_standings": _safe_read("constructor_standings.csv"),
        "circuits": _safe_read("circuits.csv"),
        "drivers": _safe_read("drivers.csv"),
        "constructors": _safe_read("constructors.csv"),
        "pit_stops": _safe_read("pit_stops.csv"),
        "lap_times": _safe_read("lap_times.csv"),
        "status": _safe_read("status.csv"),
        "sprint_results": _safe_read("sprint_results.csv"),
    }

    races = raw["races"].copy()
    results = raw["results"].copy()
    qualifying = raw["qualifying"].copy()
    driver_stand = raw["driver_standings"].copy()
    cons_stand = raw["constructor_standings"].copy()
    circuits = raw["circuits"].copy()
    drivers = raw["drivers"].copy()
    constructors = raw["constructors"].copy()
    pit = raw["pit_stops"].copy()
    laps = raw["lap_times"].copy()
    status = raw["status"].copy()
    sprint = raw["sprint_results"].copy()

    # --- Basic cleaning ---
    if "date" in races.columns:
        races["date"] = _to_datetime_safe(races["date"])
    if "dob" in drivers.columns:
        drivers["dob"] = _to_datetime_safe(drivers["dob"])
    if "year" in races.columns:
        races = races[races["year"] >= min_year]

    # Merge races → circuits (track context)
    if "circuitId" in races.columns and "circuitId" in circuits.columns:
        races = races.merge(
            circuits.add_prefix("circuit_"),
            left_on="circuitId",
            right_on="circuit_circuitId",
            how="left"
        )

    # Join results → races
    keep_race_cols = [c for c in [
        "raceId","year","round","name","date",
        "circuit_circuitId","circuit_name","circuit_country","circuit_alt"
    ] if c in races.columns]
    base = results.merge(races[keep_race_cols], on="raceId", how="inner")

    # Ensure numeric points for rolling stats
    if "points" in base.columns:
        base["points"] = pd.to_numeric(base["points"], errors="coerce")

    # Attach driver & constructor metadata
    if not drivers.empty and "dob" in drivers.columns:
        base = base.merge(drivers[["driverId","dob"]], on="driverId", how="left")
        base["driver_age"] = (base["date"] - base["dob"]).dt.days / 365.25

    if not constructors.empty:
        cname = "name" if "name" in constructors.columns else constructors.columns[-1]
        base = base.merge(
            constructors[["constructorId", cname]].rename(columns={cname:"constructor_name"}),
            on="constructorId", how="left"
        )

    # Qualifying
    if not qualifying.empty:
        q = qualifying.copy()
        q["qpos"] = pd.to_numeric(q.get("position", np.nan), errors="coerce")
        q = q.sort_values(["raceId","driverId","qpos"]).drop_duplicates(["raceId","driverId"], keep="first")
        cols = ["raceId","driverId","qpos"] + [c for c in ["q1","q2","q3"] if c in q.columns]
        base = base.merge(q[cols], on=["raceId","driverId"], how="left")
        base.rename(columns={"qpos":"qual_pos"}, inplace=True)

    # Grid and finish
    base["grid_pos"] = pd.to_numeric(base.get("grid", np.nan), errors="coerce")
    base["finish_pos"] = pd.to_numeric(base.get("positionOrder", np.nan), errors="coerce")

    # Outcomes
    base["did_win"] = (base["finish_pos"] == 1).astype(int)
    base["did_podium"] = base["finish_pos"].between(1,3).astype(int)

    # Did finish (reliability)
    base["did_finish"] = 1
    if not status.empty and "statusId" in base.columns:
        base = base.merge(status, on="statusId", how="left")
        base["did_finish"] = np.where(base["status"].str.contains("Finished", case=False, na=False), 1, 0)

    # Rolling driver form
    base = base.sort_values(["driverId","date"])
    def _driver_roll(df):
        df["driver_prev_points_3r"] = df["points"].shift().rolling(3).mean()
        df["driver_prev_wins_3r"] = df["did_win"].shift().rolling(3).mean()
        df["driver_prev_podium_3r"] = df["did_podium"].shift().rolling(3).mean()
        df["driver_prev_avg_finish_3r"] = df["finish_pos"].shift().rolling(3).mean()
        df["driver_career_starts"] = np.arange(len(df))
        return df
    base = base.groupby("driverId", group_keys=False).apply(_driver_roll)

    # Rolling team form
    base = base.sort_values(["constructorId","date"])
    def _team_roll(df):
        df["team_prev_points_3r"] = df["points"].shift().rolling(3).mean()
        df["team_prev_avg_finish_3r"] = df["finish_pos"].shift().rolling(3).mean()
        df["team_prev_podium_3r"] = df["did_podium"].shift().rolling(3).mean()
        return df
    base = base.groupby("constructorId", group_keys=False).apply(_team_roll)

    # Reliability ratios (use transform to avoid index misalignment)
    base = base.sort_values(["driverId","date"])
    base["driver_total_starts"]   = base.groupby("driverId").cumcount()
    base["driver_total_finishes"] = base.groupby("driverId")["did_finish"].transform(lambda s: s.shift().cumsum())
    base["driver_reliability"]    = base["driver_total_finishes"] / base["driver_total_starts"].replace({0: np.nan})

    base = base.sort_values(["constructorId","date"])
    base["team_total_starts"]     = base.groupby("constructorId").cumcount()
    base["team_total_finishes"]   = base.groupby("constructorId")["did_finish"].transform(lambda s: s.shift().cumsum())
    base["team_reliability"]      = base["team_total_finishes"] / base["team_total_starts"].replace({0: np.nan})

    # Track history
    if "circuit_circuitId" in base.columns:
        base = base.sort_values(["driverId","date"])
        base["driver_track_hist_finish"] = (
            base.groupby(["driverId","circuit_circuitId"])["finish_pos"]
                .transform(lambda s: s.shift().expanding().mean())
        )

    # Sprint features (optional)
    if not sprint.empty:
        spr = sprint.rename(columns={"positionOrder":"sprint_finish_pos","points":"sprint_points"})
        keep = ["raceId","driverId"] + [c for c in ["sprint_finish_pos","sprint_points"] if c in spr.columns]
        base = base.merge(spr[keep], on=["raceId","driverId"], how="left")
        if "sprint_finish_pos" in base.columns:
            base["sprint_podium"] = base["sprint_finish_pos"].between(1,3).astype(float)

    # Pit and lap aggregates
    if not pit.empty:
        pit_agg = pit.groupby(["raceId","driverId"]).agg(
            pit_count=("stop","count"), pit_total_duration=("duration","sum")
        ).reset_index()
        base = base.merge(pit_agg, on=["raceId","driverId"], how="left")
    if not laps.empty and "milliseconds" in laps.columns:
        lap_agg = laps.groupby(["raceId","driverId"]).agg(
            lap_ms_mean=("milliseconds","mean"), lap_ms_std=("milliseconds","std"), laps_completed=("lap","max")
        ).reset_index()
        base = base.merge(lap_agg, on=["raceId","driverId"], how="left")

    # Standings (as-of that race)
    if not driver_stand.empty:
        ds = driver_stand.rename(columns={
            "points":"driver_season_points",
            "position":"driver_season_pos",
            "wins":"driver_season_wins"
        })
        base = base.merge(
            ds[["raceId","driverId","driver_season_points","driver_season_pos","driver_season_wins"]],
            on=["raceId","driverId"], how="left"
        )
    if not cons_stand.empty:
        cs = cons_stand.rename(columns={
            "points":"team_season_points",
            "position":"team_season_pos",
            "wins":"team_season_wins"
        })
        base = base.merge(
            cs[["raceId","constructorId","team_season_points","team_season_pos","team_season_wins"]],
            on=["raceId","constructorId"], how="left"
        )

    # Final select
    cols = [
        "raceId","driverId","constructorId","year","date","circuit_name","circuit_country",
        "constructor_name","grid_pos","qual_pos","q1","q2","q3",
        "driver_age","driver_career_starts",
        "driver_prev_points_3r","driver_prev_wins_3r","driver_prev_podium_3r","driver_prev_avg_finish_3r",
        "team_prev_points_3r","team_prev_avg_finish_3r","team_prev_podium_3r",
        "driver_reliability","team_reliability","driver_track_hist_finish",
        "sprint_finish_pos","sprint_points","sprint_podium","pit_count","pit_total_duration",
        "lap_ms_mean","lap_ms_std","laps_completed","driver_season_points","driver_season_pos",
        "team_season_points","team_season_pos","finish_pos","did_win","did_podium","did_finish","points"
    ]
    cols = [c for c in cols if c in base.columns]
    df = base[cols].sort_values(["year","raceId","driverId"]).reset_index(drop=True)
    return df

# ------------------------------------------
# 🚀 Run in Jupyter
# ------------------------------------------
print("Building F1 features... please wait ⏳")
df = build_feature_table(DATA_DIR, min_year=2014)
print("✅ Done! Shape:", df.shape)

# Save CSVs
df.to_csv("f1_features_2014plus.csv", index=False)
df.head(200).to_csv("f1_features_preview.csv", index=False)

print("Saved:")
print(" - f1_features_2014plus.csv (full)")
print(" - f1_features_preview.csv (preview)")

# Show first few rows
df.head()

# ---- CELL [3] ----

