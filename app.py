import os, pandas as pd, joblib, numpy as np
from pathlib import Path

DATA_DIR = "/Users/anooplashiyal/PycharmProjects/F1project"
MODEL_PATHS = ["model_clean.pkl", f"{DATA_DIR}/model_clean.pkl"]

def build_simulated_race(track, df_all, model_cols, rng_seed=42):
    """
    Build a synthetic 'future race' for the given track.

    - Uses the latest year in df_all where this track appears.
    - Takes each driver's latest row at that track as a baseline.
    - Randomizes the starting grid for a new race.
    - Returns a DataFrame ready for preprocess_for_model + model.predict_proba.
    """
    df_track = df_all[df_all["circuit_name"] == track].copy()
    if df_track.empty:
        raise ValueError(f"No historical data for track '{track}'")

    latest_year = int(df_track["year"].max())
    df_latest = df_track[df_track["year"] == latest_year].copy()

    # For each driver, keep their last appearance at this track in that year
    df_latest = (df_latest
                 .sort_values(["driverId", "date"])
                 .groupby("driverId", as_index=False)
                 .tail(1)
                 .reset_index(drop=True))

    # If no grid_pos column exists, create a simple 1..N baseline
    if "grid_pos" not in df_latest.columns:
        if "grid" in df_latest.columns:
            df_latest["grid_pos"] = pd.to_numeric(df_latest["grid"], errors="coerce")
        else:
            df_latest["grid_pos"] = np.arange(1, len(df_latest) + 1)

    # Randomize grid for a new race (simple simulation: random permutation)
    rng = np.random.default_rng(rng_seed)
    df_sim = df_latest.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)
    df_sim["grid_pos"] = np.arange(1, len(df_sim) + 1)
    if "grid" in df_sim.columns:
        df_sim["grid"] = df_sim["grid_pos"]

    # Mark as future race (optional; purely cosmetic)
    df_sim["year"] = latest_year + 1
    df_sim["simulated"] = 1

    # We keep all other engineered features (season form, track history, etc.)
    # The model will only use model_cols anyway.
    return df_sim

# Load model bundle (model + expected columns)
for p in MODEL_PATHS:
    if os.path.exists(p):
        bundle = joblib.load(p); break
else:
    raise FileNotFoundError("model_clean.pkl not found in CWD or /mnt/data")

model, model_cols = bundle["model"], bundle["columns"]

# Load your integrated dataset
df = pd.read_csv("/Users/anooplashiyal/Documents/ADM/f1_features_2014plus_selected_drivers.csv")

# Pick a race (change as you like)
TRACK, YEAR = "Circuit de Monaco", 2023
subset = df[(df["year"] == YEAR) & (df["circuit_name"].str.lower() == TRACK.lower())].copy()
if subset.empty:
    raise ValueError(f"No rows found for {TRACK} {YEAR}. Try another track/year.")

# Keep model columns (align & fill)
X = subset.copy()
for c in model_cols:
    if c not in X.columns:
        X[c] = 0
X = X[model_cols].fillna(0)

# ---- Make X numeric (fix time-like strings) ----
import pandas as pd

def time_to_seconds(t):
    if pd.isna(t): return np.nan
    if isinstance(t, (int, float)): return float(t)
    try:
        parts = str(t).strip().split(':')
        if len(parts) == 2:   # M:S(.ms)
            return float(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:   # H:M:S(.ms)
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        return np.nan
    return np.nan

# convert known time columns if they exist in X
for col in ("q1", "q2", "q3", "pit_total_duration"):
    if col in X.columns:
        X[col] = X[col].apply(time_to_seconds)

# finally, force everything to numeric
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# Predict
subset["podium_probability"] = model.predict_proba(X)[:, 1]

# Pretty output
drivers = pd.read_csv("/Users/anooplashiyal/PycharmProjects/F1project/data/drivers.csv")
constructors = pd.read_csv("/Users/anooplashiyal/PycharmProjects/F1project/data/constructors.csv")
out = (subset[["driverId","constructorId","podium_probability"]]
       .merge(drivers[["driverId","forename","surname"]], on="driverId", how="left")
       .merge(constructors[["constructorId","name"]].rename(columns={"name":"constructor"}), on="constructorId", how="left")
       .sort_values("podium_probability", ascending=False)
       .reset_index(drop=True))

out.head(10).style.format({"podium_probability": "{:.1%}"})

# ---- CELL [3] ----
# app.py — F1 Podium Predictor (CSV + model_clean.pkl)
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Point to your data. If drivers/constructors are in ./archive, keep this.
DATA_DIR = Path("/Users/anooplashiyal/PycharmProjects/F1project/data")  # change if needed

CSV_PATHS = [Path("f1_features_2014plus_selected_drivers.csv"), "/Users/anooplashiyal/PycharmProjects/F1project/f1_features_2014plus_selected_drivers.csv"]
MODEL_PATHS = [Path("model_clean.pkl"), "/Users/anooplashiyal/PycharmProjects/F1project/model_clean.pkl"]
DRIVERS_PATHS = [Path("drivers.csv"), DATA_DIR / "drivers.csv"]
CONSTRUCTORS_PATHS = [Path("constructors.csv"), DATA_DIR / "constructors.csv"]

def load_first(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these exist:\n" + "\n".join(map(str, paths)))

@st.cache_data
def load_data():
    df = pd.read_csv(load_first(CSV_PATHS))
    drv = pd.read_csv(load_first(DRIVERS_PATHS))
    con = pd.read_csv(load_first(CONSTRUCTORS_PATHS))
    return df, drv, con

@st.cache_resource
def load_model():
    bundle = joblib.load(load_first(MODEL_PATHS))
    return bundle["model"], bundle["columns"]

def time_to_seconds(t):
    if pd.isna(t): return np.nan
    if isinstance(t, (int, float)): return float(t)
    try:
        parts = str(t).strip().split(':')
        if len(parts) == 2:   # M:S(.ms)
            return float(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:   # H:M:S(.ms)
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        return np.nan
    return np.nan

def preprocess_for_model(X, model_cols):
    for c in ("q1", "q2", "q3", "pit_total_duration"):
        if c in X.columns:
            X[c] = X[c].apply(time_to_seconds)
    for c in model_cols:
        if c not in X.columns:
            X[c] = 0
    return X[model_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

st.set_page_config(page_title="F1 Podium Predictor", layout="wide")
st.title("🏁 F1 Podium Predictor (2014+)")

df_all, drivers, constructors = load_data()
model, model_cols = load_model()

# ---------------- Sidebar: mode + track/year ----------------
mode = st.sidebar.radio(
    "Mode",
    ["Historical replay", "Simulated future race"],
    index=0,
    help="Historical replay: score a real past race. Simulated: generate a new hypothetical race at this track."
)

tracks = sorted(df_all["circuit_name"].dropna().unique().tolist())
track = st.sidebar.selectbox(
    "Track",
    tracks,
    index=tracks.index("Monza") if "Monza" in tracks else 0
)

if mode == "Historical replay":
    years = sorted(df_all["year"].dropna().unique().tolist())
    year_default = years.index(2024) if 2024 in years else len(years) - 1
    year = st.sidebar.selectbox("Year", years, index=year_default)

    subset = df_all[(df_all["year"] == year) & (df_all["circuit_name"] == track)].copy()
    if subset.empty:
        st.warning("No rows for that track/year. Try another choice.")
        st.stop()

    st.write(f"Using historical race: **{track} {year}**")
    # Prepare features -> predict
    X = preprocess_for_model(subset.copy(), model_cols)
    subset["podium_probability"] = model.predict_proba(X)[:, 1]

    # Attach names
    out = (subset[["driverId", "constructorId", "podium_probability"]]
           .merge(drivers[["driverId", "forename", "surname"]], on="driverId", how="left")
           .merge(constructors[["constructorId", "name"]].rename(columns={"name": "constructor"}), on="constructorId",
                  how="left")
           .sort_values("podium_probability", ascending=False)
           .reset_index(drop=True))

    st.subheader(f"Predicted Podium Probabilities — {track}")
    st.dataframe(
        out[["forename", "surname", "constructor", "podium_probability"]]
        .style.format({"podium_probability": "{:.2%}"}),
        width="stretch"
    )

    st.success("🏆 Top 3")
    for i, row in out.head(3).iterrows():
        st.write(
            f"**{i + 1}. {row['forename']} {row['surname']} ({row['constructor']})** — {row['podium_probability']:.1%}")
else:  # mode == "Simulated future race"
    st.sidebar.write("Grid & race are simulated based on latest available season at this track.")
    # Build a synthetic future race
    subset = build_simulated_race(track, df_all, model_cols, rng_seed=42)
    sim_year = int(subset["year"].iloc[0])
    st.write(f"Simulated future race at **{track} {sim_year}** (randomized grid based on latest season data)")
    # Prepare features -> predict
    X = preprocess_for_model(subset.copy(), model_cols)
    subset["podium_probability"] = model.predict_proba(X)[:, 1]

    # Attach names
    out = (subset[["driverId", "constructorId", "podium_probability"]]
           .merge(drivers[["driverId", "forename", "surname"]], on="driverId", how="left")
           .merge(constructors[["constructorId", "name"]].rename(columns={"name": "constructor"}), on="constructorId",
                  how="left")
           .sort_values("podium_probability", ascending=False)
           .reset_index(drop=True))

    st.subheader(f"Predicted Podium Probabilities — {track}")
    st.dataframe(
        out[["forename", "surname", "constructor", "podium_probability"]]
        .style.format({"podium_probability": "{:.2%}"}),
        width="stretch"
    )

    st.success("🏆 Top 3")
    for i, row in out.head(3).iterrows():
        st.write(
            f"**{i + 1}. {row['forename']} {row['surname']} ({row['constructor']})** — {row['podium_probability']:.1%}")

# # ---------------- UI ----------------
# st.set_page_config(page_title="F1 Podium Predictor", layout="wide")
# st.title("🏁 F1 Podium Predictor (2014+)")
#
# df_all, drivers, constructors = load_data()
# model, model_cols = load_model()
#
# # Sidebar selectors
# tracks = sorted(df_all["circuit_name"].dropna().unique().tolist())
# years  = sorted(df_all["year"].dropna().unique().tolist())
# col1, col2 = st.sidebar.columns(2)
# track_default = tracks.index("Monza") if "Monza" in tracks else 0
# year_default  = years.index(2024) if 2024 in years else len(years)-1
# track = col1.selectbox("Track", tracks, index=track_default)
# year  = col2.selectbox("Year", years, index=year_default)
#
# # Filter the CSV for selected race
# subset = df_all[(df_all["year"] == year) & (df_all["circuit_name"] == track)].copy()
# if subset.empty:
#     st.warning("No rows for that track/year. Try another choice.")
#     st.stop()
#
# # Prepare features -> predict
# X = preprocess_for_model(subset.copy(), model_cols)
# subset["podium_probability"] = model.predict_proba(X)[:, 1]
#
# # Attach names
# out = (subset[["driverId","constructorId","podium_probability"]]
#        .merge(drivers[["driverId","forename","surname"]], on="driverId", how="left")
#        .merge(constructors[["constructorId","name"]].rename(columns={"name":"constructor"}), on="constructorId", how="left")
#        .sort_values("podium_probability", ascending=False)
#        .reset_index(drop=True))
#
# st.subheader(f"Predicted Podium Probabilities — {track} ({year})")
# st.dataframe(
#     out[["forename","surname","constructor","podium_probability"]]
#       .style.format({"podium_probability": "{:.2%}"}),
#     width="stretch"
# )
#
# st.success("🏆 Top 3")
# for i, row in out.head(3).iterrows():
#     st.write(f"**{i+1}. {row['forename']} {row['surname']} ({row['constructor']})** — {row['podium_probability']:.1%}")
#
# st.caption("Model: XGBoost (clean, no leakage). Data: 2014–present Kaggle F1.")
