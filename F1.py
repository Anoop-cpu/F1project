# ---- Install if needed ----
# !pip install scikit-learn
# !pip install xgboost
# !pip install joblib

# ---- Imports ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, classification_report, confusion_matrix, average_precision_score
import joblib

#-- for lift --
def lift_curve(y_true, y_proba, n_bins=10):
    """
    Returns:
      pct_population: cumulative % of population (0..1)
      lift: lift at each cumulative % (>=0)
      gains: cumulative gains (0..1)
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    # sort by predicted probability (descending)
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]

    total_positives = y_sorted.sum()
    if total_positives == 0:
        raise ValueError("No positive samples in y_true; lift is undefined.")

    N = len(y_sorted)
    bin_size = int(np.ceil(N / n_bins))

    cum_pop = []
    cum_pos = []

    running_pos = 0
    running_n = 0

    for b in range(n_bins):
        start = b * bin_size
        end = min((b + 1) * bin_size, N)
        if start >= N:
            break

        running_n += (end - start)
        running_pos += y_sorted[start:end].sum()

        cum_pop.append(running_n / N)                 # cumulative % of population
        cum_pos.append(running_pos / total_positives) # cumulative % of positives (gains)

    cum_pop = np.array(cum_pop)
    gains = np.array(cum_pos)

    # lift = gains / population%
    lift = gains / cum_pop

    return cum_pop, lift, gains

# ---- Training ----

# Load dataset
df = pd.read_csv("f1_features_2014plus_selected_drivers.csv")
print("Data loaded ✅ Shape:", df.shape)

# Convert times → seconds
def time_to_seconds(t):
    if pd.isna(t): return np.nan
    if isinstance(t, (int, float)): return float(t)
    try:
        parts = str(t).split(':')
        if len(parts) == 2:  # M:S
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:  # H:M:S
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except:
        return np.nan
    return np.nan

for col in ["q1", "q2", "q3", "pit_total_duration"]:
    if col in df.columns:
        df[col] = df[col].apply(time_to_seconds)

# Keep a race grouping key for Top-K evaluation
race_key = None
for candidate in ["raceId", "race_id", "raceID"]:
    if candidate in df.columns:
        race_key = candidate
        break

if race_key is None:
    # fallback if you have year/round (common in F1 datasets)
    if "year" in df.columns and "round" in df.columns:
        df["race_key"] = df["year"].astype(str) + "_" + df["round"].astype(str)
        race_key = "race_key"

# Drop object columns
df = df.drop(columns=df.select_dtypes(include="object").columns, errors="ignore")

# ------------------------------------------
# 🔒 Drop post-race / leakage columns (but KEEP target)
# ------------------------------------------
target = "did_podium"

leaky_cols = [
    "finish_pos","did_win","did_finish","points",
    "driver_season_points","driver_season_pos","driver_season_wins",
    "team_season_points","team_season_pos","team_season_wins",
    "sprint_finish_pos","sprint_points","sprint_podium"
]
df = df.drop(columns=[c for c in leaky_cols if c in df.columns], errors="ignore")

# ------------------------------------------
# Define target
# ------------------------------------------
if target not in df.columns:
    raise ValueError("Target column 'did_podium' not found — check data!")

y = df[target].astype(int)
X = df.drop(columns=[target])
X = X.fillna(0)

# ------------------------------------------
# Split data
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

test_idx = X_test.index
race_test = df.loc[test_idx, race_key] if race_key is not None else None

# ------------------------------------------
# Train model
# ------------------------------------------
# 1) XGBoost (your current model)
xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 2) Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# ------------------------------------------
# Evaluate
# ------------------------------------------
def eval_binary_full(model, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\n{name} performance:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}")
    print("  Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("  Classification report:\n", classification_report(y_test, y_pred))

    return {"name": name, "acc": acc, "roc_auc": roc_auc, "pr_auc": pr_auc, "y_proba": y_proba}

xgb_res = eval_binary_full(xgb_model, "XGBoost")
rf_res  = eval_binary_full(rf_model,  "Random Forest")

#---- Plotting the curves ----
# ROC Curves
plt.figure(figsize=(7,5))
for res in [xgb_res, rf_res]:
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    plt.plot(fpr, tpr, label=f'{res["name"]} (AUC={res["roc_auc"]:.3f})')
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# PR Curves
plt.figure(figsize=(7,5))
for res in [xgb_res, rf_res]:
    prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
    plt.plot(rec, prec, label=f'{res["name"]} (AP={res["pr_auc"]:.3f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve Comparison")
plt.legend()
plt.show()


#------ Bar chart -----
names = [xgb_res["name"], rf_res["name"]]
accs  = [xgb_res["acc"], rf_res["acc"]]
rocs  = [xgb_res["roc_auc"], rf_res["roc_auc"]]
prs   = [xgb_res["pr_auc"], rf_res["pr_auc"]]

plt.figure(figsize=(7,5))
x = np.arange(len(names))
w = 0.25
plt.bar(x - w, accs, width=w, label="Accuracy")
plt.bar(x, rocs, width=w, label="ROC-AUC")
plt.bar(x + w, prs, width=w, label="PR-AUC")
plt.xticks(x, names)
plt.ylim(0, 1.0)
plt.title("Model Metrics Comparison")
plt.legend()
plt.show()

# Lift chart
plt.figure(figsize=(7,5))

for name, proba in [
    ("XGBoost", xgb_res["y_proba"]),
    ("Random Forest", rf_res["y_proba"]),
]:
    pop, lift, gains = lift_curve(y_test, proba, n_bins=10)
    plt.plot(pop, lift, marker="o", label=name)

# Baseline lift = 1 (random selection)
plt.plot([0, 1], [1, 1], linestyle="--", label="Random baseline")

plt.xlabel("Cumulative % of Drivers (sorted by predicted probability)")
plt.ylabel("Lift")
plt.title("Lift Chart (did_podium)")
plt.legend()
plt.ylim(bottom=0)
plt.show()

#--- podium finish hit rate ---
def top_k_hit_rate(y_true, y_proba, race_ids, k=3):
    if race_ids is None:
        print("Top-K skipped: no race key found (raceId or year+round).")
        return None

    tmp = pd.DataFrame({
        "race": race_ids.values,
        "y_true": y_true.values,
        "y_proba": y_proba
    })

    hits = 0
    total = 0
    for race, g in tmp.groupby("race"):
        g = g.sort_values("y_proba", ascending=False).head(k)
        # hit if ANY of the top-k is actually podium (=1)
        hits += int(g["y_true"].sum() > 0)
        total += 1

    return hits / total if total > 0 else None

xgb_top3 = top_k_hit_rate(y_test, xgb_res["y_proba"], race_test, k=3)
rf_top3  = top_k_hit_rate(y_test, rf_res["y_proba"],  race_test, k=3)

if xgb_top3 is not None and rf_top3 is not None:
    print(f"\nTop-3 Hit Rate (per race):")
    print(f"  XGBoost       : {xgb_top3:.4f}")
    print(f"  Random Forest : {rf_top3:.4f}")

# ------------------------------------------
# Pick best model automatically (by ROC-AUC)
# ------------------------------------------
best_name, best_model = ("XGBoost", xgb_model) if xgb_res["roc_auc"] >= rf_res["roc_auc"] else ("Random Forest", rf_model)
print(f"\n✅ Selected best model: {best_name}")

# ------------------------------------------
# Save bundle (best + both models for reference)
# ------------------------------------------
bundle = {
    "best_model_name": best_name,
    "model": best_model,              # keep this key name for compatibility
    "xgb_model": xgb_model,
    "rf_model": rf_model,
    "columns": list(X.columns),
}
joblib.dump(bundle, "model_clean.pkl")

print("✅ Saved to model_clean.pkl (contains best model + both models)")

