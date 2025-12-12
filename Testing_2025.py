import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def time_to_seconds(t):
    if pd.isna(t): return np.nan
    if isinstance(t, (int, float)): return float(t)
    try:
        parts = str(t).split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except:
        return np.nan
    return np.nan

# Load trained bundle
bundle = joblib.load("model_clean.pkl")
model = bundle["model"]
train_cols = bundle["columns"]

# Load 2025 unseen data
df_2025 = pd.read_csv("f1_features_2025_all_drivers_with_grid_formatted.csv")

# Apply SAME preprocessing
for col in ["q1", "q2", "q3", "pit_total_duration"]:
    if col in df_2025.columns:
        df_2025[col] = df_2025[col].apply(time_to_seconds)

# Drop object columns (same as training)
df_2025 = df_2025.drop(columns=df_2025.select_dtypes(include="object").columns, errors="ignore")

# Drop leakage cols if present (same list you used)
leaky_cols = [
    "finish_pos","did_win","did_finish","points",
    "driver_season_points","driver_season_pos","driver_season_wins",
    "team_season_points","team_season_pos","team_season_wins",
    "sprint_finish_pos","sprint_points","sprint_podium"
]
df_2025 = df_2025.drop(columns=[c for c in leaky_cols if c in df_2025.columns], errors="ignore")

# Align columns to training schema
X_2025 = df_2025.copy()
if "did_podium" in X_2025.columns:
    y_2025 = X_2025["did_podium"].astype(int)
    X_2025 = X_2025.drop(columns=["did_podium"])
else:
    y_2025 = None

# add missing cols, drop extras, order columns
for c in train_cols:
    if c not in X_2025.columns:
        X_2025[c] = 0
X_2025 = X_2025[train_cols].fillna(0)

# Predict
proba_2025 = model.predict_proba(X_2025)[:, 1]
pred_2025 = (proba_2025 >= 0.5).astype(int)

print("2025 predictions generated ✅")
print("Avg predicted podium probability:", float(np.mean(proba_2025)))

# Evaluate if labels exist
if y_2025 is not None:
    acc = accuracy_score(y_2025, pred_2025)
    auc = roc_auc_score(y_2025, proba_2025)
    print("\n2025 Test Performance:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_2025, pred_2025))
else:
    print("\nNo 'did_podium' in 2025 file — cannot compute accuracy/AUC yet.")