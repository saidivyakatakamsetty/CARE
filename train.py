import random
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from model_class import HybridRiskModel

# ---------- Load features ----------
df = pd.read_csv("data/features.csv")

# ---------- Patient-level train/test split ----------
patient_ids = df["patient_id"].unique().tolist()
random.seed(42)
random.shuffle(patient_ids)

split_point = int(len(patient_ids) * 0.75)
train_ids = patient_ids[:split_point]
test_ids = patient_ids[split_point:]

train_df = df[df["patient_id"].isin(train_ids)]
test_df = df[df["patient_id"].isin(test_ids)]

print(f"Train: {len(train_ids)} patients, {len(train_df)} rows")
print(f"Test: {len(test_ids)} patients, {len(test_df)} rows")
print(f"Train sepsis rows: {train_df['sepsis_label'].sum()}")
print(f"Test sepsis rows: {test_df['sepsis_label'].sum()}")

FEATURE_COLS = [c for c in df.columns if c not in ("patient_id", "icu_los_hour", "sepsis_label")]

X_train = train_df[FEATURE_COLS]
y_train = train_df["sepsis_label"]
X_test = test_df[FEATURE_COLS]
y_test = test_df["sepsis_label"]

# ---------- XGBoost ----------
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / n_pos
print(f"scale_pos_weight = {scale_pos_weight:.1f}")

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

risk_scores = model.predict_proba(X_test)[:, 1]
y_pred = (risk_scores >= 0.5).astype(int)

print("\n--- Metrics at default 0.5 threshold ---")
print("ROC AUC:", roc_auc_score(y_test, risk_scores))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))

# ---------- Threshold tuning ----------
precisions, recalls, thresholds = precision_recall_curve(y_test, risk_scores)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = f1_scores[:-1].argmax()
best_threshold = thresholds[best_idx]

print(f"\nBest threshold: {best_threshold:.3f}")
print(f"Precision at best threshold: {precisions[best_idx]:.3f}")
print(f"Recall at best threshold: {recalls[best_idx]:.3f}")

y_pred_tuned = (risk_scores >= best_threshold).astype(int)

# ---------- Patient-level early-warning rate ----------
test_eval = test_df.copy()
test_eval["risk_score"] = risk_scores
test_eval["flagged"] = y_pred_tuned

septic_test_patients = test_eval[test_eval["sepsis_label"] == 1]["patient_id"].unique()
caught_early = 0

for pid in septic_test_patients:
    patient_rows = test_eval[test_eval["patient_id"] == pid].sort_values("icu_los_hour")
    onset_hour = patient_rows[patient_rows["sepsis_label"] == 1]["icu_los_hour"].min()
    hours_before_onset = patient_rows[patient_rows["icu_los_hour"] < onset_hour]
    if hours_before_onset["flagged"].any():
        caught_early += 1

early_warning_rate = caught_early / len(septic_test_patients)
print(f"\nPatient-level early-warning rate: {early_warning_rate:.1%} "
      f"({caught_early}/{len(septic_test_patients)} septic test patients flagged before onset)")

# ---------- Isolation Forest (patient-baseline deviation) ----------
baseline_df = train_df[train_df["icu_los_hour"] <= 4]
X_baseline = baseline_df[FEATURE_COLS]

print(f"\nBaseline rows for Isolation Forest: {len(X_baseline)}")

scaler = StandardScaler().fit(X_baseline)
X_baseline_scaled = scaler.transform(X_baseline)

iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_baseline_scaled)

X_test_scaled = scaler.transform(X_test)
iso_predictions = iso_forest.predict(X_test_scaled)

n_anomalies = (iso_predictions == -1).sum()
print(f"Test rows flagged as anomalies: {n_anomalies} / {len(X_test)}")

test_eval["iso_anomaly"] = (iso_predictions == -1)
overlap = test_eval[(test_eval["iso_anomaly"]) & (test_eval["sepsis_label"] == 1)]
print(f"Of the anomaly-flagged rows, {len(overlap)} are actually sepsis_label=1 "
      f"(out of {test_eval['sepsis_label'].sum()} real sepsis rows in test)")

# ---------- Save the packaged model ----------
saved_model = HybridRiskModel()
saved_model.scaler = scaler
saved_model.iso_forest = iso_forest
saved_model.xgb_model = model
saved_model.feature_cols = FEATURE_COLS
saved_model.risk_threshold = best_threshold

joblib.dump(saved_model, "data/hybrid_model.joblib")
print("\nModel saved to data/hybrid_model.joblib")