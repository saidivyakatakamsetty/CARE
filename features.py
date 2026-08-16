import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "data/care.db"

# The columns we'll compute rolling features for
FEATURE_COLS = ["hr", "resp", "lactate"]
WINDOW = 6


def get_slope(values):
    if len(values) < 2 or np.all(np.isnan(values)):
        return 0.0
    x = np.arange(len(values))
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return 0.0
    return np.polyfit(x[mask], values[mask], 1)[0]


def load_patient(patient_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM vitals_hourly WHERE patient_id = ? ORDER BY icu_los_hour",
        conn,
        params=(patient_id,),
    )
    conn.close()
    return df


def compute_features(df):
    result = df[["patient_id", "icu_los_hour", "sepsis_label"]].copy()

    for col in FEATURE_COLS:
        result[f"{col}_mean"] = df[col].rolling(window=WINDOW, min_periods=1).mean()
        result[f"{col}_std"] = df[col].rolling(window=WINDOW, min_periods=1).std()
        result[f"{col}_slope"] = df[col].rolling(window=WINDOW, min_periods=1).apply(
            get_slope, raw=True
        )

    # Shock Index: HR / SBP. Normal ~0.5-0.7; rising above ~0.9 is an early
    # red flag, often before blood pressure alone visibly drops.
    result["shock_index"] = df["hr"] / df["sbp"]

    # Approximate qSOFA: count of (resp >= 22) + (sbp <= 100).
    # fillna(False) so missing readings don't count as "meeting criteria".
    result["qsofa_approx"] = (
        (df["resp"] >= 22).fillna(False).astype(int)
        + (df["sbp"] <= 100).fillna(False).astype(int)
    )

    return result

def get_all_patient_ids():
    conn = sqlite3.connect(DB_PATH)
    ids = pd.read_sql_query("SELECT patient_id FROM patients", conn)["patient_id"].tolist()
    conn.close()
    return ids


def build_full_feature_table():
    all_feats = []
    for pid in get_all_patient_ids():
        df = load_patient(pid)
        feats = compute_features(df)
        all_feats.append(feats)
    full_table = pd.concat(all_feats, ignore_index=True)
    full_table = full_table.fillna(0)  # fill remaining NaNs (e.g. std with 1 point) with 0
    return full_table


if __name__ == "__main__":
    table = build_full_feature_table()
    print("Full feature table shape:", table.shape)
    print(table["sepsis_label"].value_counts())
    table.to_csv("data/features.csv", index=False)
    print("Saved to data/features.csv")