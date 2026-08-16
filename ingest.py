import sqlite3
import pandas as pd
import os
from pathlib import Path

RAW_DIR = Path("data/raw")
DB_PATH = Path("data/care.db")


def create_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            age REAL,
            gender INTEGER,
            ever_septic INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vitals_hourly (
            patient_id TEXT,
            icu_los_hour INTEGER,
            hr REAL,
            o2sat REAL,
            temp REAL,
            sbp REAL,
            resp REAL,
            wbc REAL,
            lactate REAL,
            sepsis_label INTEGER
        )
    """)
    conn.commit()


def ingest_patient_file(conn, filepath, patient_id):
    df = pd.read_csv(filepath, sep="|")

    age = df["Age"].iloc[0]
    gender = df["Gender"].iloc[0]
    ever_septic = int(df["SepsisLabel"].sum() > 0)

    conn.execute(
        "INSERT INTO patients (patient_id, age, gender, ever_septic) VALUES (?, ?, ?, ?)",
        (patient_id, age, gender, ever_septic)
    )

    for idx, row in df.iterrows():
        conn.execute(
            """INSERT INTO vitals_hourly
               (patient_id, icu_los_hour, hr, o2sat, temp, sbp, resp, wbc, lactate, sepsis_label)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                patient_id,
                row["ICULOS"],
                row["HR"],
                row["O2Sat"],
                row["Temp"],
                row["SBP"],
                row["Resp"],
                row["WBC"],
                row["Lactate"],
                row["SepsisLabel"],
            )
        )

    conn.commit()


def ingest_all():
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    for folder in ["training_setA", "training_setB"]:
        folder_path = RAW_DIR / folder
        files = sorted(os.listdir(folder_path))[:50]  # first 50 files, for speed

        for fname in files:
            if not fname.endswith(".psv"):
                continue
            patient_id = fname.replace(".psv", "")
            filepath = folder_path / fname
            ingest_patient_file(conn, filepath, patient_id)
            print(f"Ingested {patient_id}")

    conn.close()


if __name__ == "__main__":
    ingest_all()