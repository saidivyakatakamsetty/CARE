import sqlite3

conn = sqlite3.connect("data/care.db")
conn.row_factory = sqlite3.Row  # lets us access columns by name

# 1. How many patients total, and how many were ever septic?
cur = conn.execute("SELECT COUNT(*) as total, SUM(ever_septic) as septic FROM patients")
row = cur.fetchone()
print("Total patients:", row["total"])
print("Ever septic:", row["septic"])

# 2. How many hourly rows total?
cur = conn.execute("SELECT COUNT(*) as total FROM vitals_hourly")
print("Total hourly rows:", cur.fetchone()["total"])

# 3. Pull one septic patient's full timeline, ordered by hour
cur = conn.execute("""
    SELECT patient_id FROM patients WHERE ever_septic = 1 LIMIT 1
""")
pid = cur.fetchone()["patient_id"]
print("\nLooking at patient:", pid)

cur = conn.execute("""
    SELECT icu_los_hour, hr, resp, lactate, sepsis_label
    FROM vitals_hourly
    WHERE patient_id = ?
    ORDER BY icu_los_hour
""", (pid,))

for row in cur.fetchall():
    print(dict(row))

conn.close()