import pandas as pd

path = "data/raw/training_setA/p000028.psv"
df = pd.read_csv(path, sep="|")

# Just show ICULOS (hour number) and SepsisLabel side by side
print(df[["ICULOS", "SepsisLabel"]].to_string())