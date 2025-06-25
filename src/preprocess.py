import pandas as pd
import os

# Load raw data
df = pd.read_csv("data/raw/loan_data.csv")

# Simple preprocessing: drop nulls
df_cleaned = df.dropna()

# Save processed data
os.makedirs("data/processed", exist_ok=True)
df_cleaned.to_csv("data/processed/processed_data.csv", index=False)

print("✅ Data preprocessing complete.")
