import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import os

# -------------------------
# Config
# -------------------------
TARGET_SIZE = 10000   # total rows for base dataset
OUTPUT_PATH = "data/raw/simulated_subsidy_claims_10k.csv"

SUBSIDY_TYPES = ["Education", "Farming", "LPG", "Housing", "Healthcare", "Equipment"]

# Typical claim ranges by type
CLAIM_RANGES = {
    "Education": (2000, 20000),
    "Farming": (3000, 25000),
    "LPG": (1500, 8000),
    "Housing": (10000, 100000),
    "Healthcare": (500, 15000),
    "Equipment": (5000, 60000),
}

SEED = 42

# -------------------------
# Faker & random seeds
# -------------------------
fake = Faker("en_IN")
random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)

# -------------------------
# Helper to generate Aadhaar-like numbers
# -------------------------
def generate_aadhaar():
    return "".join(str(random.randint(0, 9)) for _ in range(12))

# -------------------------
# Generate base dataset
# -------------------------
def generate_base_dataset_scaled(target_size=TARGET_SIZE):
    records = []
    start_date = datetime(2024, 7, 1)

    for _ in range(target_size):
        name = fake.name()
        aadhaar = generate_aadhaar()
        subsidy = random.choice(SUBSIDY_TYPES)
        claim_min, claim_max = CLAIM_RANGES[subsidy]
        amount = int(random.uniform(claim_min, claim_max))
        date = start_date + timedelta(days=random.randint(0, 365))
        records.append([name, aadhaar, amount, subsidy, date.strftime("%Y-%m-%d"), 0])  # FraudLabel=0

    df = pd.DataFrame(records, columns=["Name", "Aadhaar", "ClaimAmount", "SubsidyType", "Date", "FraudLabel"])
    return df

# -------------------------
# Inject Fraud Patterns (scaled)
# -------------------------
def inject_fraud_scaled(df):
    random.seed(SEED)
    frauds = []

    target_size = len(df)

    # Scale fraud counts proportionally
    n_aadhaar_dup = int(0.03 * target_size)       # 3%
    n_double_claim = int(0.02 * target_size)      # 2%
    n_high_claim = int(0.015 * target_size)       # 1.5%
    n_temporal_fraud = int(0.025 * target_size)   # 2.5%

    # 1️⃣ Duplicate Aadhaar across names
    for _ in range(n_aadhaar_dup):
        aadhaar = generate_aadhaar()
        for name in random.sample(list(df["Name"].unique()), 2):
            row = df[df["Name"] == name].sample(1).iloc[0]
            frauds.append({
                "Name": row["Name"],
                "Aadhaar": aadhaar,
                "ClaimAmount": row["ClaimAmount"],
                "SubsidyType": row["SubsidyType"],
                "Date": row["Date"],
                "FraudLabel": 1
            })

    # 2️⃣ Same person claiming same subsidy twice
    for _ in range(n_double_claim):
        person = random.choice(df["Name"].unique())
        sub = random.choice(SUBSIDY_TYPES)
        date = fake.date_between(start_date="-30d", end_date="today")
        frauds.append({
            "Name": person,
            "Aadhaar": df[df["Name"] == person]["Aadhaar"].values[0],
            "ClaimAmount": random.randint(2000, 50000),
            "SubsidyType": sub,
            "Date": date,
            "FraudLabel": 1
        })

    # 3️⃣ High claim anomalies
    for _ in range(n_high_claim):
        row = df.sample(1).iloc[0]
        frauds.append({
            "Name": row["Name"],
            "Aadhaar": row["Aadhaar"],
            "ClaimAmount": random.randint(100000, 500000),
            "SubsidyType": row["SubsidyType"],
            "Date": row["Date"],
            "FraudLabel": 1
        })

    # 4️⃣ Multiple subsidies in short time window (temporal fraud)
    for _ in range(n_temporal_fraud):
        name = random.choice(df["Name"].unique())
        aadhaar = df[df["Name"] == name]["Aadhaar"].values[0]
        base_date = datetime(2025, 5, random.randint(1, 20))
        for sub in random.sample(SUBSIDY_TYPES, 3):
            frauds.append({
                "Name": name,
                "Aadhaar": aadhaar,
                "ClaimAmount": random.randint(3000, 20000),
                "SubsidyType": sub,
                "Date": (base_date + timedelta(days=random.randint(0, 6))).strftime("%Y-%m-%d"),
                "FraudLabel": 1
            })

    # Combine original data + frauds
    df_fraud = pd.DataFrame(frauds)
    df_combined = pd.concat([df, df_fraud], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df_combined

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_base_dataset_scaled(TARGET_SIZE)
    df_final = inject_fraud_scaled(df)
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Synthetic dataset with FraudLabel created → {OUTPUT_PATH} ({len(df_final)} records)")
