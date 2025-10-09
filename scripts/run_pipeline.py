# scripts/run_pipeline.py
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

# ---------------- Load Autoencoder -----------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def load_autoencoder():
    model_path = "artifacts/models/fraud_autoencoder.pt"
    scaler_path = "artifacts/models/scaler.pkl"
    enc_path = "artifacts/models/encoder.pkl"
    if not os.path.exists(model_path):
        print("⚠️ Autoencoder not found, skipping deep anomaly detection.")
        return None, None, None
    scaler = joblib.load(scaler_path)
    enc = joblib.load(enc_path)
    sample = np.hstack([scaler.transform([[0]]),
                        enc.transform([["Education"]])])
    model = AutoEncoder(input_dim=sample.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, scaler, enc

# ---------------- Cleaning & Rule Checks -----------------
def clean_data(input_csv, cleaned_csv):
    df = pd.read_csv(input_csv)
    df.dropna(subset=['Name','Aadhaar','ClaimAmount','SubsidyType','Date'], inplace=True)
    df['Aadhaar'] = df['Aadhaar'].astype(str)
    df['ClaimAmount'] = pd.to_numeric(df['ClaimAmount'], errors='coerce')
    df.to_csv(cleaned_csv, index=False)
    return df

def detect_fraud(df):
    """
    Detects fraud using:
    - Duplicate Aadhaar
    - Duplicate Claims (same Name + SubsidyType)
    - ML Anomaly on ClaimAmount
    - Temporal Fraud (multiple claims in short time)
    - Subsidy Overlap (many subsidy types in short window)
    """
    # Ensure proper datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by=['Aadhaar', 'Date'])

    # --- 1️⃣ Rule-based duplicates ---
    df['AadhaarFlag'] = df.duplicated(subset=['Aadhaar'], keep=False)
    df['ClaimFlag'] = df.duplicated(subset=['Name', 'SubsidyType'], keep=False)

    # --- 2️⃣ Temporal Fraud (same Aadhaar, claims < 3 days apart) ---
    df['TemporalFlag'] = False
    for aadhaar, group in df.groupby('Aadhaar'):
        if len(group) > 1:
            dates = group['Date'].sort_values().values
            diffs = np.diff(dates).astype('timedelta64[D]').astype(int)
            if any(d <= 3 for d in diffs):
                df.loc[df['Aadhaar'] == aadhaar, 'TemporalFlag'] = True

    # --- 3️⃣ Subsidy Overlap Fraud (same Name, many subsidy types within 7 days) ---
    df['OverlapFlag'] = False
    for name, group in df.groupby('Name'):
        if len(group) > 1:
            for i, row in group.iterrows():
                recent = group[
                    (abs((group['Date'] - row['Date']).dt.days) <= 7)
                ]
                if recent['SubsidyType'].nunique() >= 3:
                    df.loc[df['Name'] == name, 'OverlapFlag'] = True
                    break

    # --- 4️⃣ Combine all rule-based flags ---
    def combine_flags(row):
        flags = []
        if row['AadhaarFlag']:
            flags.append("Duplicate Aadhaar")
        if row['ClaimFlag']:
            flags.append("Duplicate Claim")
        if row['TemporalFlag']:
            flags.append("Frequent Claims")
        if row['OverlapFlag']:
            flags.append("Subsidy Overlap")
        return " | ".join(flags) if flags else None

    df['FraudType'] = df.apply(combine_flags, axis=1)

    # --- 5️⃣ ML anomaly on ClaimAmount ---
    if 'ClaimAmount' in df.columns:
        model = IsolationForest(contamination=0.05, random_state=42)
        df['ML_Anomaly'] = model.fit_predict(df[['ClaimAmount']])
        df.loc[df['ML_Anomaly'] == -1, 'FraudType'] = (
            df['FraudType'].fillna('') + " | ML Anomaly"
        ).str.strip(" |")

    return df

# ---------------- Deep Learning Inference -----------------
def add_autoencoder_anomaly(df):
    model, scaler, enc = load_autoencoder()
    if model is None:
        df['DL_Anomaly'] = False
        return df

    feats = df[['ClaimAmount','SubsidyType']].copy()
    sub_encoded = enc.transform(feats[['SubsidyType']])
    claim_scaled = scaler.transform(feats[['ClaimAmount']])
    X = np.hstack([claim_scaled, sub_encoded])
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        recon = model(X_tensor)
        loss = ((X_tensor - recon)**2).mean(dim=1).numpy()
    threshold = np.percentile(loss, 95)   # top 5 % = anomalies
    df['DL_Anomaly'] = loss > threshold
    df.loc[df['DL_Anomaly'], 'FraudType'] = (
        df['FraudType'].fillna('') + " | Deep Anomaly").str.strip(" |")
    return df

# ---------------- Pipeline -----------------
def run_pipeline():
    raw_csv = "data/raw/uploaded_subsidy_claims.csv"
    cleaned_csv = "data/cleaned/cleaned_subsidy_claims.csv"
    flagged_csv = "data/results/fraud_flagged.csv"
    os.makedirs(os.path.dirname(flagged_csv), exist_ok=True)

    print("🚀 Starting FraudDetectX Pipeline...")
    df = clean_data(raw_csv, cleaned_csv)
    df = detect_fraud(df)
    df = add_autoencoder_anomaly(df)
    df.to_csv(flagged_csv, index=False)
    print(f"✅ Results saved → {flagged_csv}")

if __name__ == "__main__":
    run_pipeline()
