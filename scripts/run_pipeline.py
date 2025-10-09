# scripts/run_pipeline.py
import os
import pandas as pd
from sklearn.ensemble import IsolationForest

def clean_data(input_csv, cleaned_csv):
    """
    Cleans raw subsidy claims CSV.
    Removes empty rows, trims strings, fixes datatypes.
    """
    df = pd.read_csv(input_csv)
    df.dropna(subset=['Name', 'Aadhaar', 'ClaimAmount', 'SubsidyType', 'Date'], inplace=True)
    df['Name'] = df['Name'].str.strip()
    df['SubsidyType'] = df['SubsidyType'].str.strip()
    df['Aadhaar'] = df['Aadhaar'].astype(str)
    df['ClaimAmount'] = pd.to_numeric(df['ClaimAmount'], errors='coerce')
    df.to_csv(cleaned_csv, index=False)
    print(f"✅ Cleaned data saved to: {cleaned_csv}")
    return df

def detect_fraud(df):
    """
    Detects suspicious claims:
    - Duplicate Aadhaar
    - Duplicate Claim (same Name + SubsidyType)
    - ML anomaly on ClaimAmount
    """
    # Duplicate Aadhaar
    df['AadhaarFlag'] = df.duplicated(subset=['Aadhaar'], keep=False)

    # Duplicate Claim
    df['ClaimFlag'] = df.duplicated(subset=['Name', 'SubsidyType'], keep=False)

    # Combine flags into FraudType
    def combine_flags(row):
        flags = []
        if row['AadhaarFlag']:
            flags.append("Duplicate Aadhaar")
        if row['ClaimFlag']:
            flags.append("Duplicate Claim")
        return " | ".join(flags) if flags else None

    df['FraudType'] = df.apply(combine_flags, axis=1)

    # ML Anomaly Detection
    if 'ClaimAmount' in df.columns:
        model = IsolationForest(contamination=0.05, random_state=42)
        df['ML_Anomaly'] = model.fit_predict(df[['ClaimAmount']])
        df.loc[df['ML_Anomaly'] == -1, 'FraudType'] = df['FraudType'].fillna('') + " | ML Anomaly"
        df['FraudType'] = df['FraudType'].str.strip(" |").replace("", None)

    return df

def run_pipeline():
    raw_csv = "data/raw/uploaded_subsidy_claims.csv"
    cleaned_csv = "data/cleaned/cleaned_subsidy_claims.csv"
    flagged_csv = "data/results/fraud_flagged.csv"

    os.makedirs(os.path.dirname(cleaned_csv), exist_ok=True)
    os.makedirs(os.path.dirname(flagged_csv), exist_ok=True)

    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"❌ Input CSV not found: {raw_csv}")

    print("🚀 Starting FraudDetectX Pipeline...")
    df = clean_data(raw_csv, cleaned_csv)
    df_flagged = detect_fraud(df)
    df_flagged.to_csv(flagged_csv, index=False)
    print(f"✅ Fraud detection completed → flagged results saved at: {flagged_csv}")
    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
