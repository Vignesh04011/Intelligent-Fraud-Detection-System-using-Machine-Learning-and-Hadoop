import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # add scripts/ to path

import pandas as pd
import numpy as np
import torch
import joblib
from model_def import FraudAutoEncoder  # now works reliably

def main():
    try:
        # ---------------- Load CSV ----------------
        input_path = "data/raw/uploaded_subsidy_claims.csv"
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"{input_path} not found.")
        df = pd.read_csv(input_path)
        df['Date'] = pd.to_datetime(df['Date'])
        identifiers = df[['Name', 'Aadhaar']]

        # ---------------- Rule-based Detection ----------------
        df['RuleFraud'] = ''
        # Duplicate Aadhaar used by multiple names
        aadhaar_counts = df.groupby('Aadhaar')['Name'].nunique()
        duplicate_aadhaar = aadhaar_counts[aadhaar_counts > 1].index.tolist()
        df.loc[df['Aadhaar'].isin(duplicate_aadhaar), 'RuleFraud'] += 'DuplicateAadhaar;'

        # Same Name with multiple Aadhaar numbers
        name_counts = df.groupby('Name')['Aadhaar'].nunique()
        multi_aadhaar = name_counts[name_counts > 1].index.tolist()
        df.loc[df['Name'].isin(multi_aadhaar), 'RuleFraud'] += 'MultiAadhaar;'

        # High ClaimAmount (99th percentile)
        threshold_amount = df['ClaimAmount'].quantile(0.99)
        df.loc[df['ClaimAmount'] > threshold_amount, 'RuleFraud'] += 'HighClaimAmount;'

        # Frequent claims within 7 days
        df = df.sort_values(['Aadhaar', 'Date'])
        df['PrevDate'] = df.groupby('Aadhaar')['Date'].shift(1)
        df['DaysDiff'] = (df['Date'] - df['PrevDate']).dt.days
        df.loc[df['DaysDiff'] <= 7, 'RuleFraud'] += 'FrequentClaims;'

        df['RuleFraud'] = df['RuleFraud'].replace('', 'Normal')

        # ---------------- ML-based Detection ----------------
        X = df[['ClaimAmount', 'SubsidyType', 'Date']].copy()
        X['Date'] = (X['Date'] - X['Date'].min()).dt.days

        encoder = joblib.load("models/encoder.pkl")
        scaler = joblib.load("models/scaler.pkl")

        X_cat = encoder.transform(X[['SubsidyType']])
        X_num = scaler.transform(X[['ClaimAmount', 'Date']])
        X_combined = np.hstack([X_num, X_cat])
        X_tensor = torch.tensor(X_combined, dtype=torch.float32)

        input_dim = X_combined.shape[1]
        model = FraudAutoEncoder(input_dim)
        model.load_state_dict(torch.load("models/fraud_autoencoder.pt", map_location="cpu"))
        model.eval()

        with torch.no_grad():
            X_hat = model(X_tensor)
            errors = torch.mean((X_tensor - X_hat)**2, axis=1).numpy()

        threshold = errors.mean() + 2 * errors.std()
        df['MLFraud'] = ['Suspicious' if e > threshold else 'Normal' for e in errors]

        # ---------------- Combine Rule + ML ----------------
        def combine_fraud(row):
            if row['RuleFraud'] != 'Normal':
                return row['RuleFraud']
            elif row['MLFraud'] != 'Normal':
                return row['MLFraud']
            else:
                return 'Normal'

        df['FraudType'] = df.apply(combine_fraud, axis=1)

        # ---------------- Save results ----------------
        os.makedirs("data/results", exist_ok=True)
        df_results = identifiers.copy()
        df_results['ClaimAmount'] = df['ClaimAmount']
        df_results['SubsidyType'] = df['SubsidyType']
        df_results['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df_results['ReconstructionError'] = errors
        df_results['FraudType'] = df['FraudType']
        df_results.to_csv("data/results/fraud_results.csv", index=False)

        # Write status file
        with open("pipeline_status.txt", "w") as f:
            f.write("success")

        print("✅ Hybrid fraud detection complete. Results saved.")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        with open("pipeline_status.txt", "w") as f:
            f.write("failed")
        exit(1)


if __name__ == "__main__":
    main()
