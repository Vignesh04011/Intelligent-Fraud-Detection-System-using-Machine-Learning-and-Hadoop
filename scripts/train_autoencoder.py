# scripts/train_autoencoder.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score

MODEL_PATH = "artifacts/models/fraud_autoencoder.pt"
SCALER_PATH = "artifacts/models/scaler.pkl"
ENCODER_PATH = "artifacts/models/encoder.pkl"

# ---------------- Autoencoder -----------------
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

# ---------------- Training -----------------
def train_autoencoder(csv_path="data/raw/simulated_subsidy_claims_10k.csv"):
    os.makedirs("artifacts/models", exist_ok=True)

    # Load dataset
    df = pd.read_csv(csv_path)
    features = df[['ClaimAmount', 'SubsidyType']].copy()

    # Encode categorical feature
    enc = OneHotEncoder(sparse_output=False)
    sub_type_encoded = enc.fit_transform(features[['SubsidyType']])

    # Scale numerical feature
    scaler = MinMaxScaler()
    claim_scaled = scaler.fit_transform(features[['ClaimAmount']])

    # Combine features
    X = np.hstack([claim_scaled, sub_type_encoded])
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Save encoders
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(enc, ENCODER_PATH)

    # DataLoader
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoEncoder(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"🚀 Training Autoencoder on {device} ({len(X)} samples)")
    for epoch in range(25):
        epoch_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1:02d}  |  Loss: {epoch_loss/len(loader):.6f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")

    # ---------------- Evaluate Reconstruction Error ----------------
    model.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        X_pred = model(X_tensor)
        recon_error = torch.mean((X_pred - X_tensor)**2, dim=1).cpu().numpy()

    # If FraudLabel exists, compute fraud detection metrics
    if 'FraudLabel' in df.columns:
        threshold = np.percentile(recon_error, 95)  # top 5% as fraud
        y_pred = (recon_error > threshold).astype(int)
        y_true = df['FraudLabel'].values

        print("\n===== Fraud Detection Report =====")
        print(classification_report(y_true, y_pred))
        print("ROC-AUC:", roc_auc_score(y_true, recon_error))
    else:
        print("\n⚠️ FraudLabel not found. Reconstruction error computed but no accuracy metrics available.")
        print(f"Sample reconstruction errors: {recon_error[:10]}")

if __name__ == "__main__":
    train_autoencoder()
