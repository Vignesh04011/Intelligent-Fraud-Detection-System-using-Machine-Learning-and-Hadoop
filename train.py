import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import os

# ---------------- Load dataset ----------------
df = pd.read_csv("data/raw/train.csv")
identifiers = df[['Name', 'Aadhaar']]

# ---------------- Preprocessing ----------------
X = df[['ClaimAmount', 'SubsidyType', 'Date']].copy()

# Convert Date to numeric
X['Date'] = pd.to_datetime(X['Date'])
X['Date'] = (X['Date'] - X['Date'].min()).dt.days

# Categorical encoding for SubsidyType
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(X[['SubsidyType']])

# Scale numeric features
scaler = StandardScaler()
X_num = scaler.fit_transform(X[['ClaimAmount', 'Date']])

# Combine numeric + categorical
X_combined = np.hstack([X_num, X_cat])
print(f"Feature shape: {X_combined.shape}")

# Save preprocessing objects
os.makedirs("models", exist_ok=True)
joblib.dump(encoder, "models/encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# ---------------- PyTorch Dataset ----------------
X_tensor = torch.tensor(X_combined, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ---------------- Autoencoder Definition ----------------
class FraudAutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# ---------------- Train ----------------
input_dim = X_combined.shape[1]
model = FraudAutoEncoder(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        x_batch = batch[0]
        optimizer.zero_grad()
        x_hat = model(x_batch)
        loss = criterion(x_hat, x_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x_batch)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.6f}")

# ---------------- Save model ----------------
torch.save(model.state_dict(), "models/fraud_autoencoder.pt")
print("✅ Model saved: models/fraud_autoencoder.pt")

# ---------------- Compute reconstruction error ----------------
model.eval()
with torch.no_grad():
    X_hat = model(X_tensor)
    errors = torch.mean((X_tensor - X_hat)**2, axis=1).numpy()

# Threshold for suspicious claims
threshold = errors.mean() + 2 * errors.std()
df_results = identifiers.copy()
df_results['ReconstructionError'] = errors
df_results['FraudType'] = ['Suspicious' if e > threshold else 'Normal' for e in errors]

# Save results
os.makedirs("data/results", exist_ok=True)
df_results.to_csv("data/results/fraud_results.csv", index=False)
print("✅ Results saved: data/results/fraud_results.csv")
