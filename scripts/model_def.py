# scripts/model_def.py
import torch
import torch.nn as nn

class FraudAutoEncoder(nn.Module):
    """
    Flexible AutoEncoder model for Fraud Detection.
    Automatically adapts to the number of input features.
    """

    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32):
        super(FraudAutoEncoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 🔧 Optional helper function (you can use it for debugging or testing)
if __name__ == "__main__":
    model = FraudAutoEncoder(input_dim=10)
    sample = torch.randn(1, 10)
    out = model(sample)
    print("✅ Model initialized and forward pass successful, output shape:", out.shape)
