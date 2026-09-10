import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from feature_extraction import FeatureExtractor
from ai_model import UAVEstimator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANGE_MAX = 500.0
VEL_MAX = 50.0
DOA_MAX = 90.0
CARRIER_FREQ = 6e9
N_ANTENNAS = 8


class UAVDataset(Dataset):
    """Dataset with real array-based DOA estimation and no DOA label leakage."""
    def __init__(self, files):
        self.X, self.y = [], []
        self.fe = FeatureExtractor(carrier_freq=CARRIER_FREQ)

        for f in files:
            with h5py.File(f, "r") as hf:
                Xd = hf["X"][:]
                Yd = hf["y"][:]

            if Xd.ndim != 3:
                raise ValueError(f"{f}: expected X shape [samples, antennas, time], got {Xd.shape}")
            if Xd.shape[1] != N_ANTENNAS:
                raise ValueError(f"{f}: expected {N_ANTENNAS} antennas, got {Xd.shape[1]}")

            labels = Yd.astype(np.float32).copy()
            labels[:, 0] /= RANGE_MAX
            labels[:, 1] /= VEL_MAX
            labels[:, 2] /= DOA_MAX
            self.X.append(Xd)
            self.y.append(labels)

        if not self.X:
            raise FileNotFoundError("No dataset files were provided.")

        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        rx_array = self.X[idx]
        label = self.y[idx]

        # Array-based DOA is estimated only from received antenna signals.
        estimated_doa = self.fe.doa_from_ula(rx_array)

        # Combine antenna channels for the existing range/Doppler branches.
        rx_combined = np.mean(rx_array, axis=0)
        feats = self.fe.extract_features(rx_combined, doa_deg=estimated_doa)

        return (
            torch.tensor(feats["range"], dtype=torch.float32),
            torch.tensor(feats["doppler"], dtype=torch.float32),
            torch.tensor(feats["doa"], dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(estimated_doa, dtype=torch.float32),
        )


def train_model(files, epochs=15, batch_size=32, lr=1e-4):
    dataset = UAVDataset(files)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = UAVEstimator().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for range_feat, doppler_feat, doa_feat, labels, _ in train_loader:
            range_feat = range_feat.to(device)
            doppler_feat = doppler_feat.to(device)
            doa_feat = doa_feat.to(device)
            labels = labels.to(device)

            preds = model(range_feat, doppler_feat, doa_feat)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for range_feat, doppler_feat, doa_feat, labels, _ in test_loader:
                range_feat = range_feat.to(device)
                doppler_feat = doppler_feat.to(device)
                doa_feat = doa_feat.to(device)
                labels = labels.to(device)
                preds = model(range_feat, doppler_feat, doa_feat)
                test_loss += criterion(preds, labels).item()

        test_loss /= max(len(test_loader), 1)
        scheduler.step(test_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss={train_loss:.6f} | "
            f"Test Loss={test_loss:.6f} | LR={optimizer.param_groups[0]['lr']:.6f}"
        )

    torch.save(model.state_dict(), "adyant_uav_estimator.pth")
    print("Model trained and saved as adyant_uav_estimator.pth")


if __name__ == "__main__":
    files = sorted(glob.glob("datasets/adyant_isac_*.h5"))
    print(f"Training on {len(files)} scenario datasets: {files}")
    train_model(files, epochs=15, batch_size=32, lr=1e-4)
