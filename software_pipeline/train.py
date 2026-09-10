import glob
import random

import h5py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from ai_model import UAVEstimator
from feature_extraction import FeatureExtractor

SEED = 42
RANGE_MAX = 500.0
VEL_MAX = 50.0
DOA_MAX = 90.0
CARRIER_FREQ = 6e9
N_ANTENNAS = 8
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UAVDataset(Dataset):
    """Precomputed range, Doppler and array-DOA features with no label leakage."""

    def __init__(self, files):
        if not files:
            raise FileNotFoundError("No dataset files found.")

        self.X, self.TX, self.y = [], [], []
        self.fe = FeatureExtractor(carrier_freq=CARRIER_FREQ, n_subcarriers=128, n_symbols=14)

        for f in files:
            with h5py.File(f, "r") as hf:
                x = hf["X"][:]
                tx = hf["TX"][:]
                y = hf["y"][:]
            if x.ndim != 3 or x.shape[1] != N_ANTENNAS:
                raise ValueError(f"{f}: expected X=[samples,{N_ANTENNAS},time], got {x.shape}")
            if tx.ndim != 2 or len(tx) != len(x):
                raise ValueError(f"{f}: expected TX=[samples,time], got {tx.shape}")
            self.X.append(x)
            self.TX.append(tx)
            self.y.append(y.astype(np.float32))

        self.X = np.concatenate(self.X, axis=0)
        self.TX = np.concatenate(self.TX, axis=0)
        self.y_raw = np.concatenate(self.y, axis=0)
        self.y = self.y_raw.copy()
        self.y[:, 0] /= RANGE_MAX
        self.y[:, 1] /= VEL_MAX
        self.y[:, 2] /= DOA_MAX

        n = len(self.X)
        self.range_features = np.empty((n, 128), dtype=np.float32)
        self.doppler_features = np.empty((n, 128), dtype=np.float32)
        self.doa_features = np.empty((n, 181), dtype=np.float32)
        self.estimated_doa = np.empty(n, dtype=np.float32)

        print(f"Precomputing signal features for {n:,} samples...")
        for i, (rx_array, tx_signal) in enumerate(zip(self.X, self.TX)):
            feats = self.fe.extract_features(rx_array[0], rx_array, tx_signal)
            self.range_features[i] = feats["range"]
            self.doppler_features[i] = feats["doppler"]
            self.doa_features[i] = feats["doa"]
            self.estimated_doa[i] = feats["doa_deg"] / DOA_MAX

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.range_features[idx]),
            torch.from_numpy(self.doppler_features[idx]),
            torch.from_numpy(self.doa_features[idx]),
            torch.from_numpy(self.y[idx]),
            torch.tensor(self.estimated_doa[idx], dtype=torch.float32),
        )


def train_model(files, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    set_seed()
    dataset = UAVDataset(files)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                             pin_memory=torch.cuda.is_available())

    model = UAVEstimator().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = torch.nn.MSELoss()
    history = {"train_loss": [], "test_loss": []}
    best_loss = float("inf")

    print(f"Device: {device}")
    print(f"Train: {len(train_set):,} | Test: {len(test_set):,}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for r, d, doa, labels, _ in train_loader:
            r, d, doa, labels = r.to(device), d.to(device), doa.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(r, d, doa), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for r, d, doa, labels, _ in test_loader:
                preds = model(r.to(device), d.to(device), doa.to(device))
                test_loss += criterion(preds, labels.to(device)).item()
        test_loss /= max(len(test_loader), 1)
        scheduler.step(test_loss)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        print(f"Epoch {epoch+1:02d}/{epochs} | Train={train_loss:.6f} | Test={test_loss:.6f} | LR={optimizer.param_groups[0]['lr']:.2e}")
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "adyant_uav_estimator.pth")

    np.savez("training_history.npz", **history)
    print("Best model saved as adyant_uav_estimator.pth")
    print("Training history saved as training_history.npz")


if __name__ == "__main__":
    files = sorted(glob.glob("datasets/adyant_isac_*.h5"))
    print(f"Training on {len(files)} dataset files: {files}")
    train_model(files)
