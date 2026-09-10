import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import numpy as np
import h5py
import glob
from feature_extraction import FeatureExtractor
from ai_model import UAVEstimator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANGE_MAX = 500.0
VEL_MAX = 50.0
DOA_MAX = 90.0

class UAVDataset(Dataset):
    def __init__(self, files):
        self.X, self.y = [], []
        self.fe = FeatureExtractor()
        for f in files:
            with h5py.File(f, "r") as hf:
                Xd = hf["X"][:]
                Yd = hf["y"][:]
                Yd[:,0] /= RANGE_MAX
                Yd[:,1] /= VEL_MAX
                Yd[:,2] /= DOA_MAX
                self.X.append(Xd)
                self.y.append(Yd)
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        rx = self.X[idx]
        label = self.y[idx]
        feats = self.fe.extract_features(rx, doa_deg=label[2]*DOA_MAX)
        range_feat = torch.tensor(feats["range"], dtype=torch.float32)
        doppler_feat = torch.tensor(feats["doppler"], dtype=torch.float32)
        doa_feat = torch.tensor(feats["doa"], dtype=torch.float32)
        return range_feat, doppler_feat, doa_feat, torch.tensor(label, dtype=torch.float32)

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
        train_loss = 0
        for range_feat, doppler_feat, doa_feat, labels in train_loader:
            range_feat, doppler_feat, doa_feat, labels = \
                range_feat.to(device), doppler_feat.to(device), doa_feat.to(device), labels.to(device)
            preds = model(range_feat, doppler_feat, doa_feat)
            loss = criterion(preds, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval(); test_loss = 0
        with torch.no_grad():
            for range_feat, doppler_feat, doa_feat, labels in test_loader:
                range_feat, doppler_feat, doa_feat, labels = \
                    range_feat.to(device), doppler_feat.to(device), doa_feat.to(device), labels.to(device)
                preds = model(range_feat, doppler_feat, doa_feat)
                loss = criterion(preds, labels)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        scheduler.step(test_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss={train_loss:.6f} | Test Loss={test_loss:.6f} | LR={optimizer.param_groups[0]['lr']:.6f}")

    torch.save(model.state_dict(), "adyant_uav_estimator.pth")
    print("Model trained and saved as adyant_uav_estimator.pth")

if __name__ == "__main__":
    files = glob.glob("datasets/adyant_isac_*.h5")
    print(f"Training on {len(files)} scenario datasets: {files}")
    train_model(files, epochs=15, batch_size=32, lr=1e-4)
