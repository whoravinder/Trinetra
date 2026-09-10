import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import UAVDataset, UAVEstimator
import glob
import pandas as pd
from tqdm import tqdm
import time

def evaluate_model(files, model_file="adyant_uav_estimator.pth"):
    print("Loading datasets...")
    dataset = UAVDataset(files)

    print("Preparing DataLoader...")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Initializing model...")
    model = UAVEstimator()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    preds, truths = [], []
    print("Running inference...")
    for range_feat, doppler_feat, doa_feat, labels in tqdm(loader, desc="Evaluating"):
        with torch.no_grad():
            out = model(range_feat, doppler_feat, doa_feat)
            preds.append(out.numpy()[0])
            truths.append(labels.numpy()[0])

    preds = np.array(preds)
    truths = np.array(truths)

    print("Computing metrics...")
    range_rmse = np.sqrt(np.mean((preds[:,0] - truths[:,0])**2))
    vel_mae = np.mean(np.abs(preds[:,1] - truths[:,1]))
    doa_mae = np.mean(np.abs(preds[:,2] - truths[:,2]))
    print("\nEvaluation Metrics")
    print(f"Range RMSE   : {range_rmse:.2f} m")
    print(f"Velocity MAE : {vel_mae:.2f} m/s")
    print(f"DOA MAE      : {doa_mae:.2f} °")

    df = pd.DataFrame({
        "True Range (m)": truths[:,0],
        "Pred Range (m)": preds[:,0],
        "True Vel (m/s)": truths[:,1],
        "Pred Vel (m/s)": preds[:,1],
        "True DOA (°)": truths[:,2],
        "Pred DOA (°)": preds[:,2],
    })
    print("\nSample Predictions vs Ground Truth (first 10 samples):")
    print(df.head(10).to_string(index=False))

    print("Plotting results...")
    time.sleep(1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    axs[0].plot(truths[:,0], label="True Range")
    axs[0].plot(preds[:,0], "--", label="Pred Range")
    axs[0].legend()
    axs[1].plot(truths[:,1], label="True Velocity")
    axs[1].plot(preds[:,1], "--", label="Pred Velocity")
    axs[1].legend()
    axs[2].plot(truths[:,2], label="True DOA")
    axs[2].plot(preds[:,2], "--", label="Pred DOA")
    axs[2].legend()
    plt.tight_layout()
    plt.show()

    true_x = truths[:,0] * np.cos(np.deg2rad(truths[:,2]))
    true_y = truths[:,0] * np.sin(np.deg2rad(truths[:,2]))
    pred_x = preds[:,0] * np.cos(np.deg2rad(preds[:,2]))
    pred_y = preds[:,0] * np.sin(np.deg2rad(preds[:,2]))
    plt.figure(figsize=(6,6))
    plt.plot(true_x, true_y, "b-o", label="True Trajectory")
    plt.plot(pred_x, pred_y, "r--o", label="Pred Trajectory")
    plt.axis("equal")
    plt.legend()
    plt.title("UAV Trajectory (True vs Predicted)")
    plt.show()

if __name__ == "__main__":
    print("Searching for datasets...")
    files = glob.glob("datasets/adyant_isac_*.h5")
    print(f"Found {len(files)} dataset files")
    evaluate_model(files)
