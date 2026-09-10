import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import UAVDataset, UAVEstimator, DOA_MAX


def evaluate_model(files, model_file="adyant_uav_estimator.pth"):
    if not files:
        raise FileNotFoundError("No datasets found in datasets/adyant_isac_*.h5")

    print("Loading datasets...")
    dataset = UAVDataset(files)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Loading trained model...")
    model = UAVEstimator()
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()

    preds, truths, doa_inputs = [], [], []
    with torch.no_grad():
        for range_feat, doppler_feat, doa_feat, labels, estimated_doa in loader:
            out = model(range_feat, doppler_feat, doa_feat)
            preds.append(out.numpy()[0])
            truths.append(labels.numpy()[0])
            doa_inputs.append(float(estimated_doa.numpy()[0]))

    preds = np.asarray(preds)
    truths = np.asarray(truths)
    doa_inputs = np.asarray(doa_inputs)

    # Model outputs and labels are normalized during training.
    preds[:, 0] *= 500.0
    preds[:, 1] *= 50.0
    preds[:, 2] *= DOA_MAX
    truths[:, 0] *= 500.0
    truths[:, 1] *= 50.0
    truths[:, 2] *= DOA_MAX

    range_rmse = np.sqrt(np.mean((preds[:, 0] - truths[:, 0]) ** 2))
    vel_mae = np.mean(np.abs(preds[:, 1] - truths[:, 1]))
    doa_mae = np.mean(np.abs(preds[:, 2] - truths[:, 2]))
    beamforming_doa_mae = np.mean(np.abs(doa_inputs - truths[:, 2]))

    print("\nEvaluation Metrics")
    print(f"Range RMSE             : {range_rmse:.2f} m")
    print(f"Velocity MAE           : {vel_mae:.2f} m/s")
    print(f"Model DOA MAE          : {doa_mae:.2f} deg")
    print(f"Beamforming DOA MAE    : {beamforming_doa_mae:.2f} deg")

    df = pd.DataFrame({
        "True Range (m)": truths[:, 0],
        "Pred Range (m)": preds[:, 0],
        "True Vel (m/s)": truths[:, 1],
        "Pred Vel (m/s)": preds[:, 1],
        "True DOA (deg)": truths[:, 2],
        "Beamforming DOA (deg)": doa_inputs,
        "Model DOA (deg)": preds[:, 2],
    })
    print("\nSample Predictions vs Ground Truth:")
    print(df.head(10).to_string(index=False))

    # Post-training prediction plots.
    fig, axs = plt.subplots(3, 1, figsize=(11, 9))
    axs[0].plot(truths[:, 0], label="True Range")
    axs[0].plot(preds[:, 0], "--", label="Predicted Range")
    axs[0].set_ylabel("Range (m)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(truths[:, 1], label="True Velocity")
    axs[1].plot(preds[:, 1], "--", label="Predicted Velocity")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(truths[:, 2], label="True DOA")
    axs[2].plot(doa_inputs, ":", label="Beamforming DOA")
    axs[2].plot(preds[:, 2], "--", label="Model DOA")
    axs[2].set_ylabel("DOA (deg)")
    axs[2].set_xlabel("Test sample")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # True vs predicted scatter plots.
    fig, axs = plt.subplots(3, 1, figsize=(8, 14))
    for ax, true, pred, title, unit in [
        (axs[0], truths[:, 0], preds[:, 0], "Range: True vs Predicted", "m"),
        (axs[1], truths[:, 1], preds[:, 1], "Velocity: True vs Predicted", "m/s"),
        (axs[2], truths[:, 2], preds[:, 2], "DOA: True vs Predicted", "deg"),
    ]:
        ax.scatter(true, pred, s=10, alpha=0.5)
        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "--", label="Ideal")
        ax.set_xlabel(f"True ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2D trajectory derived from range and DOA.
    true_x = truths[:, 0] * np.cos(np.deg2rad(truths[:, 2]))
    true_y = truths[:, 0] * np.sin(np.deg2rad(truths[:, 2]))
    pred_x = preds[:, 0] * np.cos(np.deg2rad(preds[:, 2]))
    pred_y = preds[:, 0] * np.sin(np.deg2rad(preds[:, 2]))

    plt.figure(figsize=(7, 7))
    plt.plot(true_x, true_y, "-o", markersize=2, label="True Trajectory")
    plt.plot(pred_x, pred_y, "--o", markersize=2, label="Predicted Trajectory")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("UAV Trajectory: True vs Predicted")
    plt.axis("equal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    files = sorted(glob.glob("datasets/adyant_isac_*.h5"))
    print(f"Found {len(files)} dataset files")
    evaluate_model(files)
