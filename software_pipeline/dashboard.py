import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from ai_model import UAVEstimator
from train import DOA_MAX, RANGE_MAX, SEED, UAVDataset, VEL_MAX


OUTPUT_DIR = "results"


def evaluate_model(files, model_file="adyant_uav_estimator.pth"):
    if not files:
        raise FileNotFoundError("No datasets found in datasets/adyant_isac_*.h5")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}. Run train.py first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading dataset and precomputing features...")
    dataset = UAVDataset(files)

    # Use the same deterministic 80/20 split as training for an honest test report.
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_set = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(SEED)
    )
    loader = DataLoader(test_set, batch_size=64, shuffle=False)

    print("Loading trained model...")
    model = UAVEstimator()
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()

    preds, truths, doa_inputs = [], [], []
    with torch.no_grad():
        for r, d, doa, labels, estimated_doa in loader:
            out = model(r, d, doa)
            preds.append(out.numpy())
            truths.append(labels.numpy())
            doa_inputs.append(estimated_doa.numpy())

    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    doa_inputs = np.concatenate(doa_inputs) * DOA_MAX

    # Convert normalized values back to physical units.
    preds[:, 0] *= RANGE_MAX
    preds[:, 1] *= VEL_MAX
    preds[:, 2] *= DOA_MAX
    truths[:, 0] *= RANGE_MAX
    truths[:, 1] *= VEL_MAX
    truths[:, 2] *= DOA_MAX

    range_rmse = np.sqrt(np.mean((preds[:, 0] - truths[:, 0]) ** 2))
    velocity_mae = np.mean(np.abs(preds[:, 1] - truths[:, 1]))
    model_doa_mae = np.mean(np.abs(preds[:, 2] - truths[:, 2]))
    beamforming_doa_mae = np.mean(np.abs(doa_inputs - truths[:, 2]))

    print("\n=== Trinetra UAS Parameter Estimation — Test Results ===")
    print(f"Test samples            : {len(truths)}")
    print(f"Range RMSE              : {range_rmse:.2f} m")
    print(f"Velocity MAE            : {velocity_mae:.2f} m/s")
    print(f"Model DOA MAE           : {model_doa_mae:.2f} deg")
    print(f"Beamforming DOA MAE     : {beamforming_doa_mae:.2f} deg")

    df = pd.DataFrame({
        "True Range (m)": truths[:, 0], "Pred Range (m)": preds[:, 0],
        "True Velocity (m/s)": truths[:, 1], "Pred Velocity (m/s)": preds[:, 1],
        "True DOA (deg)": truths[:, 2], "Beamforming DOA (deg)": doa_inputs,
        "Model DOA (deg)": preds[:, 2],
    })
    df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    print("\nFirst 10 test predictions:")
    print(df.head(10).to_string(index=False))

    # 1. Prediction traces: ideal for a technical demonstration.
    fig, axs = plt.subplots(3, 1, figsize=(11, 9))
    items = [
        (truths[:, 0], preds[:, 0], "Range (m)", "True Range", "Model Range"),
        (truths[:, 1], preds[:, 1], "Velocity (m/s)", "True Velocity", "Model Velocity"),
        (truths[:, 2], preds[:, 2], "DOA (deg)", "True DOA", "Model DOA"),
    ]
    for ax, (true, pred, ylabel, a, b) in zip(axs, items):
        ax.plot(true, label=a)
        ax.plot(pred, "--", label=b)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axs[-1].set_xlabel("Test sample")
    fig.suptitle("Trinetra: True vs Predicted UAS Parameters")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "prediction_traces.png"), dpi=180)
    plt.show()

    # 2. DOA comparison: classical beamforming vs learned correction.
    plt.figure(figsize=(8, 6))
    plt.scatter(truths[:, 2], doa_inputs, s=12, alpha=0.45, label="Beamforming")
    plt.scatter(truths[:, 2], preds[:, 2], s=12, alpha=0.45, label="Neural model")
    lo, hi = -90, 90
    plt.plot([lo, hi], [lo, hi], "--", label="Ideal")
    plt.xlabel("True DOA (deg)")
    plt.ylabel("Estimated DOA (deg)")
    plt.title("DOA: Classical Baseline vs Neural Estimator")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "doa_comparison.png"), dpi=180)
    plt.show()

    # 3. Range/velocity/DOA regression quality.
    fig, axs = plt.subplots(3, 1, figsize=(8, 14))
    for ax, true, pred, title, unit in [
        (axs[0], truths[:, 0], preds[:, 0], "Range: True vs Predicted", "m"),
        (axs[1], truths[:, 1], preds[:, 1], "Velocity: True vs Predicted", "m/s"),
        (axs[2], truths[:, 2], preds[:, 2], "DOA: True vs Predicted", "deg"),
    ]:
        ax.scatter(true, pred, s=10, alpha=0.5)
        lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "--", label="Ideal")
        ax.set_xlabel(f"True ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "regression_scatter.png"), dpi=180)
    plt.show()

    # 4. 2D position proxy from range + DOA.
    true_x = truths[:, 0] * np.cos(np.deg2rad(truths[:, 2]))
    true_y = truths[:, 0] * np.sin(np.deg2rad(truths[:, 2]))
    pred_x = preds[:, 0] * np.cos(np.deg2rad(preds[:, 2]))
    pred_y = preds[:, 0] * np.sin(np.deg2rad(preds[:, 2]))

    plt.figure(figsize=(7, 7))
    plt.plot(true_x, true_y, "-o", markersize=2, label="True")
    plt.plot(pred_x, pred_y, "--o", markersize=2, label="Predicted")
    plt.xlabel("X position proxy (m)")
    plt.ylabel("Y position proxy (m)")
    plt.title("UAV Position Proxy: True vs Predicted")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "trajectory_proxy.png"), dpi=180)
    plt.show()

    # 5. Training/validation curve, if produced by train.py.
    history_file = "training_history.npz"
    if os.path.exists(history_file):
        history = np.load(history_file)
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="Train loss")
        plt.plot(history["test_loss"], label="Validation/test loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.title("Trinetra Training Convergence")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "training_curve.png"), dpi=180)
        plt.show()

    print(f"\nSaved plots and test_predictions.csv to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    files = sorted(glob.glob("datasets/adyant_isac_*.h5"))
    print(f"Found {len(files)} dataset files")
    evaluate_model(files)
