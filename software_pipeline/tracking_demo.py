import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from ai_model import UAVEstimator
from feature_extraction import FeatureExtractor
from generate_dataset import ISACDatasetGenerator3GPP
from tracking import PolarKalmanTracker

MODEL_FILE = "adyant_uav_estimator.pth"
OUTPUT_DIR = "results"
N_FRAMES = 80
DT = 0.1
RANGE_MAX = 500.0
VEL_MAX = 50.0
DOA_MAX = 90.0


def make_trajectory(n_frames=N_FRAMES, dt=DT):
    """Create one smooth 2-D UAV trajectory and derive polar states."""
    position = np.array([350.0, -200.0], dtype=float)
    velocity = np.array([10.0, 10.0], dtype=float)
    states = []
    for k in range(n_frames):
        p = position + velocity * (k * dt)
        r = np.linalg.norm(p)
        radial_velocity = np.dot(p, velocity) / max(r, 1e-9)
        bearing = np.degrees(np.arctan2(p[1], p[0]))
        states.append([r, radial_velocity, bearing])
    return np.asarray(states, dtype=np.float32)


def main():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"{MODEL_FILE} not found. Run train.py first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UAVEstimator().to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    generator = ISACDatasetGenerator3GPP(
        carrier_freq=6e9,
        scenario="UMi",
        n_antennas=8,
    )
    feature_extractor = FeatureExtractor(carrier_freq=6e9, n_subcarriers=128, n_symbols=14)
    tracker = PolarKalmanTracker(dt=DT, range_std=8.0, velocity_std=2.5, bearing_std=3.0)

    true_states = make_trajectory()
    measurements = []
    filtered = []

    with torch.no_grad():
        for true_range, true_velocity, true_bearing in true_states:
            tx, rx, _ = generator.generate_sample(
                float(true_range), float(true_velocity), float(true_bearing)
            )
            feats = feature_extractor.extract_features(rx[0], rx, tx)
            r = torch.from_numpy(feats["range"]).unsqueeze(0).to(device)
            d = torch.from_numpy(feats["doppler"]).unsqueeze(0).to(device)
            doa = torch.from_numpy(feats["doa"]).unsqueeze(0).to(device)
            pred = model(r, d, doa).cpu().numpy()[0]
            measurement = np.array([
                pred[0] * RANGE_MAX,
                pred[1] * VEL_MAX,
                pred[2] * DOA_MAX,
            ], dtype=float)
            measurements.append(measurement)
            filtered.append(tracker.update(measurement))

    measurements = np.asarray(measurements)
    filtered = np.asarray(filtered)
    true_states = np.asarray(true_states)

    data = np.column_stack([np.arange(N_FRAMES), true_states, measurements, filtered])
    np.savetxt(
        os.path.join(OUTPUT_DIR, "tracking_demo.csv"),
        data,
        delimiter=",",
        header=(
            "frame,true_range_m,true_velocity_mps,true_bearing_deg,"
            "measured_range_m,measured_velocity_mps,measured_bearing_deg,"
            "filtered_range_m,filtered_velocity_mps,filtered_bearing_deg"
        ),
        comments="",
    )

    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    labels = [("Range (m)", 0), ("Radial velocity (m/s)", 1), ("Bearing (deg)", 2)]
    for ax, (ylabel, idx) in zip(axs, labels):
        ax.plot(true_states[:, idx], label="True")
        ax.plot(measurements[:, idx], ".--", markersize=3, alpha=0.55, label="Neural measurement")
        ax.plot(filtered[:, idx], linewidth=2, label="Kalman filtered")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axs[-1].set_xlabel("Frame")
    fig.suptitle("Trinetra Temporal Tracking: Neural Measurements + Kalman Filter")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "tracking_timeseries.png"), dpi=180)
    plt.show()

    def polar_to_xy(states):
        r = states[:, 0]
        a = np.deg2rad(states[:, 2])
        return r * np.cos(a), r * np.sin(a)

    true_x, true_y = polar_to_xy(true_states)
    meas_x, meas_y = polar_to_xy(measurements)
    filt_x, filt_y = polar_to_xy(filtered)

    plt.figure(figsize=(8, 7))
    plt.plot(true_x, true_y, "-o", markersize=3, label="True trajectory")
    plt.plot(meas_x, meas_y, ".--", markersize=4, alpha=0.5, label="Neural measurements")
    plt.plot(filt_x, filt_y, "-", linewidth=2, label="Kalman track")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("Trinetra UAV Track")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tracking_trajectory.png"), dpi=180)
    plt.show()

    print("\n=== Trinetra Temporal Tracking Demo ===")
    print(f"Frames                   : {N_FRAMES}")
    print(f"Frame interval           : {DT:.2f} s")
    print(f"Range measurement MAE    : {np.mean(np.abs(measurements[:, 0] - true_states[:, 0])):.2f} m")
    print(f"Range filtered MAE       : {np.mean(np.abs(filtered[:, 0] - true_states[:, 0])):.2f} m")
    print(f"Velocity measurement MAE : {np.mean(np.abs(measurements[:, 1] - true_states[:, 1])):.2f} m/s")
    print(f"Velocity filtered MAE    : {np.mean(np.abs(filtered[:, 1] - true_states[:, 1])):.2f} m/s")
    print(f"Bearing measurement MAE  : {np.mean(np.abs(measurements[:, 2] - true_states[:, 2])):.2f} deg")
    print(f"Bearing filtered MAE     : {np.mean(np.abs(filtered[:, 2] - true_states[:, 2])):.2f} deg")
    print(f"Saved: {OUTPUT_DIR}/tracking_demo.csv")
    print(f"Saved: {OUTPUT_DIR}/tracking_timeseries.png")
    print(f"Saved: {OUTPUT_DIR}/tracking_trajectory.png")


if __name__ == "__main__":
    main()
