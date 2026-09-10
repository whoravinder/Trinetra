import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from feature_extraction import FeatureExtractor

files = sorted(glob.glob("datasets/adyant_isac_*.h5"))
if not files:
    raise FileNotFoundError("No datasets found. Run generate_dataset.py first.")

os.makedirs("results", exist_ok=True)
fe = FeatureExtractor(carrier_freq=6e9)

for file in files:
    with h5py.File(file, "r") as f:
        rx_array = f["X"][0]
        label = f["y"][0]

    # One reference antenna for temporal features; all antennas for spatial DOA.
    feats = fe.extract_features(rx_array[0], rx_array)
    angles, powers = fe.doa_beamforming_spectrum(rx_array)
    estimated_doa = feats["doa_deg"]
    powers_db = 10 * np.log10(np.maximum(powers, 1e-12))
    powers_db -= powers_db.max()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].plot(feats["range"])
    axs[0].set_title(f"Range Profile | True={label[0]:.1f} m")
    axs[0].set_xlabel("FFT bin")
    axs[0].set_ylabel("Normalized magnitude")
    axs[0].grid(True, alpha=0.3)

    axs[1].imshow(feats["doppler"], aspect="auto", origin="lower")
    axs[1].set_title(f"Doppler | True={label[1]:.1f} m/s")
    axs[1].set_xlabel("Time frame")
    axs[1].set_ylabel("Frequency bin")

    axs[2].plot(angles, powers_db)
    axs[2].axvline(label[2], linestyle="--", label=f"True: {label[2]:.1f}°")
    axs[2].axvline(estimated_doa, linestyle=":", label=f"Beamforming: {estimated_doa:.1f}°")
    axs[2].set_title("8-Element ULA Beamforming")
    axs[2].set_xlabel("DOA (degrees)")
    axs[2].set_ylabel("Relative power (dB)")
    axs[2].set_xlim(-90, 90)
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    fig.suptitle(f"Trinetra Sensor Features — {os.path.basename(file)}")
    fig.tight_layout()
    output = os.path.join("results", os.path.splitext(os.path.basename(file))[0] + "_features.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved: {output}")
    print(f"True DOA={label[2]:.2f}°, Beamforming DOA={estimated_doa:.2f}°, Error={abs(label[2]-estimated_doa):.2f}°")
