import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
from feature_extraction import FeatureExtractor


files = sorted(glob.glob("datasets/adyant_isac_*.h5"))
if not files:
    raise FileNotFoundError("No datasets found in datasets/adyant_isac_*.h5")

fe = FeatureExtractor(carrier_freq=6e9)

for file in files:
    with h5py.File(file, "r") as f:
        X = f["X"][:3]
        y = f["y"][:3]

    for i, rx_array in enumerate(X):
        true_doa = float(y[i, 2])
        estimated_doa = fe.doa_from_ula(rx_array)
        rx_combined = np.mean(rx_array, axis=0)
        feats = fe.extract_features(rx_combined, doa_deg=estimated_doa)
        angles, powers = fe.doa_beamforming_spectrum(rx_array)
        powers_db = 10 * np.log10(np.maximum(powers, 1e-12) / np.max(np.maximum(powers, 1e-12)))

        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

        axs[0].plot(feats["range"])
        axs[0].set_title(f"Range Profile\nTrue Range={y[i, 0]:.1f} m")
        axs[0].set_xlabel("FFT Bin")
        axs[0].set_ylabel("Normalized Magnitude")
        axs[0].grid(True, alpha=0.3)

        axs[1].imshow(feats["doppler"], aspect="auto", origin="lower")
        axs[1].set_title(f"Doppler Spectrum\nTrue Velocity={y[i, 1]:.1f} m/s")
        axs[1].set_xlabel("Time Frames")
        axs[1].set_ylabel("Frequency Bins")

        axs[2].plot(angles, powers_db)
        axs[2].axvline(true_doa, linestyle="--", label=f"True: {true_doa:.1f}°")
        axs[2].axvline(estimated_doa, linestyle=":", label=f"Estimated: {estimated_doa:.1f}°")
        axs[2].set_title("ULA Beamforming DOA")
        axs[2].set_xlabel("Angle (degrees)")
        axs[2].set_ylabel("Relative Power (dB)")
        axs[2].set_xlim(-90, 90)
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)

        plt.suptitle(f"Feature Visualization — {file.split('/')[-1]} — Sample {i}")
        plt.tight_layout()
        plt.show()
