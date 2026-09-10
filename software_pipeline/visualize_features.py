import h5py
import matplotlib.pyplot as plt
from feature_extraction import FeatureExtractor

with h5py.File("adyant_isac_dataset.h5", "r") as f:
    X = f["X"][:10]   
    y = f["y"][:10]

fe = FeatureExtractor()

for i, rx in enumerate(X):
    label = y[i]
    feats = fe.extract_features(rx, doa_deg=label[2])

    
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(feats["range"])
    plt.title(f"Range Profile\nTrue Range={label[0]:.1f} m")
    plt.xlabel("FFT Bin")
    plt.ylabel("Magnitude")

    
    plt.subplot(1,3,2)
    plt.imshow(feats["doppler"], aspect='auto', origin='lower')
    plt.title(f"Doppler Spectrum\nTrue Velocity={label[1]:.1f} m/s")
    plt.xlabel("Time Frames")
    plt.ylabel("Doppler Bins")

    
    plt.subplot(1,3,3)
    plt.plot(feats["doa"])
    plt.title(f"DOA Encoding\nTrue DOA={label[2]:.1f}°")
    plt.xlabel("Angle Bin (-90° to 90°)")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()
