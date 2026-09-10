import numpy as np
from scipy.signal import stft
from scipy.fft import fft

class FeatureExtractor:
    def __init__(self, fs=20e6, n_fft=256):
        self.fs = fs
        self.n_fft = n_fft

    def range_profile(self, rx_signal):
        spectrum = np.abs(fft(rx_signal, n=self.n_fft))
        spectrum = spectrum[:self.n_fft // 2]
        return spectrum / max(np.max(spectrum), 1e-12)

    def doppler_spectrum(self, rx_signal, window=128, overlap=64):
        f, t, Zxx = stft(rx_signal, fs=self.fs, nperseg=window, noverlap=overlap)
        doppler_map = np.abs(Zxx)
        return doppler_map / max(np.max(doppler_map), 1e-12)

    def doa_from_ula(self, rx_array, wavelength=0.03, element_spacing=None):
        """Estimate DOA from a uniform linear array using conventional beamforming.

        rx_array: complex array shaped [num_antennas, num_samples].
        Returns angle in degrees over [-90, 90].

        This is a practical baseline. For multiple targets/noisy environments,
        MUSIC/ESPRIT or a learned array-processing stage can be added later.
        """
        x = np.asarray(rx_array, dtype=np.complex128)
        if x.ndim != 2:
            raise ValueError("rx_array must have shape [num_antennas, num_samples]")
        m = x.shape[0]
        if m < 2:
            raise ValueError("At least 2 antenna elements are required for DOA estimation")

        d = element_spacing if element_spacing is not None else wavelength / 2
        angles = np.linspace(-90.0, 90.0, 181)
        # Average spatial covariance across fast-time samples.
        R = (x @ x.conj().T) / x.shape[1]
        R = (R + R.conj().T) / 2
        powers = np.empty_like(angles)
        element_idx = np.arange(m)
        for i, angle in enumerate(np.deg2rad(angles)):
            phase = 2.0 * np.pi * d * element_idx * np.sin(angle) / wavelength
            steering = np.exp(-1j * phase)
            powers[i] = np.real(np.conj(steering) @ R @ steering)

        return float(angles[int(np.argmax(powers))])

    def doa_one_hot(self, doa_deg, n_bins=180):
        doa_deg = float(np.clip(doa_deg, -90.0, 90.0))
        doa_map = np.zeros(n_bins, dtype=np.float32)
        idx = int(round((doa_deg + 90.0) / 180.0 * (n_bins - 1)))
        doa_map[idx] = 1.0
        return doa_map

    def extract_features(self, rx_signal, doa_deg=None, rx_array=None):
        rp = self.range_profile(rx_signal)
        ds = self.doppler_spectrum(rx_signal)

        if rx_array is not None:
            doa_deg = self.doa_from_ula(rx_array)
        if doa_deg is None:
            raise ValueError("Provide either doa_deg or rx_array")

        return {
            "range": rp.astype(np.float32),
            "doppler": ds.astype(np.float32),
            "doa": self.doa_one_hot(doa_deg),
            "doa_deg": float(doa_deg),
        }
