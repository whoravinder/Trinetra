import numpy as np
from scipy.constants import c
from scipy.fft import fft
from scipy.signal import stft


class FeatureExtractor:
    """Signal features for the Trinetra UAS parameter-estimation pipeline.

    Range/Doppler are extracted from one calibrated receiver channel so the
    original signal-processing path is preserved. DOA is independently
    estimated from the complete ULA using conventional beamforming.
    """

    def __init__(self, fs=20e6, n_fft=256, carrier_freq=6e9, element_spacing=None):
        self.fs = fs
        self.n_fft = n_fft
        self.carrier_freq = carrier_freq
        self.wavelength = c / carrier_freq
        self.element_spacing = (
            element_spacing if element_spacing is not None else self.wavelength / 2
        )

    def range_profile(self, rx_signal):
        spectrum = np.abs(fft(rx_signal, n=self.n_fft))
        spectrum = spectrum[: self.n_fft // 2]
        peak = np.max(spectrum)
        return (spectrum / max(peak, 1e-12)).astype(np.float32)

    def doppler_spectrum(self, rx_signal, window=128, overlap=64):
        _, _, zxx = stft(
            rx_signal, fs=self.fs, nperseg=window, noverlap=overlap, boundary=None
        )
        doppler_map = np.abs(zxx)
        peak = np.max(doppler_map)
        return (doppler_map / max(peak, 1e-12)).astype(np.float32)

    def doa_beamforming_spectrum(self, rx_array, angles=None):
        """Vectorized conventional beamforming for an M-element ULA."""
        x = np.asarray(rx_array, dtype=np.complex128)
        if x.ndim != 2:
            raise ValueError("rx_array must have shape [num_antennas, num_samples]")
        m = x.shape[0]
        if m < 2:
            raise ValueError("At least 2 antenna elements are required")

        if angles is None:
            angles = np.linspace(-90.0, 90.0, 181)
        angles = np.asarray(angles, dtype=np.float64)

        # Spatial covariance matrix.
        r = (x @ x.conj().T) / max(x.shape[1], 1)
        r = (r + r.conj().T) / 2

        element_idx = np.arange(m)[:, None]
        phase = (
            2.0
            * np.pi
            * self.element_spacing
            * element_idx
            * np.sin(np.deg2rad(angles))[None, :]
            / self.wavelength
        )
        steering = np.exp(-1j * phase)  # [M, angles]
        r_steering = r @ steering
        powers = np.real(np.sum(steering.conj() * r_steering, axis=0)) / (m * m)
        powers = np.maximum(powers, 0.0)
        powers /= max(np.max(powers), 1e-12)
        return angles, powers.astype(np.float32)

    def doa_from_ula(self, rx_array):
        angles, powers = self.doa_beamforming_spectrum(rx_array)
        return float(angles[int(np.argmax(powers))])

    def extract_features(self, rx_signal, rx_array):
        """Extract all model inputs without using the ground-truth DOA."""
        angles, doa_spectrum = self.doa_beamforming_spectrum(rx_array)
        estimated_doa = float(angles[int(np.argmax(doa_spectrum))])

        return {
            "range": self.range_profile(rx_signal),
            "doppler": self.doppler_spectrum(rx_signal),
            # The complete beamforming spectrum is more informative than a
            # hard one-hot angle and lets the neural network learn corrections.
            "doa": doa_spectrum.astype(np.float32),
            "doa_angles": angles.astype(np.float32),
            "doa_deg": estimated_doa,
        }
