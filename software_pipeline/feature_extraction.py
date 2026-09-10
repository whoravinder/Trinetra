import numpy as np
from scipy.constants import c
from scipy.fft import fft, ifft


class FeatureExtractor:
    """Physically motivated signal features for the Trinetra UAS prototype.

    Range uses matched filtering against the known transmitted OFDM frame.
    Doppler uses coherent slow-time processing across repeated OFDM symbols.
    DOA uses conventional beamforming across the full ULA.
    """

    def __init__(self, fs=20e6, n_fft=256, carrier_freq=6e9,
                 n_subcarriers=128, n_symbols=14, element_spacing=None):
        self.fs = fs
        self.n_fft = n_fft
        self.carrier_freq = carrier_freq
        self.n_subcarriers = n_subcarriers
        self.n_symbols = n_symbols
        self.symbol_len = n_subcarriers
        self.wavelength = c / carrier_freq
        self.element_spacing = (
            element_spacing if element_spacing is not None else self.wavelength / 2
        )

    def range_profile(self, rx_signal, tx_signal):
        """Matched-filter range profile; first bins correspond to positive delay."""
        rx = np.asarray(rx_signal, dtype=np.complex128)
        tx = np.asarray(tx_signal, dtype=np.complex128)
        n = 1 << int(np.ceil(np.log2(len(rx) + len(tx) - 1)))
        corr = ifft(fft(rx, n=n) * np.conj(fft(tx, n=n)))
        profile = np.abs(corr[:self.n_subcarriers])
        profile /= max(np.max(profile), 1e-12)
        return profile.astype(np.float32)

    def doppler_spectrum(self, rx_signal, tx_signal, range_profile=None):
        """Estimate Doppler from coherent phase progression over OFDM symbols."""
        rx = np.asarray(rx_signal, dtype=np.complex128)
        tx = np.asarray(tx_signal, dtype=np.complex128)
        if len(rx) != len(tx):
            raise ValueError("rx_signal and tx_signal must have the same length")

        if range_profile is None:
            range_profile = self.range_profile(rx, tx)
        delay_samples = int(np.argmax(range_profile))
        aligned = np.roll(rx, -delay_samples)

        slow_time = []
        for k in range(self.n_symbols):
            start = k * self.symbol_len
            stop = start + self.symbol_len
            if stop > len(rx):
                break
            # Matched coherent sum for each repeated symbol.
            value = np.vdot(tx[start:stop], aligned[start:stop])
            slow_time.append(value)

        slow_time = np.asarray(slow_time, dtype=np.complex128)
        if len(slow_time) == 0:
            return np.zeros(128, dtype=np.float32)

        # Zero-pad slow-time FFT to obtain a smooth Doppler spectrum.
        doppler = np.abs(np.fft.fftshift(np.fft.fft(slow_time, n=128)))
        doppler /= max(np.max(doppler), 1e-12)
        return doppler.astype(np.float32)

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

        r = (x @ x.conj().T) / max(x.shape[1], 1)
        r = (r + r.conj().T) / 2

        element_idx = np.arange(m)[:, None]
        phase = (
            2.0 * np.pi * self.element_spacing * element_idx
            * np.sin(np.deg2rad(angles))[None, :] / self.wavelength
        )
        steering = np.exp(-1j * phase)
        r_steering = r @ steering
        powers = np.real(np.sum(steering.conj() * r_steering, axis=0)) / (m * m)
        powers = np.maximum(powers, 0.0)
        powers /= max(np.max(powers), 1e-12)
        return angles, powers.astype(np.float32)

    def doa_from_ula(self, rx_array):
        angles, powers = self.doa_beamforming_spectrum(rx_array)
        return float(angles[int(np.argmax(powers))])

    def extract_features(self, rx_signal, rx_array, tx_signal):
        """Extract range, Doppler and DOA without using target labels."""
        range_feat = self.range_profile(rx_signal, tx_signal)
        doppler_feat = self.doppler_spectrum(rx_signal, tx_signal, range_feat)
        angles, doa_spectrum = self.doa_beamforming_spectrum(rx_array)
        estimated_doa = float(angles[int(np.argmax(doa_spectrum))])

        return {
            "range": range_feat,
            "doppler": doppler_feat,
            "doa": doa_spectrum.astype(np.float32),
            "doa_angles": angles.astype(np.float32),
            "doa_deg": estimated_doa,
        }
