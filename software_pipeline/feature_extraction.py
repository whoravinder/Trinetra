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
        return spectrum / np.max(spectrum)      

    def doppler_spectrum(self, rx_signal, window=128, overlap=64):
        
        f, t, Zxx = stft(rx_signal, fs=self.fs, nperseg=window, noverlap=overlap)
        doppler_map = np.abs(Zxx)
        doppler_map = doppler_map / np.max(doppler_map)
        return doppler_map

    def doa_placeholder(self, doa_deg, n_bins=180):
        
        doa_map = np.zeros(n_bins)
        idx = int((doa_deg + 90) / 180 * (n_bins - 1))
        doa_map[idx] = 1
        return doa_map

    def extract_features(self, rx_signal, doa_deg):
        rp = self.range_profile(rx_signal)
        ds = self.doppler_spectrum(rx_signal)
        doa = self.doa_placeholder(doa_deg)
        return {
            "range": rp.astype(np.float32),
            "doppler": ds.astype(np.float32),
            "doa": doa.astype(np.float32)
        }
