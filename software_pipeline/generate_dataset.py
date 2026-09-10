import numpy as np
import h5py
import os
from scipy.constants import c

class ISACDatasetGenerator3GPP:
    def __init__(self, carrier_freq=6e9, bandwidth=20e6, n_subcarriers=128, n_symbols=14, snr_db=20, n_clusters=10, scenario="UMi"):
        self.fc = carrier_freq
        self.bw = bandwidth
        self.n_subcarriers = n_subcarriers
        self.n_symbols = n_symbols
        self.snr_db = snr_db
        self.n_clusters = n_clusters
        self.scenario = scenario

    def generate_ofdm_symbol(self):
        data = np.random.choice([1, -1], size=self.n_subcarriers) + 1j*np.random.choice([1, -1], size=self.n_subcarriers)
        return np.fft.ifft(data)

    def generate_ofdm_frame(self):
        return np.hstack([self.generate_ofdm_symbol() for _ in range(self.n_symbols)])

    def los_probability(self, range_m):
        if self.scenario == "UMi":
            return np.exp(-range_m/100)
        elif self.scenario == "UMa":
            return np.exp(-range_m/200)
        elif self.scenario == "RMa":
            return np.exp(-range_m/500)
        elif self.scenario == "SMa":
            return np.exp(-range_m/300)
        return 0.5

    def generate_clusters(self, range_m, velocity_mps, doa_deg, is_target=True):
        clusters = []
        for _ in range(self.n_clusters):
            delay = range_m/c + np.random.uniform(0, 200e-9 if is_target else 500e-9)
            doppler_shift = (2*velocity_mps*self.fc/c) + np.random.uniform(-20, 20)
            angle = doa_deg + np.random.uniform(-15, 15)
            power = np.random.exponential(1.0 if is_target else 0.3)
            clusters.append((delay, doppler_shift, angle, power))
        return clusters

    def apply_channel(self, tx_signal, clusters):
        n = len(tx_signal)
        t = np.arange(n) / self.bw
        rx = np.zeros(n, dtype=complex)
        for delay, doppler_shift, angle, power in clusters:
            delayed = np.roll(tx_signal, int(delay * self.bw))
            doppler = delayed * np.exp(1j*2*np.pi*doppler_shift*t)
            rx += np.sqrt(power) * doppler
        return rx

    def apply_uav_channel(self, tx_signal, range_m, velocity_mps, doa_deg, uav_size="small"):
        rcs_factor = 1.0 if uav_size == "small" else 2.5
        clusters = self.generate_clusters(range_m, velocity_mps, doa_deg, is_target=True)
        rx = self.apply_channel(tx_signal, clusters) * rcs_factor
        return rx

    def apply_background_channel(self, tx_signal, range_m):
        los = np.random.rand() < self.los_probability(range_m)
        clusters = self.generate_clusters(range_m, velocity_mps=0, doa_deg=0, is_target=False)
        if los:
            clusters.append((range_m/c, 0, 0, 1.0))
        rx = self.apply_channel(tx_signal, clusters)
        return rx

    def apply_full_channel(self, tx_signal, range_m, velocity_mps, doa_deg, uav_size="small"):
        target_rx = self.apply_uav_channel(tx_signal, range_m, velocity_mps, doa_deg, uav_size)
        background_rx = self.apply_background_channel(tx_signal, range_m)
        rx = target_rx + background_rx
        sig_power = np.mean(np.abs(rx)**2)
        noise_power = sig_power / (10**(self.snr_db/10))
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(rx)) + 1j*np.random.randn(len(rx)))
        return rx + noise

    def generate_sample(self, range_m, velocity_mps, doa_deg, uav_size="small"):
        tx = self.generate_ofdm_frame()
        rx = self.apply_full_channel(tx, range_m, velocity_mps, doa_deg, uav_size)
        return rx, np.array([range_m, velocity_mps, doa_deg])

    def build_dataset(self, n_samples=5000, out_file="adyant_isac_dataset.h5"):
        X, y = [], []
        for _ in range(n_samples):
            r = np.random.uniform(10, 500)
            v = np.random.uniform(0, 50)
            a = np.random.uniform(-90, 90)
            size = "small" if np.random.rand() < 0.5 else "large"
            rx, label = self.generate_sample(r, v, a, size)
            X.append(rx)
            y.append(label)
        X = np.array(X, dtype=np.complex64)
        y = np.array(y, dtype=np.float32)
        os.makedirs("datasets", exist_ok=True)
        path = os.path.join("datasets", out_file)
        with h5py.File(path, "w") as f:
            f.create_dataset("X", data=X)
            f.create_dataset("y", data=y)
        print(f"Dataset generated: {path} with {n_samples} samples, scenario={self.scenario}")

if __name__ == "__main__":
    scenarios = ["UMi", "UMa", "RMa", "SMa"]
    for sc in scenarios:
        gen = ISACDatasetGenerator3GPP(carrier_freq=6e9, scenario=sc)
        gen.build_dataset(n_samples=5000, out_file=f"adyant_isac_{sc}.h5")
