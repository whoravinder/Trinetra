import serial
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from ai_model import UAVEstimator


class ImputerAE(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 16), torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(16, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x): return self.decoder(self.encoder(x))


class PredictorRNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=1, horizon=10):
        super().__init__()
        self.horizon = horizon
        self.rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        preds, h = [], None
        seq = x
        for _ in range(self.horizon):
            out, h = self.rnn(seq, h)
            step = self.fc(out[:, -1, :])
            preds.append(step)
            seq = torch.cat([seq, step.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

def run_integration(port="COM3", baud=115200, model_file="adyant_uav_estimator.pth"):
    ser = serial.Serial(port, baud, timeout=1); time.sleep(2)
    print(f"Connected to {port} at {baud}")

    model = UAVEstimator(); model.load_state_dict(torch.load(model_file)); model.eval()
    imputer = ImputerAE(); predictor = PredictorRNN(horizon=10)
    imputer.eval(); predictor.eval()

    max_points = 500
    time_window = deque(maxlen=max_points)
    raw_range, raw_velocity, raw_doa = deque(maxlen=max_points), deque(maxlen=max_points), deque(maxlen=max_points)
    ai_range, ai_velocity, ai_doa = deque(maxlen=max_points), deque(maxlen=max_points), deque(maxlen=max_points)

    past_predicted_paths = []  

    t0 = time.time(); prev_range, prev_time, doa_val = None, None, 0.0

    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(9, 12))

    while True:
        try:
            line = ser.readline().decode("utf-8").strip()
            if not line: continue
            distance_cm = float(line); curr_time = time.time(); distance_m = distance_cm/100.0

            velocity = 0.0
            if prev_range is not None:
                dt = curr_time - prev_time
                if dt > 0: velocity = (distance_m - prev_range)/dt
            prev_range, prev_time = distance_m, curr_time

            doa_val += velocity*10; doa_val = max(-60, min(60, doa_val))

            t = curr_time - t0
            sample = np.array([distance_m, velocity, doa_val])
            if distance_m <= 0: 
                with torch.no_grad(): sample = imputer(torch.tensor(sample).unsqueeze(0).float()).numpy()[0]

            time_window.append(t); raw_range.append(sample[0]); raw_velocity.append(sample[1]); raw_doa.append(sample[2])

           
            range_tensor = torch.tensor([sample[0]]*64, dtype=torch.float32).unsqueeze(0)   
            doppler_tensor = torch.zeros((1,64,64), dtype=torch.float32)                   
            doa_tensor = torch.zeros((1,180), dtype=torch.float32)
            doa_tensor[0, int((sample[2]+90)/180*179)] = 1

            with torch.no_grad():
                pred = model(range_tensor, doppler_tensor, doa_tensor).numpy()[0]

            ai_range.append(pred[0]); ai_velocity.append(pred[1]); ai_doa.append(pred[2])

            
            true_x = np.array(raw_range)*np.cos(np.deg2rad(raw_doa))
            true_y = np.array(raw_range)*np.sin(np.deg2rad(raw_doa))
            pred_x = np.array(ai_range)*np.cos(np.deg2rad(ai_doa))
            pred_y = np.array(ai_range)*np.sin(np.deg2rad(ai_doa))

            
            fut_x, fut_y = [], []
            if len(ai_range) > 10:
                seq = np.stack([ai_range, ai_velocity, ai_doa], axis=1)[-10:]
                with torch.no_grad():
                    fut_seq = predictor(torch.tensor(seq).unsqueeze(0).float()).numpy()[0]
                fut_x = fut_seq[:,0]*np.cos(np.deg2rad(fut_seq[:,2]))
                fut_y = fut_seq[:,0]*np.sin(np.deg2rad(fut_seq[:,2]))
                past_predicted_paths.append((fut_x, fut_y))

            
            axs[0].cla()
            axs[0].plot(true_x, true_y, "b-", linewidth=1.5, label="Raw Path")
            axs[0].plot(pred_x, pred_y, "r--", linewidth=2, label="AI Path")
            if len(fut_x) > 0:
                axs[0].plot(fut_x, fut_y, "g--o", linewidth=2.5, markersize=5, label="Predicted Future")
            for i, (old_fx, old_fy) in enumerate(past_predicted_paths[-3:]):
                alpha = 0.3 - i*0.1 if 0.3 - i*0.1 > 0 else 0.1
                axs[0].plot(old_fx, old_fy, "g:", linewidth=1.2, alpha=alpha)
            axs[0].set_title("UAV Trajectory", fontsize=13)
            axs[0].set_xlabel("X Position (m)")
            axs[0].set_ylabel("Y Position (m)")
            axs[0].grid(True, linestyle="--", alpha=0.6)
            axs[0].legend()
            axs[0].axis("equal")

            
            axs[1].cla()
            time_array = np.array(time_window)
            mask = time_array > time_array[-1] - 30 if len(time_array) > 0 else slice(None)
            axs[1].plot(time_array[mask], np.array(ai_range)[mask], "r-", linewidth=1.5, label="AI Range (m)")
            axs[1].plot(time_array[mask], np.array(ai_velocity)[mask], "g-", linewidth=1.5, label="AI Velocity (m/s)")
            axs[1].plot(time_array[mask], np.array(ai_doa)[mask], "b-", linewidth=1.5, label="AI DOA (°)")
            axs[1].set_title("AI Estimated Parameters (Last 30s)", fontsize=13)
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Values")
            axs[1].grid(True, linestyle="--", alpha=0.6)
            axs[1].legend()

            
            axs[2].cla()
            err_range = np.array(raw_range)[-len(ai_range):] - np.array(ai_range)
            err_vel   = np.array(raw_velocity)[-len(ai_velocity):] - np.array(ai_velocity)
            err_doa   = np.array(raw_doa)[-len(ai_doa):] - np.array(ai_doa)
            axs[2].plot(time_array[mask], err_range[mask], "r-", label="Range Error")
            axs[2].plot(time_array[mask], err_vel[mask], "g-", label="Velocity Error")
            axs[2].plot(time_array[mask], err_doa[mask], "b-", label="DOA Error")
            axs[2].set_title("AI Estimation Error (Raw − AI, Last 30s)", fontsize=13)
            axs[2].set_xlabel("Time (s)")
            axs[2].set_ylabel("Error Value")
            axs[2].grid(True, linestyle="--", alpha=0.6)
            axs[2].legend()

            plt.tight_layout()
            plt.pause(0.01)

        except Exception as e: 
            print("Error:", e)

if __name__ == "__main__":
    run_integration()
