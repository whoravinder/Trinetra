import serial
import time
import matplotlib.pyplot as plt
from collections import deque
import csv

def hardware_demo(port="COM3", baud=115200, log_file="ultrasonic_log.csv"):
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)
    max_points = 100
    time_window = deque(maxlen=max_points)
    range_window = deque(maxlen=max_points)
    velocity_window = deque(maxlen=max_points)
    doa_window = deque(maxlen=max_points)
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Range_m", "Velocity_mps", "DOA_deg"])
        plt.ion()
        fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        prev_range, prev_time = None, None
        t0 = time.time()
        doa = 0.0
        while True:
            try:
                line = ser.readline().decode("utf-8").strip()
                if not line:
                    continue
                distance_cm = float(line)
                curr_time = time.time()
                distance_m = distance_cm / 100.0
                velocity = 0.0
                if prev_range is not None:
                    dt = curr_time - prev_time
                    if dt > 0:
                        velocity = (distance_m - prev_range) / dt
                prev_range, prev_time = distance_m, curr_time
                doa += velocity * 10
                doa = max(-60, min(60, doa))
                time_window.append(curr_time - t0)
                range_window.append(distance_m)
                velocity_window.append(velocity)
                doa_window.append(doa)
                writer.writerow([curr_time, distance_m, velocity, doa])
                f.flush()
                axs[0].cla(); axs[0].plot(list(time_window), list(range_window), label="Range (m)"); axs[0].legend()
                axs[1].cla(); axs[1].plot(list(time_window), list(velocity_window), label="Velocity (m/s)"); axs[1].legend()
                axs[2].cla(); axs[2].plot(list(time_window), list(doa_window), label="DOA (deg)"); axs[2].legend()
                plt.pause(0.01)
            except Exception as e:
                print("Error:", e)

if __name__ == "__main__":
    hardware_demo()
