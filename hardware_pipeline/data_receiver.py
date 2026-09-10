import serial
import time

def read_ultrasonic(port="COM3", baud=115200):
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)
    print(f"📡 Listening on {port} at {baud} baud...")

    while True:
        try:
            line = ser.readline().decode("utf-8").strip()
            if line:
                distance = float(line)
                print(f"Distance: {distance:.2f} cm")
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    read_ultrasonic()
