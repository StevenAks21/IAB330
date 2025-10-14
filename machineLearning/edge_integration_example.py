"""
edge_integration_example.py — How to use the trained model in a streaming loop.
This example shows the classify path only. Replace the "read_sample()" function with your BLE callback.
"""
import time
from har_ml import HarInferencer

# Replace with the joblib you trained
MODEL_PATH = "har_model.joblib"

# Dummy generator — replace with your BLE notification payload parser
def read_sample():
    """
    Yield (ax, ay, az, gx, gy, gz) at ~50 Hz.
    In your code, parse bytes from the Arduino notify payload into these six floats/ints.
    """
    import math, random
    t = 0.0
    while True:
        # Fake walk-ish acceleration
        ax = 100 * math.sin(2*math.pi*1.2*t) + random.uniform(-10,10)
        ay = 980 + random.uniform(-20,20)  # gravity + noise (mg)
        az = 50 * math.cos(2*math.pi*1.2*t) + random.uniform(-10,10)
        gx = random.uniform(-5,5); gy = random.uniform(-5,5); gz = random.uniform(-5,5)
        yield ax, ay, az, gx, gy, gz
        t += 1/50.0
        time.sleep(1/50.0)

def main():
    infer = HarInferencer(MODEL_PATH, use_heuristic_fall=True)
    for ax, ay, az, gx, gy, gz in read_sample():
        label, conf = infer.push_sample(ax, ay, az, gx, gy, gz)
        if label != "warming":
            print(time.time(), "|", f"{label:<10}", f"conf={conf:.2f}")

if __name__ == "__main__":
    main()
