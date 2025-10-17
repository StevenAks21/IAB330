device_name = "Nano33-Steven"
characteristic_uuid = "12345678-1234-5678-1234-56789abcdef1"

mongo_url = "mongodb+srv://steven:2121@iab330steven.t74aifz.mongodb.net/?retryWrites=true&w=majority&appName=IAB330Steven"
mongo_database_name = "HARData"
mongo_collection_name = "Data"

csv_file_path = "./data.csv"
model_file_path = "./har.joblib"

sample_rate_hz = 50
window_size_samples = sample_rate_hz
accel_impact_mg = 2800

import os, sys, math, struct, asyncio, csv
from collections import deque
import numpy as np, pandas as pd
from bleak import BleakClient, BleakScanner
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load
from pymongo import MongoClient
import certifi

async def find_device():
    devices = await BleakScanner.discover(timeout=10.0)
    for device in devices:
        name = (device.name or "")
        if device_name.lower() in name.lower():
            print(name)
            return device.address

async def collect_mode(label, csv_path=csv_file_path):
    is_new_file = not os.path.exists(csv_path)
    file_handle = open(csv_path, "a", newline="")
    csv_writer = csv.writer(file_handle)
    if is_new_file:
        csv_writer.writerow(["ts","ax","ay","az","gx","gy","gz","label"])
    device_address = await find_device()
    client = BleakClient(device_address, timeout=10)
    await client.connect()
    print("connected")
    async def on_notify(_, payload):
        timestamp, ax, ay, az, gx, gy, gz = struct.unpack("<I6h", payload)
        csv_writer.writerow([timestamp, ax, ay, az, gx, gy, gz, label])
        file_handle.flush()
    await client.start_notify(characteristic_uuid, on_notify)
    print("collecting...")
    while True:
        await asyncio.sleep(1)

def train_mode(csv_path=csv_file_path, model_path=model_file_path):
    print("training...")
    data_frame = pd.read_csv(csv_path)
    samples = data_frame[["ax","ay","az","gx","gy","gz","label"]].values.tolist()

    feature_vectors = []
    window_labels = []

    total_samples = len(samples)
    window_step = window_size_samples

    for window_start_index in range(0, total_samples - window_step + 1, window_step):
        window_samples_list = samples[window_start_index:window_start_index + window_step]

        window_label_list = [row[6] for row in window_samples_list]
        majority_label = max(set(window_label_list), key=window_label_list.count)

        ax_series = [row[0] for row in window_samples_list]
        ay_series = [row[1] for row in window_samples_list]
        az_series = [row[2] for row in window_samples_list]
        gx_series = [row[3] for row in window_samples_list]
        gy_series = [row[4] for row in window_samples_list]
        gz_series = [row[5] for row in window_samples_list]

        feature_values = []

        for series in (ax_series, ay_series, az_series, gx_series, gy_series, gz_series):
            series_length = len(series)
            mean_value = sum(series) / series_length
            variance_value = sum((x - mean_value) * (x - mean_value) for x in series) / series_length
            std_value = math.sqrt(variance_value)
            min_value = min(series)
            max_value = max(series)
            range_value = max_value - min_value
            feature_values += [mean_value, std_value, min_value, max_value, range_value]

        magnitude_series = [math.sqrt(x*x + y*y + z*z) for x, y, z in zip(ax_series, ay_series, az_series)]
        magnitude_length = len(magnitude_series)
        mean_magnitude = sum(magnitude_series) / magnitude_length
        variance_magnitude = sum((m - mean_magnitude) * (m - mean_magnitude) for m in magnitude_series) / magnitude_length
        std_magnitude = math.sqrt(variance_magnitude)
        signal_magnitude_area = sum(abs(x) + abs(y) + abs(z) for x, y, z in zip(ax_series, ay_series, az_series)) / len(ax_series)

        feature_values += [mean_magnitude, std_magnitude, signal_magnitude_area]

        feature_vectors.append(feature_values)
        window_labels.append(majority_label)

    model = DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42)
    model.fit(np.array(feature_vectors), np.array(window_labels))
    dump(model, model_path)
    print("training done")

async def infer_mode(model_path=model_file_path):
    model = load(model_path)
    print("model ready")
    device_address = await find_device()
    client = BleakClient(device_address, timeout=10)
    await client.connect()
    print("connected")
    print("inferring...")

    window_buffer = deque(maxlen=window_size_samples)
    samples_count_in_window = 0

    async def on_notify(_, payload):
        nonlocal samples_count_in_window
        _, ax, ay, az, gx, gy, gz = struct.unpack("<I6h", payload)

        window_buffer.append([ax, ay, az, gx, gy, gz])
        samples_count_in_window += 1

        current_magnitude = math.sqrt(ax*ax + ay*ay + az*az)
        if current_magnitude > accel_impact_mg:
            print("FALL")

        if samples_count_in_window >= window_size_samples and len(window_buffer) >= window_size_samples:
            samples_count_in_window = 0
            rows_in_window = list(window_buffer)

            ax_series = [row[0] for row in rows_in_window]
            ay_series = [row[1] for row in rows_in_window]
            az_series = [row[2] for row in rows_in_window]
            gx_series = [row[3] for row in rows_in_window]
            gy_series = [row[4] for row in rows_in_window]
            gz_series = [row[5] for row in rows_in_window]

            feature_values = []

            for series in (ax_series, ay_series, az_series, gx_series, gy_series, gz_series):
                series_length = len(series)
                mean_value = sum(series) / series_length
                variance_value = sum((x - mean_value) * (x - mean_value) for x in series) / series_length
                std_value = math.sqrt(variance_value)
                min_value = min(series)
                max_value = max(series)
                range_value = max_value - min_value
                feature_values += [mean_value, std_value, min_value, max_value, range_value]

            magnitude_series = [math.sqrt(x*x + y*y + z*z) for x, y, z in zip(ax_series, ay_series, az_series)]
            magnitude_length = len(magnitude_series)
            mean_magnitude = sum(magnitude_series) / magnitude_length
            variance_magnitude = sum((m - mean_magnitude) * (m - mean_magnitude) for m in magnitude_series) / magnitude_length
            std_magnitude = math.sqrt(variance_magnitude)
            signal_magnitude_area = sum(abs(x) + abs(y) + abs(z) for x, y, z in zip(ax_series, ay_series, az_series)) / len(ax_series)

            feature_values += [mean_magnitude, std_magnitude, signal_magnitude_area]

            prediction_label = model.predict([feature_values])[0]
            print(str(prediction_label))

    await client.start_notify(characteristic_uuid, on_notify)
    while True:
        await asyncio.sleep(1)

def get_database():
    return MongoClient(
        mongo_url,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=10000,
    )[mongo_database_name]

def cloud_upload(csv_path=csv_file_path):
    print("uploading...")
    collection = get_database()[mongo_collection_name]
    data_frame = pd.read_csv(csv_path, usecols=["ts","ax","ay","az","gx","gy","gz","label"])
    data_frame["ts"] = data_frame["ts"].astype(int)
    for column_name in ["ax","ay","az","gx","gy","gz"]:
        data_frame[column_name] = data_frame[column_name].astype(int)
    data_frame["label"] = data_frame["label"].astype(str)
    collection.insert_many(data_frame.to_dict("records"))
    print("upload done")

def cloud_update(csv_path=csv_file_path):
    print("updating...")
    collection = get_database()[mongo_collection_name]
    collection.delete_many({})
    cloud_upload(csv_path)
    print("update done")

def cloud_retrieve(csv_path=csv_file_path):
    print("downloading...")
    collection = get_database()[mongo_collection_name]
    documents = list(collection.find({}, {"_id":0,"ts":1,"ax":1,"ay":1,"az":1,"gx":1,"gy":1,"gz":1,"label":1}).sort("ts",1))
    with open(csv_path, "w", newline="") as file_handle:
        csv_writer = csv.writer(file_handle)
        csv_writer.writerow(["ts","ax","ay","az","gx","gy","gz","label"])
        for document in documents:
            csv_writer.writerow([
                int(document.get("ts",0)),
                int(document.get("ax",0)),
                int(document.get("ay",0)),
                int(document.get("az",0)),
                int(document.get("gx",0)),
                int(document.get("gy",0)),
                int(document.get("gz",0)),
                str(document.get("label","")),
            ])
    print("download done")

def cloud_delete():
    collection = get_database()[mongo_collection_name]
    collection.delete_many({})
    print("delete done")

def menu():
    while True:
        print("\n1) collect  2) train  3) infer  4) upload  5) retrieve  6) update  7) delete  0) quit")
        choice = input("> ").strip()
        if choice == "1":
            asyncio.run(collect_mode(input("label: ").strip()))
        elif choice == "2":
            train_mode()
        elif choice == "3":
            asyncio.run(infer_mode())
        elif choice == "4":
            cloud_upload()
        elif choice == "5":
            cloud_retrieve()
        elif choice == "6":
            cloud_update()
        elif choice == "7":
            cloud_delete()
        elif choice == "0":
            sys.exit(0)

if __name__ == "__main__":
    menu()