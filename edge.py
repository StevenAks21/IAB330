device_name = "Nano33-FallServer"
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

async def find_device():
    devices = await BleakScanner.discover(timeout=10.0)
    for device in devices:
        name = (device.name or "")
        if device_name.lower() in name.lower():
            print(name)
            return device.address

async def collect_mode(label, csv_path=csv_file_path):
    new_file = not os.path.exists(csv_path)
    file_handle = open(csv_path, "a", newline="")
    writer = csv.writer(file_handle)
    if new_file:
        writer.writerow(["ts","ax","ay","az","gx","gy","gz","label"])
    device_address = await find_device()
    client = BleakClient(device_address, timeout=10)
    await client.connect()
    print("connected")
    async def on_notify(_, payload):
        timestamp,ax,ay,az,gx,gy,gz = struct.unpack("<I6h", payload)
        writer.writerow([timestamp,ax,ay,az,gx,gy,gz,label])
        file_handle.flush()
    await client.start_notify(characteristic_uuid, on_notify)
    print("collecting...")
    while True:
        await asyncio.sleep(1)

def train_mode(csv_path=csv_file_path, model_path=model_file_path):
    print("training...")
    data_frame = pd.read_csv(csv_path)
    rows_list = data_frame[["ax","ay","az","gx","gy","gz","label"]].to_numpy().tolist()
    features, labels = [], []
    total_rows = len(rows_list)
    for start in range(0, total_rows - window_size_samples + 1, window_size_samples):
        segment = rows_list[start:start + window_size_samples]
        segment_labels = [row[6] for row in segment]
        label_majority = max(set(segment_labels), key=segment_labels.count)
        ax = [row[0] for row in segment]; ay = [row[1] for row in segment]; az = [row[2] for row in segment]
        gx = [row[3] for row in segment]; gy = [row[4] for row in segment]; gz = [row[5] for row in segment]
        feature_values = []
        for series in (ax, ay, az, gx, gy, gz):
            mean_value = sum(series) / len(series)
            std_value = (sum((x - mean_value) * (x - mean_value) for x in series) / len(series)) ** 0.5
            min_value = min(series)
            max_value = max(series)
            range_value = max_value - min_value
            feature_values += [mean_value, std_value, min_value, max_value, range_value]
        magnitude_series = [math.sqrt(ax[i]*ax[i] + ay[i]*ay[i] + az[i]*az[i]) for i in range(len(ax))]
        mean_magnitude = sum(magnitude_series) / len(magnitude_series)
        std_magnitude = (sum((x - mean_magnitude) * (x - mean_magnitude) for x in magnitude_series) / len(magnitude_series)) ** 0.5
        signal_magnitude_area = sum(abs(ax[i]) + abs(ay[i]) + abs(az[i]) for i in range(len(ax))) / len(ax)
        feature_values += [mean_magnitude, std_magnitude, signal_magnitude_area]
        features.append(feature_values)
        labels.append(label_majority)
    model = DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42)
    model.fit(np.array(features), np.array(labels))
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
    samples_in_window = 0
    async def on_notify(_, payload):
        nonlocal samples_in_window
        _,ax,ay,az,gx,gy,gz = struct.unpack("<I6h", payload)
        window_buffer.append([ax,ay,az,gx,gy,gz])
        samples_in_window += 1
        magnitude = math.sqrt(ax*ax + ay*ay + az*az)
        if magnitude > accel_impact_mg:
            print("FALL")
        if samples_in_window >= window_size_samples and len(window_buffer) >= window_size_samples:
            samples_in_window = 0
            array_window = list(window_buffer)
            ax = [row[0] for row in array_window]; ay = [row[1] for row in array_window]; az = [row[2] for row in array_window]
            gx = [row[3] for row in array_window]; gy = [row[4] for row in array_window]; gz = [row[5] for row in array_window]
            feature_values = []
            for series in (ax, ay, az, gx, gy, gz):
                mean_value = sum(series) / len(series)
                std_value = (sum((x - mean_value) * (x - mean_value) for x in series) / len(series)) ** 0.5
                min_value = min(series)
                max_value = max(series)
                range_value = max_value - min_value
                feature_values += [mean_value, std_value, min_value, max_value, range_value]
            magnitude_series = [math.sqrt(ax[i]*ax[i] + ay[i]*ay[i] + az[i]*az[i]) for i in range(len(ax))]
            mean_magnitude = sum(magnitude_series) / len(magnitude_series)
            std_magnitude = (sum((x - mean_magnitude) * (x - mean_magnitude) for x in magnitude_series) / len(magnitude_series)) ** 0.5
            signal_magnitude_area = sum(abs(ax[i]) + abs(ay[i]) + abs(az[i]) for i in range(len(ax))) / len(ax)
            feature_values += [mean_magnitude, std_magnitude, signal_magnitude_area]
            print(str(model.predict([feature_values])[0]))
    await client.start_notify(characteristic_uuid, on_notify)
    while True:
        await asyncio.sleep(1)

def get_database():
    return MongoClient(
        mongo_url,
        tls=True,
        tlsCAFile="/etc/ssl/certs/ca-certificates.crt",
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=10000,
    )[mongo_database_name]

def cloud_upload(csv_path=csv_file_path):
    print("uploading...")
    collection = get_database()[mongo_collection_name]
    data_frame = pd.read_csv(csv_path, usecols=["ts","ax","ay","az","gx","gy","gz","label"])
    data_frame["ts"] = data_frame["ts"].astype(int)
    for column in ["ax","ay","az","gx","gy","gz"]:
        data_frame[column] = data_frame[column].astype(int)
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
        writer = csv.writer(file_handle)
        writer.writerow(["ts","ax","ay","az","gx","gy","gz","label"])
        for document in documents:
            writer.writerow([
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