"""
train_har.py — Train a HAR model from CSV and export a joblib model + metadata.
Usage:
    python train_har.py --csv data.csv --model har_model.joblib
"""
import argparse
from machineLearning.har_ml import train_and_select

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to labeled sensor CSV (ts,ax,ay,az,gx,gy,gz,label)")
    ap.add_argument("--model", required=True, help="Output path, e.g., har_model.joblib")
    args = ap.parse_args()
    info = train_and_select(args.csv, args.model)
    print("Saved model:", args.model, "info:", info)

if __name__ == "__main__":
    main()
