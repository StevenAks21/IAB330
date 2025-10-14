"""
har_ml.py — Feature extraction, training, and streaming inference for HAR + fall detection.
Assumes 50 Hz IMU from Arduino Nano 33 IoT, windows of 2 s (100 samples).

Expected CSV schema for training:
    ts, ax, ay, az, gx, gy, gz, label
where accelerometer is in mg (or m/s^2) and gyro in dps (consistent units across all rows).

Author: ChatGPT (for Tri Cuong Dinh)
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Tuple
from collections import deque, Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# ----------------------------- constants -----------------------------

FS = 50                 # Hz
WINDOW_SEC = 2.0
WIN = int(FS * WINDOW_SEC)
STEP = int(WIN // 4)    # 0.5 s hop
LABELS = ["walk", "sit", "stand", "lie", "fall", "moving", "transition"]  # superset; you can subset as needed

# Fall detection heuristic thresholds (tune if needed)
IMPACT_G = 2.8          # ≈2.8 g impact
STILL_STD_MG = 80       # post-impact stillness in mg (std dev threshold)
UPRIGHT_DEG = 30        # near upright angle threshold in degrees
LYING_DEG = 60          # lying angle threshold in degrees

# ----------------------------- utilities -----------------------------

def vector_mag(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.sqrt(a*a + b*b + c*c)

def posture_angle_deg(ax: float, ay: float, az: float) -> float:
    """
    Simple tilt angle from gravity direction; assumes accelerometer includes gravity.
    """
    g = math.sqrt(ax*ax + ay*ay + az*az) + 1e-9
    # angle between gravity vector and "up" (y axis) in degrees; adjust to your board orientation
    cos_theta = max(-1.0, min(1.0, ay / g))
    return math.degrees(math.acos(cos_theta))

def zero_crossings(x: np.ndarray) -> int:
    return int(((x[:-1] * x[1:]) < 0).sum())

def spectral_energy(x: np.ndarray) -> float:
    # Use numpy FFT; remove DC
    X = np.fft.rfft(x - np.mean(x))
    return float(np.sum(np.abs(X)**2) / len(X))

def dominant_freq(x: np.ndarray, fs: int = FS) -> float:
    X = np.fft.rfft(x - np.mean(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
    idx = np.argmax(np.abs(X))
    return float(freqs[idx])

def basic_stats(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "range": float(np.max(x) - np.min(x)),
    }

# ----------------------------- feature extraction -----------------------------

FEATURE_NAMES: List[str] = []  # populated at module import

def _axis_features(prefix: str, x: np.ndarray) -> Dict[str, float]:
    d = {}
    s = basic_stats(x)
    for k, v in s.items():
        d[f"{prefix}_{k}"] = v
    d[f"{prefix}_zcr"] = float(zero_crossings(x))
    d[f"{prefix}_domfreq"] = dominant_freq(x)
    d[f"{prefix}_specen"] = spectral_energy(x)
    return d

def extract_window_features(ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
                            gx: np.ndarray, gy: np.ndarray, gz: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Compute time+frequency features for a single window (shape: 100 samples per axis).
    Returns feature vector and names.
    """
    feats: Dict[str, float] = {}
    # Per-axis features
    feats.update(_axis_features("ax", ax))
    feats.update(_axis_features("ay", ay))
    feats.update(_axis_features("az", az))
    feats.update(_axis_features("gx", gx))
    feats.update(_axis_features("gy", gy))
    feats.update(_axis_features("gz", gz))

    # Combined magnitudes
    amag = vector_mag(ax, ay, az)
    gmag = vector_mag(gx, gy, gz)
    feats.update(_axis_features("amag", amag))
    feats.update(_axis_features("gmag", gmag))

    # Signal Magnitude Area (SMA)
    feats["sma_acc"] = float(np.mean(np.abs(ax)) + np.mean(np.abs(ay)) + np.mean(np.abs(az)))
    feats["sma_gyr"] = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)) + np.mean(np.abs(gz)))

    # Correlations between axes (acc)
    feats["corr_ax_ay"] = float(np.corrcoef(ax, ay)[0,1])
    feats["corr_ax_az"] = float(np.corrcoef(ax, az)[0,1])
    feats["corr_ay_az"] = float(np.corrcoef(ay, az)[0,1])

    # Posture angle (use last sample in window as current tilt)
    feats["tilt_deg_last"] = posture_angle_deg(ax[-1], ay[-1], az[-1])

    names = list(feats.keys())
    return np.array([feats[n] for n in names], dtype=float), names

def windows_from_df(df: pd.DataFrame, win: int = WIN, step: int = STEP) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """
    Yield sliding windows with labels. A window gets the label of its majority label rows.
    """
    n = len(df)
    for start in range(0, max(0, n - win + 1), step):
        seg = df.iloc[start:start+win]
        ax, ay, az = seg["ax"].values, seg["ay"].values, seg["az"].values
        gx, gy, gz = seg["gx"].values, seg["gy"].values, seg["gz"].values
        if "label" in seg.columns:
            label = Counter(seg["label"].values).most_common(1)[0][0]
        else:
            label = None
        yield ax, ay, az, gx, gy, gz, label

def features_from_csv(csv_path: str, label_map: Dict[str, str] | None = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    if label_map:
        df["label"] = df["label"].map(lambda x: label_map.get(x, x))
    X_list, y_list = [], []
    names_cache: List[str] | None = None
    for ax, ay, az, gx, gy, gz, label in windows_from_df(df):
        fv, names = extract_window_features(ax, ay, az, gx, gy, gz)
        names_cache = names
        if label is not None:
            X_list.append(fv)
            y_list.append(label)
    X = np.vstack(X_list)
    y = np.array(y_list)
    global FEATURE_NAMES
    FEATURE_NAMES = names_cache or []
    return X, y, FEATURE_NAMES

# ----------------------------- training -----------------------------

def train_and_select(csv_path: str, model_out: str, label_map: Dict[str,str] | None = None) -> Dict[str, float]:
    """
    Train DecisionTree, KNN, SVM and choose the best by validation accuracy.
    Saves the best as a joblib pipeline at model_out, plus a metadata JSON alongside.
    """
    X, y, names = features_from_csv(csv_path, label_map=label_map)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Pipelines
    dt = Pipeline([("clf", DecisionTreeClassifier(random_state=42))])
    knn = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
    svm = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced"))])

    # Grids
    grids = [
        ("DecisionTree", dt,  {"clf__max_depth": [None, 5, 10, 15], "clf__min_samples_leaf": [1, 3, 5]}),
        ("KNN",         knn, {"clf__n_neighbors": [3,5,7,9]}),
        ("SVM",         svm, {"clf__C": [0.5, 1, 2], "clf__gamma": ["scale", 0.05, 0.1]}),
    ]

    best_name, best_est, best_acc = None, None, -1.0
    for name, pipe, grid in grids:
        gs = GridSearchCV(pipe, grid, cv=3, n_jobs=-1)
        gs.fit(Xtr, ytr)
        ypred = gs.predict(Xte)
        acc = accuracy_score(yte, ypred)
        print(f"[{name}] acc={acc:.3f} best_params={gs.best_params_}")
        if acc > best_acc:
            best_name, best_est, best_acc = name, gs.best_estimator_, acc

    # Evaluate best
    yhat = best_est.predict(Xte)
    acc  = accuracy_score(yte, yhat)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, yhat, average="weighted", zero_division=0)
    cm = confusion_matrix(yte, yhat, labels=sorted(np.unique(y)))
    print("\nBest:", best_name)
    print("Accuracy:", acc)
    print("Weighted P/R/F1:", pr, rc, f1)
    print("Labels:", sorted(np.unique(y)))
    print("Confusion matrix:\n", cm)
    print("\nReport:\n", classification_report(yte, yhat, zero_division=0))

    # Save model + metadata
    joblib.dump(best_est, model_out)
    meta = {
        "feature_names": names,
        "labels": sorted(list(set(y.tolist()))),
        "fs": FS,
        "window_sec": WINDOW_SEC,
        "step": STEP,
        "model": best_name,
        "accuracy": float(acc),
    }
    with open(os.path.splitext(model_out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return {"accuracy": float(acc), "model": best_name}

# ----------------------------- streaming inference -----------------------------

@dataclass
class StreamState:
    ax: Deque[float]
    ay: Deque[float]
    az: Deque[float]
    gx: Deque[float]
    gy: Deque[float]
    gz: Deque[float]

    @classmethod
    def new(cls, win: int = WIN) -> "StreamState":
        return cls(deque(maxlen=win), deque(maxlen=win), deque(maxlen=win),
                   deque(maxlen=win), deque(maxlen=win), deque(maxlen=win))

class HarInferencer:
    """
    Wraps a trained sklearn pipeline (joblib) to classify streaming IMU windows.
    Optionally runs a simple fall heuristic in parallel for immediate FALL flag.
    """
    def __init__(self, model_path: str, use_heuristic_fall: bool = True):
        self.model = joblib.load(model_path)
        meta_path = os.path.splitext(model_path)[0] + ".meta.json"
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.state = StreamState.new(WIN)
        self.use_heuristic_fall = use_heuristic_fall
        self.last_impact_ts = 0.0

    def push_sample(self, ax: float, ay: float, az: float, gx: float, gy: float, gz: float, ts: float | None = None) -> Tuple[str, float]:
        self.state.ax.append(ax); self.state.ay.append(ay); self.state.az.append(az)
        self.state.gx.append(gx); self.state.gy.append(gy); self.state.gz.append(gz)

        if len(self.state.ax) < WIN:
            return ("warming", 0.0)

        # ML classification for current window
        fv, _ = extract_window_features(
            np.array(self.state.ax), np.array(self.state.ay), np.array(self.state.az),
            np.array(self.state.gx), np.array(self.state.gy), np.array(self.state.gz)
        )
        proba = None
        if hasattr(self.model, "predict_proba"):
            P = self.model.predict_proba([fv])[0]
            y = self.model.classes_[int(np.argmax(P))]
            conf = float(np.max(P))
        else:
            y = self.model.predict([fv])[0]
            conf = 1.0  # SVM without probability=True
        label_ml, conf_ml = str(y), float(conf)

        if not self.use_heuristic_fall:
            return (label_ml, conf_ml)

        # Heuristic fall overlay: detect high-impact + posture change
        amag = vector_mag(np.array(self.state.ax), np.array(self.state.ay), np.array(self.state.az))
        impact = float(np.max(amag))
        tilt = posture_angle_deg(self.state.ax[-1], self.state.ay[-1], self.state.az[-1])
        still_std = float(np.std(amag[-int(FS*1):]))  # last ~1s

        likely_fall = (impact > IMPACT_G * 1000.0) or (label_ml.lower() == "fall")
        post_fall_posture = (tilt > LYING_DEG) and (still_std < STILL_STD_MG)

        if likely_fall and post_fall_posture:
            return ("FALL", 0.99)

        return (label_ml, conf_ml)
