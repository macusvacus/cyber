"""
classifier.py
-------------
Machine learning classifier using a Random Forest model.
Trained on the synthetic dataset produced by simulator.py.

Targets from the proposal:
  - Accuracy  >= 85% across attack categories
  - False positive rate < 5%

Usage:
    python classifier.py          # trains, evaluates, and saves the model
    from classifier import MLClassifier
    clf = MLClassifier()
    clf.load()                    # loads saved model
    result = clf.predict(record)  # record is a dict of feature values
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "network_logs.csv")
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "rf_classifier.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# ── Features used for training ─────────────────────────────────────────────────
# These must match the columns produced by simulator.py
FEATURE_COLS = [
    "duration",
    "packet_count",
    "byte_count",
    "packets_per_sec",
    "bytes_per_sec",
    "src_port",
    "dst_port",
    "protocol_num",          # TCP=0, UDP=1, ICMP=2
    "syn_flag_count",
    "ack_flag_count",
    "fin_flag_count",
    "rst_flag_count",
    "idle_time",
    "connection_count",
]

LABEL_COL = "label"
RANDOM_SEED = 42
TEST_SIZE   = 0.20   # 80% train / 20% test


# ── Trainer ────────────────────────────────────────────────────────────────────

class MLClassifier:
    """
    Wraps a scikit-learn RandomForestClassifier with load/save and
    a predict() method compatible with the detector and dashboard.
    """

    def __init__(self):
        self.model   = None
        self.encoder = None        # LabelEncoder maps class names ↔ integers
        self.trained = False

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, data_path=DATA_PATH):
        """Load CSV, train RandomForest, print evaluation report."""
        print("Loading dataset...")
        df = pd.read_csv(data_path)

        # Ensure required columns are present
        missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}\n"
                             f"Run simulator.py first to generate the dataset.")

        X = df[FEATURE_COLS].values
        y = df[LABEL_COL].values

        # Encode string labels → integers
        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=TEST_SIZE,
            random_state=RANDOM_SEED, stratify=y_encoded
        )

        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")

        # Random Forest — 200 trees, handles class imbalance with balanced weights
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        self.trained = True

        # ── Evaluation ──────────────────────────────────────────────────────────
        y_pred = self.model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        labels = self.encoder.classes_

        print("\n" + "=" * 60)
        print("ML Classifier Evaluation Results")
        print("=" * 60)
        print(f"Overall Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"Proposal target  : >= 85.00%")
        print(f"Status           : {'✓ PASS' if acc >= 0.85 else '✗ BELOW TARGET'}")
        print()
        print("Per-class Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=labels,
            digits=4
        ))

        # False positive rate per class
        print("False Positive Rate per class:")
        cm = confusion_matrix(y_test, y_pred)
        for i, cls in enumerate(labels):
            fp  = cm[:, i].sum() - cm[i, i]       # predicted as i but not i
            tn  = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            status = "✓" if fpr < 0.05 else "✗"
            print(f"  {status} {cls:<12}: FPR = {fpr:.4f}  ({fpr*100:.2f}%)")

        print("=" * 60)
        return acc

    # ── Save / Load ────────────────────────────────────────────────────────────

    def save(self, model_path=MODEL_PATH, encoder_path=ENCODER_PATH):
        """Save trained model and label encoder to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(encoder_path, "wb") as f:
            pickle.dump(self.encoder, f)
        print(f"\nModel saved   : {model_path}")
        print(f"Encoder saved : {encoder_path}")

    def load(self, model_path=MODEL_PATH, encoder_path=ENCODER_PATH):
        """Load a previously saved model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No saved model found at {model_path}.\n"
                f"Run 'python classifier.py' first to train and save the model."
            )
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(encoder_path, "rb") as f:
            self.encoder = pickle.load(f)
        self.trained = True

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, record: dict) -> dict:
        """
        Predict the label for a single traffic record.

        Parameters
        ----------
        record : dict
            Must contain all keys in FEATURE_COLS.

        Returns
        -------
        dict with:
            label       – predicted class string
            confidence  – probability of the predicted class (0.0–1.0)
            probabilities – dict of {class: probability} for all classes
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Build feature vector in the correct column order
        try:
            features = np.array([[record[col] for col in FEATURE_COLS]])
        except KeyError as e:
            raise ValueError(f"Missing feature in record: {e}")

        pred_encoded  = self.model.predict(features)[0]
        proba         = self.model.predict_proba(features)[0]
        label         = self.encoder.inverse_transform([pred_encoded])[0]
        probabilities = {
            cls: float(p)
            for cls, p in zip(self.encoder.classes_, proba)
        }

        return {
            "label":         label,
            "confidence":    float(proba[pred_encoded]),
            "probabilities": probabilities,
        }

    def predict_batch(self, records: list) -> list:
        """Predict labels for a list of record dicts."""
        return [self.predict(r) for r in records]


# ── Main: train, evaluate, save ────────────────────────────────────────────────

if __name__ == "__main__":
    clf = MLClassifier()
    clf.train()
    clf.save()

    # Quick live prediction test
    print("\nLive prediction test:")
    clf.load()
    sample = {
        "duration": 0.01, "packet_count": 1, "byte_count": 60,
        "packets_per_sec": 100, "bytes_per_sec": 6000,
        "src_port": 44000, "dst_port": 8080, "protocol_num": 0,
        "syn_flag_count": 1, "ack_flag_count": 0, "fin_flag_count": 0,
        "rst_flag_count": 0, "idle_time": 0.001, "connection_count": 800,
    }
    result = clf.predict(sample)
    print(f"  Prediction : {result['label']}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print(f"  All probs  : { {k: f'{v:.2%}' for k, v in result['probabilities'].items()} }")
