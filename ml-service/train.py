"""Train the Name That LLM baseline classifier.

Run from ml-service:
    python -m pip install -r requirements.txt
    python train.py
"""
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            sublinear_tf=True, max_features=200_000, dtype=np.float32
        )),
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, max_features=300_000, dtype=np.float32
        )),
    ])
    return Pipeline([
        ("features", features),
        ("model", LinearSVC(C=1.0, random_state=42)),
    ])

def main() -> None:
    train_path = DATA / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Put the competition train.csv in ml-service/data/")
    df = pd.read_csv(train_path)
    required = {"text", "model", "prompt_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["text", "model", "prompt_id"]).copy()
    df["text"] = df["text"].astype(str)
    df["model"] = df["model"].astype(str)
    if df["model"].nunique() < 2:
        raise ValueError("At least two model labels are required.")

    # Group by prompt_id to avoid validating on prompts seen during training.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(splitter.split(df["text"], df["model"], groups=df["prompt_id"]))
    validator = build_pipeline()
    started = time.time()
    validator.fit(df["text"].iloc[train_idx], df["model"].iloc[train_idx])
    pred = validator.predict(df["text"].iloc[val_idx])
    metrics = {
        "validation_method": "GroupShuffleSplit by prompt_id, test_size=0.20, random_state=42",
        "validation_rows": int(len(val_idx)),
        "classes": int(df["model"].nunique()),
        "accuracy": float(accuracy_score(df["model"].iloc[val_idx], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(df["model"].iloc[val_idx], pred)),
        "classification_report": classification_report(
            df["model"].iloc[val_idx], pred, output_dict=True, zero_division=0
        ),
        "warning": "Single grouped holdout; use repeated/grouped cross-validation for a more stable estimate.",
    }
    (MODEL_DIR / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"Validation fit time: {time.time() - started:.1f}s")

    # Refit on all labelled data for inference.
    final_model = build_pipeline()
    final_model.fit(df["text"], df["model"])
    joblib.dump(final_model, MODEL_DIR / "classifier.joblib", compress=3)
    labels = sorted(df["model"].unique().tolist())
    (MODEL_DIR / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_DIR / 'classifier.joblib'} ({len(labels)} labels).")

if __name__ == "__main__":
    main()
