"""
NAME THAT LLM - Training script (v2)
 
Run from ml-service:
    python -m pip install -r requirements.txt
    python train.py
 
Input : data/train.csv  with columns  text, model, prompt_id
Output: model/classifier.joblib, model/labels.json, model/validation_metrics.json
 
At inference time pass the RAW text to the pipeline (no lowercasing, no
stripping, no whitespace cleanup) - the features rely on casing, newlines,
markdown symbols and punctuation, which are the main LLM "fingerprints".
"""
 
from pathlib import Path
import json
import time
import warnings
 
import joblib
import numpy as np
import pandas as pd
 
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
 
warnings.filterwarnings("ignore")
 
# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
 
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
 
TRAIN_FILE = DATA / "train.csv"
 
C_VALUES = [0.3, 1.0, 3.0]   # SVM strengths to compare
N_FOLDS = 4                  # grouped cross-validation folds
USE_CALIBRATION = False      # True -> model gets predict_proba (slower, ~3x)
RANDOM_STATE = 42
 
 
# --------------------------------------------------
# FEATURES + MODEL
# --------------------------------------------------
 
def make_features():
    """Word + character TF-IDF that KEEPS style information.
 
    Key differences from a default TfidfVectorizer:
      * lowercase=False  -> capitalisation is a strong style signal
      * punctuation is kept as tokens (em-dashes, colons, quotes, **, ## ...)
      * char n-grams start at 1 and include newlines/spaces, so paragraph
        layout, bullet style and markdown habits are captured
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
            ngram_range=(1, 3),
            min_df=2,
            sublinear_tf=True,
            max_features=300000,
            dtype=np.float32,
        )),
        ("char", TfidfVectorizer(
            analyzer="char",
            lowercase=False,
            ngram_range=(1, 5),
            min_df=3,
            sublinear_tf=True,
            max_features=500000,
            dtype=np.float32,
        )),
    ])
 
 
def make_classifier(C, calibrate=False):
    svm = LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    if calibrate:
        return CalibratedClassifierCV(svm, cv=3)
    return svm
 
 
def build_pipeline(C, calibrate=False):
    return Pipeline([
        ("features", make_features()),
        ("model", make_classifier(C, calibrate)),
    ])
 
 
# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
 
def load_dataset():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
 
    df = pd.read_csv(TRAIN_FILE)
 
    required = ["text", "model", "prompt_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in train.csv: {missing}")
 
    df = df.dropna(subset=required).copy()
 
    # NOTE: text is NOT stripped/modified - whitespace is a useful signal.
    df["text"] = df["text"].astype(str)
    df["model"] = df["model"].astype(str)
    df["prompt_id"] = df["prompt_id"].astype(str)
 
    df = df[df["text"].str.strip().str.len() >= 20].reset_index(drop=True)
 
    if len(df) == 0:
        raise ValueError("No valid training examples found.")
    if df["model"].nunique() < 2:
        raise ValueError("At least two model classes are required.")
    if df["prompt_id"].nunique() < 2:
        raise ValueError("At least two unique prompts are required.")
 
    print("\nDATASET SUMMARY")
    print("-------------------------")
    print("Total samples  :", len(df))
    print("LLM classes    :", df["model"].nunique())
    print("Unique prompts :", df["prompt_id"].nunique())
    print("\nClass distribution:")
    print(df["model"].value_counts())
 
    return df
 
 
# --------------------------------------------------
# GROUPED CROSS-VALIDATION (choose C)
# --------------------------------------------------
 
def cross_validate(df):
    """Group by prompt_id so the same prompt never appears in both train and
    validation. Features are built once per fold and reused for every C."""
    X = df["text"].values
    y = df["model"].values
    groups = df["prompt_id"].values
 
    n_splits = min(N_FOLDS, df["prompt_id"].nunique())
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
 
    scores = {C: [] for C in C_VALUES}
    oof = {C: np.empty(len(df), dtype=object) for C in C_VALUES}
 
    print("\nGROUPED CROSS-VALIDATION  (%d folds)" % n_splits)
    print("============================")
 
    for i, (tr, va) in enumerate(folds, 1):
        t0 = time.time()
        feats = make_features()
        Xtr = feats.fit_transform(X[tr])
        Xva = feats.transform(X[va])
 
        for C in C_VALUES:
            clf = make_classifier(C)          # plain SVM is enough for selection
            clf.fit(Xtr, y[tr])
            pred = clf.predict(Xva)
            oof[C][va] = pred
            scores[C].append({
                "accuracy": accuracy_score(y[va], pred),
                "balanced_accuracy": balanced_accuracy_score(y[va], pred),
                "macro_f1": f1_score(y[va], pred, average="macro"),
            })
 
        print(f"fold {i}/{n_splits} done in {time.time() - t0:.1f}s")
 
    results = []
    for C in C_VALUES:
        row = {
            "C": C,
            "accuracy": float(np.mean([s["accuracy"] for s in scores[C]])),
            "balanced_accuracy": float(np.mean([s["balanced_accuracy"] for s in scores[C]])),
            "macro_f1": float(np.mean([s["macro_f1"] for s in scores[C]])),
        }
        results.append(row)
        print(f"C={C:<4}  acc={row['accuracy']:.4f}  "
              f"bal_acc={row['balanced_accuracy']:.4f}  macro_f1={row['macro_f1']:.4f}")
 
    best = max(results, key=lambda r: r["macro_f1"])
    print("\nBest C:", best["C"], "| macro F1:", round(best["macro_f1"], 4))
 
    report = classification_report(
        y, oof[best["C"]], output_dict=True, zero_division=0
    )
    print("\nOut-of-fold classification report (best C):")
    print(classification_report(y, oof[best["C"]], zero_division=0))
 
    return best, results, report
 
 
# --------------------------------------------------
# FINAL MODEL
# --------------------------------------------------
 
def train_final_model(df, best):
    print("\nTRAINING FINAL CLASSIFIER ON ALL DATA")
    print("============================")
 
    pipeline = build_pipeline(best["C"], calibrate=USE_CALIBRATION)
 
    t0 = time.time()
    pipeline.fit(df["text"].values, df["model"].values)
    duration = time.time() - t0
 
    model_path = MODEL_DIR / "classifier.joblib"
    joblib.dump(pipeline, model_path, compress=3)
 
    labels = [str(c) for c in pipeline.classes_]
    with open(MODEL_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
 
    print("Model saved       :", model_path)
    print("Number of classes :", len(labels))
    print("Final train time  :", round(duration, 2), "seconds")
 
    return pipeline
 
 
def save_metrics(df, best, results, report):
    metrics = {
        "validation_method": f"GroupKFold by prompt_id ({N_FOLDS} folds)",
        "rows": int(len(df)),
        "classes": int(df["model"].nunique()),
        "best_parameters": {"C": best["C"], "calibrated": USE_CALIBRATION},
        "cv_accuracy": best["accuracy"],
        "cv_balanced_accuracy": best["balanced_accuracy"],
        "cv_macro_f1": best["macro_f1"],
        "all_experiments": results,
        "out_of_fold_classification_report": report,
        "note": (
            "Scores are means over grouped folds; C was chosen on the same "
            "folds so they may be slightly optimistic."
        ),
    }
    with open(MODEL_DIR / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
 
 
# --------------------------------------------------
# MAIN
# --------------------------------------------------
 
def main():
    start = time.time()
 
    print("=" * 50)
    print("NAME THAT LLM - TRAINING")
    print("=" * 50)
 
    df = load_dataset()
    best, results, report = cross_validate(df)
    train_final_model(df, best)
    save_metrics(df, best, results, report)
 
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print("Total time:", round(time.time() - start, 2), "seconds")
 
 
if __name__ == "__main__":
    main()