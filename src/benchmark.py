"""
Model Benchmarking & Comparative Evaluation Suite
-------------------------------------------------
Evaluates multiple text classification algorithms on the identical
preprocessed transaction dataset across accuracy, Macro-F1, training
time, and inference latency per sample.
"""

import json
import os
import time
from typing import Dict, Any
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessor import TransactionTextCleaner


def run_benchmark(
    data_path: str = "data/raw/transactions.csv",
    output_path: str = "data/processed/model_benchmark.json"
) -> Dict[str, Any]:
    """Train and evaluate 5 candidate classification architectures."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(data_path)

    X = df["raw_description"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression (L2)": LogisticRegression(
            C=2.5, max_iter=1000, class_weight="balanced"
        ),
        "Linear SVC (Calibrated)": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
        ),
        "SGD Classifier (Modified Huber)": SGDClassifier(
            loss="modified_huber", max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest (100 Trees)": RandomForestClassifier(
            n_estimators=100, max_depth=25, class_weight="balanced", random_state=42
        )
    }

    results = []

    cleaner = TransactionTextCleaner()
    tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=12000, sublinear_tf=True)

    print("Preprocessing training & testing tokens...")
    X_train_clean = cleaner.transform(X_train)
    X_test_clean = cleaner.transform(X_test)

    X_train_vec = tfidf.fit_transform(X_train_clean)
    X_test_vec = tfidf.transform(X_test_clean)

    for name, clf in models.items():
        print(f"\nTraining [{name}]...")
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        train_time = round(time.time() - t0, 4)

        # Measure latency on test set
        t1 = time.time()
        y_pred = clf.predict(X_test_vec)
        eval_time = time.time() - t1
        latency_per_sample_ms = round((eval_time / len(X_test)) * 1000, 4)

        acc = round(float(accuracy_score(y_test, y_pred)), 4)
        macro_f1 = round(float(f1_score(y_test, y_pred, average="macro")), 4)
        weighted_f1 = round(float(f1_score(y_test, y_pred, average="weighted")), 4)

        print(f"  -> Accuracy: {acc * 100:.2f}% | Macro F1: {macro_f1} | Latency: {latency_per_sample_ms}ms")

        results.append({
            "model_name": name,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "training_time_sec": train_time,
            "latency_per_sample_ms": latency_per_sample_ms
        })

    benchmark_data = {
        "dataset_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "models": results
    }

    with open(output_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\nSaved benchmark comparison to {output_path}")
    return benchmark_data


if __name__ == "__main__":
    run_benchmark()

