"""
Baseline ML Classifier: TF-IDF + Logistic Regression
-----------------------------------------------------
Trains, evaluates, and serializes a high-speed, calibrated baseline model
for bank transaction categorization using an end-to-end scikit-learn pipeline.
"""

import argparse
import json
import os
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessor import TransactionTextCleaner, clean_description


def build_pipeline() -> Pipeline:
    """Build the end-to-end TF-IDF + Logistic Regression classification pipeline."""
    return Pipeline([
        ("cleaner", TransactionTextCleaner()),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_features=12000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=2.5,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])


def train_and_evaluate(
    data_path: str = "data/raw/transactions.csv",
    model_save_path: str = "models/baseline_tfidf_model.joblib",
    report_save_path: str = "data/processed/evaluation_report.json",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train pipeline on dataset and save model artifacts and metrics."""
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_save_path), exist_ok=True)

    print(f"Loading transactions from {data_path}...")
    df = pd.read_csv(data_path)

    X = df["raw_description"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    pipeline = build_pipeline()
    print("Fitting TF-IDF + Logistic Regression pipeline...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    cls_report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "model_type": "TF-IDF + Logistic Regression",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_category_metrics": cls_report
    }

    # Save metrics report
    with open(report_save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model artifact
    joblib.dump(pipeline, model_save_path)
    print(f"Model saved to: {model_save_path}")
    print(f"Report saved to: {report_save_path}")
    print(f"\n--- EVALUATION SUMMARY ---")
    print(f"Accuracy:    {acc * 100:.2f}%")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    return pipeline, metrics


def predict_single(pipeline: Pipeline, raw_text: str) -> Dict[str, Any]:
    """Classify a single raw transaction description and return confidence score."""
    probas = pipeline.predict_proba([raw_text])[0]
    classes = pipeline.classes_
    best_idx = probas.argmax()
    best_category = classes[best_idx]
    best_prob = float(probas[best_idx])

    top_3_indices = probas.argsort()[-3:][::-1]
    top_3 = [
        {"category": str(classes[idx]), "confidence": round(float(probas[idx]), 4)}
        for idx in top_3_indices
    ]

    return {
        "raw_description": raw_text,
        "cleaned_description": clean_description(raw_text),
        "predicted_category": best_category,
        "confidence": round(best_prob, 4),
        "top_predictions": top_3
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/transactions.csv")
    parser.add_argument("--model-out", default="models/baseline_tfidf_model.joblib")
    parser.add_argument("--report-out", default="data/processed/evaluation_report.json")
    args = parser.parse_args()

    pipeline, metrics = train_and_evaluate(
        data_path=args.data,
        model_save_path=args.model_out,
        report_save_path=args.report_out
    )

    # Demo quick prediction
    sample_txn = "UPI/CR/99882312/SWIGGY/ICICI/order"
    result = predict_single(pipeline, sample_txn)
    print(f"\nSingle prediction demo:")
    print(json.dumps(result, indent=2))

