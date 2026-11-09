import pytest
import pandas as pd
from src.classifier import HybridTransactionClassifier

@pytest.fixture(scope="module")
def classifier():
    return HybridTransactionClassifier()

def test_rule_engine_exact_match(classifier):
    res = classifier.classify_single("UPI/CR/999/ZOMATO/ICICI/order")
    assert res["predicted_category"] == "Food & Dining"
    assert res["engine_source"] == "rule_engine"
    assert res["confidence"] >= 0.95

def test_salary_detection(classifier):
    res = classifier.classify_single("BY TRANSFER-SALARY CREDIT TCS OCTOBER")
    assert res["predicted_category"] == "Income"

def test_low_confidence_fallback(classifier):
    res = classifier.classify_single("RANDOM GIBBERISH ABCXYZ1234567")
    assert res["predicted_category"] == "Other / Uncategorized"
    assert res["needs_review"] is True

def test_process_dataframe(classifier):
    df = pd.DataFrame({
        "raw_description": [
            "POS 4000XXXXXXXX1234 DMART",
            "UPI/CR/123/UBER INDIA/SBIN/pay"
        ]
    })
    processed = classifier.process_dataframe(df)
    assert "predicted_category" in processed.columns
    assert "confidence" in processed.columns
    assert processed.iloc[0]["predicted_category"] == "Groceries"
    assert processed.iloc[1]["predicted_category"] == "Transport"

