import os
import pytest
from fastapi.testclient import TestClient
from api.main import app
import api.main as api_module

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["supported_categories_count"] == 13

def test_categories_endpoint():
    response = client.get("/api/v1/categories")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 13
    assert "Food & Dining" in data

def test_predict_single_endpoint():
    payload = {"raw_description": "UPI/CR/998811/SWIGGY/ICICI/order"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_category"] == "Food & Dining"
    assert data["confidence"] > 0.5
    assert "cleaned_description" in data

def test_predict_batch_endpoint():
    payload = {
        "transactions": [
            "POS 4421XXXX9102 DMART BLR",
            "UPI/CR/443322/UBER INDIA/SBIN/ride",
            "ACH D- NETFLIX ENTERTAINMENT"
        ]
    }
    response = client.post("/api/v1/predict-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 3
    preds = data["predictions"]
    assert preds[0]["predicted_category"] == "Groceries"
    assert preds[1]["predicted_category"] == "Transport"
    assert preds[2]["predicted_category"] == "Entertainment"

def test_explain_endpoint():
    payload = {"raw_description": "POS 4421XXXX1234 DMART BANNERGHATTA"}
    response = client.post("/api/v1/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_category"] == "Groceries"
    assert len(data["top_features"]) > 0
    assert any("dmart" in f["token"] for f in data["top_features"])

def test_api_key_authentication_enforcement(monkeypatch):
    """Verify that when API_KEY is set in environment, unauthorized requests are rejected."""
    # Set mock API key
    monkeypatch.setattr(api_module, "API_KEY_ENV", "test-secret-key-123")

    payload = {"raw_description": "UPI/CR/111/ZOMATO/pay"}

    # Attempt 1: Without header -> 401 Unauthorized
    resp_unauthorized = client.post("/api/v1/predict", json=payload)
    assert resp_unauthorized.status_code == 401
    assert "Invalid or missing API Key" in resp_unauthorized.json()["detail"]

    # Attempt 2: With incorrect header -> 401 Unauthorized
    resp_wrong = client.post(
        "/api/v1/predict",
        json=payload,
        headers={"X-API-Key": "wrong-key"}
    )
    assert resp_wrong.status_code == 401

    # Attempt 3: With correct header -> 200 OK
    resp_ok = client.post(
        "/api/v1/predict",
        json=payload,
        headers={"X-API-Key": "test-secret-key-123"}
    )
    assert resp_ok.status_code == 200
    assert resp_ok.json()["predicted_category"] == "Food & Dining"
