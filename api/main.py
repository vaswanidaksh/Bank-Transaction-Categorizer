"""
FastAPI Application for Bank Transaction Categorization
-------------------------------------------------------
Provides high-throughput, low-latency REST endpoints for single and
batch transaction categorization and explainability with API Key authentication.
"""

import os
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from api.schemas import (
    TransactionRequest,
    TransactionResponse,
    BatchTransactionRequest,
    BatchTransactionResponse,
    ExplainResponse,
    SystemHealthResponse
)
from src.classifier import HybridTransactionClassifier
from src.data_generator import CATEGORIES
from src.explainability import explain_prediction

# Environment & Security Configuration
API_KEY_ENV = os.getenv("API_KEY", "").strip()
ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "*").strip()

if ALLOWED_ORIGINS_ENV == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()]

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """
    Validates API key from request headers.
    If API_KEY is set in environment, enforces strict validation.
    If API_KEY is not set (development mode), allows requests.
    """
    if not API_KEY_ENV:
        # Development mode: open access
        return True

    if not api_key or api_key != API_KEY_ENV:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key. Provide a valid 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return True


app = FastAPI(
    title="Bank Transaction Categorizer API",
    description="Intelligent financial NLP microservice for categorizing bank statements (UPI, NEFT, IMPS, POS, Cards).",
    version="1.0.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate singleton classifier
classifier = HybridTransactionClassifier()


@app.get("/health", response_model=SystemHealthResponse, tags=["System"])
def health_check():
    """Health check endpoint checking model readiness (Public)."""
    return {
        "status": "healthy",
        "service": "bank-transaction-categorizer",
        "model_loaded": classifier.pipeline is not None,
        "supported_categories_count": len(CATEGORIES)
    }


@app.get("/api/v1/categories", response_model=List[str], tags=["Metadata"])
def get_categories():
    """Returns the list of 13 supported transaction categories (Public)."""
    return CATEGORIES


@app.post(
    "/api/v1/predict",
    response_model=TransactionResponse,
    tags=["Inference"],
    dependencies=[Depends(verify_api_key)]
)
def predict_transaction(payload: TransactionRequest):
    """
    Classify a single bank statement description string.
    Protected by API Key header when API_KEY environment variable is configured.
    """
    if not payload.raw_description.strip():
        raise HTTPException(status_code=400, detail="Transaction description cannot be empty.")

    res = classifier.classify_single(payload.raw_description)
    return res


@app.post(
    "/api/v1/predict-batch",
    response_model=BatchTransactionResponse,
    tags=["Inference"],
    dependencies=[Depends(verify_api_key)]
)
def predict_batch(payload: BatchTransactionRequest):
    """
    Batch classify a list of transaction narratives.
    Protected by API Key header when API_KEY environment variable is configured.
    """
    if not payload.transactions:
        raise HTTPException(status_code=400, detail="Transaction list cannot be empty.")

    results = classifier.classify_batch(payload.transactions)
    return {
        "total_processed": len(results),
        "predictions": results
    }


@app.post(
    "/api/v1/explain",
    response_model=ExplainResponse,
    tags=["Explainability"],
    dependencies=[Depends(verify_api_key)]
)
def explain_transaction(payload: TransactionRequest):
    """
    Explain WHY a transaction was classified into a specific category,
    returning feature importance attribution scores for individual tokens.
    """
    if classifier.pipeline is None:
        raise HTTPException(status_code=503, detail="ML Pipeline not available for feature explanations.")

    explanation = explain_prediction(classifier.pipeline, payload.raw_description)
    return explanation


@app.get("/api/v1/rules", response_model=Dict[str, str], tags=["Custom Rules"])
def get_user_rules():
    """List persistent custom user rules."""
    return classifier.get_user_rules()


@app.post("/api/v1/rules", tags=["Custom Rules"], dependencies=[Depends(verify_api_key)])
def add_user_rule(keyword: str, category: str):
    """Add or update a persistent custom user rule."""
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Category must be one of: {CATEGORIES}")
    success = classifier.add_user_rule(keyword, category)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add rule. Keyword cannot be empty.")
    return {"status": "success", "keyword": keyword, "category": category}


@app.delete("/api/v1/rules/{keyword}", tags=["Custom Rules"], dependencies=[Depends(verify_api_key)])
def delete_user_rule(keyword: str):
    """Delete a custom user rule."""
    deleted = classifier.delete_user_rule(keyword)
    if not deleted:
        raise HTTPException(status_code=404, detail="Rule keyword not found.")
    return {"status": "deleted", "keyword": keyword}
