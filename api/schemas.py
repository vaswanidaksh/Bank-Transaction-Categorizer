"""
Pydantic Schemas for FastAPI Request/Response Validation
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    raw_description: str = Field(
        ...,
        examples=["UPI/CR/4324234/SWIGGY/ICICI/order"],
        description="Raw narrative string from bank statement."
    )
    min_confidence: Optional[float] = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Threshold below which transaction is tagged as Other / Uncategorized."
    )


class BatchTransactionRequest(BaseModel):
    transactions: List[str] = Field(
        ...,
        description="List of raw bank statement transaction descriptions."
    )
    min_confidence: Optional[float] = Field(
        0.35,
        ge=0.0,
        le=1.0
    )


class PredictionItem(BaseModel):
    category: str
    confidence: float


class TransactionResponse(BaseModel):
    raw_description: str
    cleaned_description: str
    predicted_category: str
    confidence: float
    engine_source: str
    needs_review: bool
    top_predictions: List[PredictionItem]


class BatchTransactionResponse(BaseModel):
    total_processed: int
    predictions: List[TransactionResponse]


class FeatureImportanceItem(BaseModel):
    token: str
    importance: float
    tfidf_weight: float


class ExplainResponse(BaseModel):
    raw_description: str
    cleaned_description: str
    predicted_category: str
    confidence: float
    top_features: List[FeatureImportanceItem]


class SystemHealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    supported_categories_count: int

