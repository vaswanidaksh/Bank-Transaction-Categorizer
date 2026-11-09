"""
Hybrid Ensemble Classifier: Rule Engine + ML Model
---------------------------------------------------
Integrates deterministic keyword matching with the TF-IDF ML model,
providing confidence scoring, fallback arbitration, and batch dataframe processing.
"""

import os
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessor import clean_description
from src.rule_engine import RuleEngine
from src.rule_manager import UserRuleManager

DEFAULT_MODEL_PATH = "models/baseline_tfidf_model.joblib"


class HybridTransactionClassifier:
    """
    Tiered classification system:
    1. Preprocesses raw bank string.
    2. Consults User Custom Rules (Tier 0 - user overrides).
    3. Consults RuleEngine (Tier 1 - known global/Indian merchants).
    4. Queries trained ML pipeline (Tier 2 - statistical generalization).
    5. Fallback for low confidence (Tier 3 - flags for review).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        min_confidence: float = 0.35
    ):
        self.min_confidence = min_confidence
        self.user_rule_manager = UserRuleManager()
        self.rule_engine = RuleEngine()
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        else:
            self.pipeline = None

    def add_user_rule(self, keyword: str, category: str) -> bool:
        """Add persistent user custom rule."""
        return self.user_rule_manager.add_rule(keyword, category)

    def delete_user_rule(self, keyword: str) -> bool:
        """Delete user custom rule."""
        return self.user_rule_manager.delete_rule(keyword)

    def get_user_rules(self) -> Dict[str, str]:
        """Get all custom user rules."""
        return self.user_rule_manager.load_rules()

    def classify_single(self, raw_description: str) -> Dict[str, Any]:
        """Classify a single raw transaction string."""
        cleaned = clean_description(raw_description)

        # Tier 0: User Custom Rule override
        user_match = self.user_rule_manager.match(cleaned)
        if user_match:
            return {
                "raw_description": raw_description,
                "cleaned_description": cleaned,
                "predicted_category": user_match,
                "confidence": 1.0,
                "engine_source": "user_custom_rule",
                "needs_review": False,
                "top_predictions": [{"category": user_match, "confidence": 1.0}]
            }

        # Tier 1: Built-in Rule Engine check
        rule_match = self.rule_engine.match(cleaned)
        if rule_match:
            category, conf = rule_match
            return {
                "raw_description": raw_description,
                "cleaned_description": cleaned,
                "predicted_category": category,
                "confidence": conf,
                "engine_source": "rule_engine",
                "needs_review": False,
                "top_predictions": [{"category": category, "confidence": conf}]
            }

        # Tier 2: ML Pipeline inference
        if self.pipeline is not None:
            probas = self.pipeline.predict_proba([raw_description])[0]
            classes = self.pipeline.classes_
            best_idx = probas.argmax()
            best_category = str(classes[best_idx])
            best_prob = float(probas[best_idx])

            top_3_indices = probas.argsort()[-3:][::-1]
            top_3 = [
                {"category": str(classes[idx]), "confidence": round(float(probas[idx]), 4)}
                for idx in top_3_indices
            ]

            # Tier 3: Low confidence fallback
            if best_prob < self.min_confidence:
                return {
                    "raw_description": raw_description,
                    "cleaned_description": cleaned,
                    "predicted_category": "Other / Uncategorized",
                    "confidence": round(best_prob, 4),
                    "engine_source": "low_confidence_fallback",
                    "needs_review": True,
                    "top_predictions": top_3
                }

            return {
                "raw_description": raw_description,
                "cleaned_description": cleaned,
                "predicted_category": best_category,
                "confidence": round(best_prob, 4),
                "engine_source": "ml_pipeline",
                "needs_review": False,
                "top_predictions": top_3
            }

        # Fallback if no model loaded
        return {
            "raw_description": raw_description,
            "cleaned_description": cleaned,
            "predicted_category": "Other / Uncategorized",
            "confidence": 0.0,
            "engine_source": "no_model_fallback",
            "needs_review": True,
            "top_predictions": []
        }

    def classify_batch(self, descriptions: List[str]) -> List[Dict[str, Any]]:
        """Batch classify a list of descriptions."""
        return [self.classify_single(desc) for desc in descriptions]

    def process_dataframe(
        self,
        df: pd.DataFrame,
        description_col: str = "raw_description"
    ) -> pd.DataFrame:
        """
        Takes a raw DataFrame, runs classification, and appends
        'predicted_category', 'confidence', 'cleaned_description', and 'engine_source'.
        """
        results = [self.classify_single(str(desc)) for desc in df[description_col]]
        df_out = df.copy()
        df_out["cleaned_description"] = [r["cleaned_description"] for r in results]
        df_out["predicted_category"] = [r["predicted_category"] for r in results]
        df_out["confidence"] = [r["confidence"] for r in results]
        df_out["engine_source"] = [r["engine_source"] for r in results]
        df_out["needs_review"] = [r["needs_review"] for r in results]
        return df_out


if __name__ == "__main__":
    classifier = HybridTransactionClassifier()
    samples = [
        "UPI/CR/112233/SWIGGY/ICICI/pay",
        "E-COM PURCHASE 4111XXXXXXXX1234 AMAZON INDIA",
        "NEFT-CR-N1234-INFOSYS SALARY NEFT",
        "SOME UNKNOWN UNLABELLED NARRATION XYZ 9999"
    ]
    for s in samples:
        res = classifier.classify_single(s)
        print(f"[{res['engine_source']}] {s[:35]}... -> {res['predicted_category']} ({res['confidence'] * 100:.1f}%)")

