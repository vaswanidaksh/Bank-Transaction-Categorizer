"""
Model Explainability & Feature Attribution
------------------------------------------
Extracts linear feature contributions (token importance) to explain
WHY a bank transaction was classified into a specific category.
"""

from typing import List, Dict, Any
from sklearn.pipeline import Pipeline
from src.preprocessor import clean_description


def explain_prediction(
    pipeline: Pipeline,
    raw_text: str,
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Computes exact linear feature attribution scores for a given text.
    Returns the tokens that most strongly pushed the model toward the predicted category.
    """
    cleaned = clean_description(raw_text)
    if not cleaned:
        return {
            "raw_description": raw_text,
            "cleaned_description": "",
            "predicted_category": "Other / Uncategorized",
            "top_features": []
        }

    # Extract vectorizer and classifier from pipeline
    tfidf: Any = pipeline.named_steps["tfidf"]
    clf: Any = pipeline.named_steps["clf"]

    # Transform text to tf-idf vector
    feature_vector = tfidf.transform([cleaned])
    probas = pipeline.predict_proba([raw_text])[0]
    best_idx = probas.argmax()
    predicted_category = clf.classes_[best_idx]

    # Get non-zero indices and feature names
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[best_idx]  # Shape: (n_features,)

    non_zero_indices = feature_vector.indices
    explanations = []

    for idx in non_zero_indices:
        token = feature_names[idx]
        tfidf_val = feature_vector[0, idx]
        weight = coefs[idx]
        contribution = float(tfidf_val * weight)
        explanations.append({
            "token": token,
            "importance": round(contribution, 4),
            "tfidf_weight": round(float(tfidf_val), 4)
        })

    # Sort descending by contribution
    explanations.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "raw_description": raw_text,
        "cleaned_description": cleaned,
        "predicted_category": predicted_category,
        "confidence": round(float(probas[best_idx]), 4),
        "top_features": explanations[:top_n]
    }


if __name__ == "__main__":
    import joblib
    model = joblib.load("models/baseline_tfidf_model.joblib")
    sample = "POS 4421XXXX1234 DMART BANNERGHATTA BLR"
    exp = explain_prediction(model, sample)
    print("Explainability Demo:")
    print(exp)

