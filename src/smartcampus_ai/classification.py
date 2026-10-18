"""Inference helpers for the trained request classifier."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from smartcampus_ai.paths import CLASSIFIER_PATH


class ModelNotTrainedError(RuntimeError):
    """Raised when classification is requested before training."""


@lru_cache(maxsize=4)
def load_classifier(model_path: Path = CLASSIFIER_PATH) -> Any:
    """Load and cache a classifier from disk."""
    if not model_path.exists():
        raise ModelNotTrainedError(
            f"Modèle introuvable: {model_path}. Exécutez `uv run smartcampus-train`."
        )
    return joblib.load(model_path)


def classify_request(question: str, model_path: Path = CLASSIFIER_PATH) -> dict[str, float | str]:
    """Predict a category and its maximum class probability."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("La question ne peut pas être vide.")

    model = load_classifier(model_path)
    probabilities = model.predict_proba([normalized_question])[0]
    best_index = int(probabilities.argmax())
    return {
        "categorie": str(model.classes_[best_index]),
        "confiance": float(probabilities[best_index]),
    }
