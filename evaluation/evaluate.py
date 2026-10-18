"""Evaluate the classifier on held-out and out-of-domain questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score

from smartcampus_ai.evaluation_data import OUT_OF_DOMAIN, X_TEST, Y_TEST
from smartcampus_ai.orchestrator import CLASSIFICATION_THRESHOLD
from smartcampus_ai.paths import CLASSIFIER_PATH, PROJECT_ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "evaluation" / "artifacts"


def _threshold_diagnostics(
    predictions: np.ndarray,
    confidences: np.ndarray,
    out_confidences: np.ndarray,
) -> dict[str, dict[str, float]]:
    diagnostics: dict[str, dict[str, float]] = {}
    for threshold in (0.35, 0.40, 0.42, 0.45, 0.50):
        accepted = confidences >= threshold
        accepted_accuracy = (
            float(np.mean(predictions[accepted] == np.asarray(Y_TEST)[accepted]))
            if accepted.any()
            else 0.0
        )
        diagnostics[f"{threshold:.2f}"] = {
            "in_domain_coverage": float(accepted.mean()),
            "accepted_accuracy": accepted_accuracy,
            "out_of_domain_rejection": float((out_confidences < threshold).mean()),
        }
    return diagnostics


def evaluate_classifier(
    model_path: Path = CLASSIFIER_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict[str, Any]:
    """Compute metrics, persist a JSON report and render the confusion matrix."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable: {model_path}. Exécutez `uv run smartcampus-train`."
        )
    model = joblib.load(model_path)
    predictions = np.asarray(model.predict(X_TEST))
    probabilities = np.asarray(model.predict_proba(X_TEST))
    confidences = probabilities.max(axis=1)
    out_confidences = np.asarray(model.predict_proba(OUT_OF_DOMAIN)).max(axis=1)

    report = classification_report(Y_TEST, predictions, output_dict=True, zero_division=0)
    metrics: dict[str, Any] = {
        "test_size": len(X_TEST),
        "accuracy": float(accuracy_score(Y_TEST, predictions)),
        "f1_macro": float(f1_score(Y_TEST, predictions, average="macro")),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "mean_confidence": float(confidences.mean()),
        "classification_report": report,
        "threshold_diagnostics": _threshold_diagnostics(
            predictions,
            confidences,
            out_confidences,
        ),
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / "classifier_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        Y_TEST,
        predictions,
        labels=["inscriptions", "attestations", "paiements"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title("Matrice de confusion — classifieur SmartCampus AI")
    figure.tight_layout()
    figure.savefig(artifacts_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    return metrics


def main() -> None:
    metrics = evaluate_classifier()
    print(f"Exactitude : {metrics['accuracy']:.3f}")
    print(f"F1 macro   : {metrics['f1_macro']:.3f}")
    print(f"Confiance moyenne : {metrics['mean_confidence']:.3f}")
    threshold_metrics = metrics["threshold_diagnostics"][f"{CLASSIFICATION_THRESHOLD:.2f}"]
    print(
        f"Seuil {CLASSIFICATION_THRESHOLD:.2f} — couverture: "
        f"{threshold_metrics['in_domain_coverage']:.1%}, rejet hors domaine: "
        f"{threshold_metrics['out_of_domain_rejection']:.1%}"
    )
    print(f"Artefacts : {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
