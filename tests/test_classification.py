from pathlib import Path

import pandas as pd
import pytest

from smartcampus_ai.classification import classify_request, load_classifier
from smartcampus_ai.training import load_training_data, train_classifier


@pytest.fixture
def trained_model(tmp_path: Path) -> Path:
    data_path = tmp_path / "training.csv"
    model_path = tmp_path / "classifier.joblib"
    examples = pd.DataFrame(
        {
            "question": [
                "dossier inscription admission",
                "documents pour inscription",
                "certificat scolarité attestation",
                "attestation de réussite",
                "facture paiement frais",
                "reçu remboursement paiement",
            ],
            "categorie": [
                "inscriptions",
                "inscriptions",
                "attestations",
                "attestations",
                "paiements",
                "paiements",
            ],
        }
    )
    examples.to_csv(data_path, index=False)
    train_classifier(data_path, model_path)
    load_classifier.cache_clear()
    return model_path


def test_classifier_returns_category_and_probability(trained_model: Path) -> None:
    result = classify_request("Je voudrais une facture de paiement", trained_model)

    assert result["categorie"] == "paiements"
    assert 0.0 <= result["confiance"] <= 1.0


def test_classifier_rejects_empty_question(trained_model: Path) -> None:
    with pytest.raises(ValueError, match="vide"):
        classify_request("   ", trained_model)


def test_training_data_requires_expected_columns(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.csv"
    pd.DataFrame({"text": ["bonjour"]}).to_csv(invalid_path, index=False)

    with pytest.raises(ValueError, match="Colonnes manquantes"):
        load_training_data(invalid_path)
