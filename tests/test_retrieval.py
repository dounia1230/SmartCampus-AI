from pathlib import Path

import numpy as np
import pytest

from smartcampus_ai.retrieval import Procedure, SemanticRetriever, load_procedures


class FakeEmbeddingModel:
    @staticmethod
    def _vector(text: str) -> np.ndarray:
        lowered = text.lower()
        return np.array(
            [
                float("inscri" in lowered or "admission" in lowered),
                float("certificat" in lowered or "attestation" in lowered),
                float("facture" in lowered or "paiement" in lowered),
            ]
        )

    def encode_document(self, sentences: list[str], **_: object) -> np.ndarray:
        return np.stack([self._vector(sentence) for sentence in sentences])

    def encode_query(self, sentence: str, **_: object) -> np.ndarray:
        return self._vector(sentence)


@pytest.fixture
def retriever() -> SemanticRetriever:
    procedures = [
        Procedure("INS01", "inscriptions", "Inscription", "Déposer un dossier."),
        Procedure("ATT01", "attestations", "Certificat", "Demander une attestation."),
        Procedure("PAY01", "paiements", "Facture", "Contacter le service de paiement."),
    ]
    return SemanticRetriever(procedures=procedures, model=FakeEmbeddingModel())


def test_search_returns_matching_document(retriever: SemanticRetriever) -> None:
    results = retriever.search("Je veux une facture", "paiements", threshold=0.5)

    assert [result.id for result in results] == ["PAY01"]
    assert results[0].score == pytest.approx(1.0)


def test_search_respects_category_filter(retriever: SemanticRetriever) -> None:
    assert retriever.search("Je veux une facture", "attestations", threshold=0.5) == []


def test_search_rejects_invalid_top_k(retriever: SemanticRetriever) -> None:
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("inscription", "inscriptions", top_k=0)


def test_procedure_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    procedures_path = tmp_path / "procedures.json"
    procedures_path.write_text(
        """[
          {"id":"P1","categorie":"x","titre":"A","texte":"Texte A"},
          {"id":"P1","categorie":"x","titre":"B","texte":"Texte B"}
        ]""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dupliqué"):
        load_procedures(procedures_path)
