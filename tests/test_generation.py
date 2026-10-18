from types import SimpleNamespace

import pytest

from smartcampus_ai.generation import (
    GenerationServiceError,
    InvalidCitationError,
    generate_answer,
)

DOCUMENTS = [
    {
        "id": "PAY01",
        "categorie": "paiements",
        "titre": "Factures",
        "texte": "Contactez le service comptable.",
        "score": 0.9,
    }
]


def test_generate_answer_returns_verified_sources() -> None:
    def fake_chat(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            message=SimpleNamespace(content="Contactez le service comptable [PAY01].")
        )

    result = generate_answer("Comment avoir une facture ?", DOCUMENTS, chat=fake_chat)

    assert result["sources"] == ["PAY01"]


def test_generate_answer_rejects_invented_source() -> None:
    def fake_chat(**_: object) -> SimpleNamespace:
        return SimpleNamespace(message=SimpleNamespace(content="Utilisez le portail [PAY99]."))

    with pytest.raises(InvalidCitationError, match="non autorisées"):
        generate_answer("Comment avoir une facture ?", DOCUMENTS, chat=fake_chat)


def test_generate_answer_wraps_service_failure() -> None:
    def failing_chat(**_: object) -> None:
        raise ConnectionError("offline")

    with pytest.raises(GenerationServiceError, match="indisponible"):
        generate_answer("Comment avoir une facture ?", DOCUMENTS, chat=failing_chat)
