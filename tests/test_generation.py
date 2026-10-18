import json
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
    received: dict[str, object] = {}

    def fake_chat(**kwargs: object) -> SimpleNamespace:
        received.update(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(
                content=json.dumps(
                    {
                        "answerable": True,
                        "reponse": "Contactez le service comptable.",
                        "sources": ["PAY01"],
                    }
                )
            )
        )

    result = generate_answer("Comment avoir une facture ?", DOCUMENTS, chat=fake_chat)

    assert result["sources"] == ["PAY01"]
    assert result["answerable"] is True
    assert received["model"] == "qwen3.5:4b"
    assert received["format"]["properties"]["sources"]["items"]["enum"] == ["PAY01"]


def test_generate_answer_rejects_invented_source() -> None:
    def fake_chat(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            message=SimpleNamespace(
                content=json.dumps(
                    {
                        "answerable": True,
                        "reponse": "Utilisez le portail [PAY99].",
                        "sources": ["PAY99"],
                    }
                )
            )
        )

    with pytest.raises(InvalidCitationError, match="pas autorisées"):
        generate_answer("Comment avoir une facture ?", DOCUMENTS, chat=fake_chat)


def test_generate_answer_wraps_service_failure() -> None:
    def failing_chat(**_: object) -> None:
        raise ConnectionError("offline")

    with pytest.raises(GenerationServiceError, match="indisponible"):
        generate_answer("Comment avoir une facture ?", DOCUMENTS, chat=failing_chat)


def test_generate_answer_can_report_insufficient_context() -> None:
    def fake_chat(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            message=SimpleNamespace(
                content=json.dumps(
                    {
                        "answerable": False,
                        "reponse": "Le document ne précise pas les moyens acceptés.",
                        "sources": ["PAY01"],
                    }
                )
            )
        )

    result = generate_answer("Quels moyens sont acceptés ?", DOCUMENTS, chat=fake_chat)

    assert result["answerable"] is False
