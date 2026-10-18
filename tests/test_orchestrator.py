from typing import Never

from smartcampus_ai.orchestrator import analyse_request


def fail_if_called(*_: object, **__: object) -> Never:
    raise AssertionError("Cette dépendance ne devait pas être appelée.")


def confident_classification(_: str) -> dict[str, float | str]:
    return {"categorie": "paiements", "confiance": 0.91}


def test_empty_question_stops_before_ai_calls() -> None:
    result = analyse_request(" ", classifier=fail_if_called)

    assert result["statut"] == "vide"


def test_refund_requires_human_before_ai_calls() -> None:
    result = analyse_request("Je veux récupérer mon argent", classifier=fail_if_called)

    assert result["statut"] == "validation_humaine"
    assert result["sources"] == ["PAY02"]


def test_personal_document_requires_authentication() -> None:
    result = analyse_request("Je veux mon certificat de scolarité", classifier=fail_if_called)

    assert result["statut"] == "authentification_requise"


def test_low_confidence_requests_clarification() -> None:
    result = analyse_request(
        "J'ai un problème",
        classifier=lambda _: {"categorie": "paiements", "confiance": 0.2},
        retriever=fail_if_called,
    )

    assert result["statut"] == "clarification_requise"


def test_no_relevant_document_abstains() -> None:
    result = analyse_request(
        "Quel est le menu du restaurant ?",
        classifier=confident_classification,
        retriever=lambda *_, **__: [],
        generator=fail_if_called,
    )

    assert result["statut"] == "information_insuffisante"


def test_successful_pipeline_returns_metadata() -> None:
    documents = [
        {
            "id": "PAY01",
            "categorie": "paiements",
            "titre": "Factures",
            "texte": "Contactez la comptabilité.",
            "score": 0.9,
        }
    ]
    result = analyse_request(
        "Comment obtenir une facture ?",
        classifier=confident_classification,
        retriever=lambda *_, **__: documents,
        generator=lambda *_: {
            "reponse": "Contactez la comptabilité [PAY01].",
            "sources": ["PAY01"],
        },
    )

    assert result["statut"] == "termine"
    assert result["sources"] == ["PAY01"]
    assert result["categorie"] == "paiements"
