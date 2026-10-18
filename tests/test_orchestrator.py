from typing import Never

from smartcampus_ai.generation import InvalidCitationError
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


def test_greeting_stops_before_administrative_rules() -> None:
    result = analyse_request("Bonjour", classifier=fail_if_called)

    assert result["statut"] == "conversation"
    assert result["reponse"].startswith("Bonjour")


def test_echoed_welcome_message_is_treated_as_conversation() -> None:
    result = analyse_request(
        "Bonjour. Décrivez votre demande concernant une inscription, "
        "une attestation ou un paiement.",
        classifier=fail_if_called,
    )

    assert result["statut"] == "conversation"


def test_informational_attestation_question_does_not_require_authentication() -> None:
    result = analyse_request(
        "Quel service délivre les attestations ?",
        classifier=lambda _: {"categorie": "attestations", "confiance": 0.9},
        retriever=lambda *_, **__: [],
    )

    assert result["statut"] == "information_insuffisante"


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
            "answerable": True,
            "reponse": "Contactez la comptabilité [PAY01].",
            "sources": ["PAY01"],
        },
    )

    assert result["statut"] == "termine"
    assert result["sources"] == ["PAY01"]
    assert result["categorie"] == "paiements"


def test_generated_insufficiency_uses_information_status() -> None:
    documents = [
        {
            "id": "PAY01",
            "categorie": "paiements",
            "titre": "Factures",
            "texte": "Contactez la comptabilité.",
            "score": 0.7,
        }
    ]
    result = analyse_request(
        "Quels moyens de paiement sont acceptés ?",
        classifier=confident_classification,
        retriever=lambda *_, **__: documents,
        generator=lambda *_: {
            "answerable": False,
            "reponse": "Cette procédure ne précise pas les moyens acceptés [PAY01].",
            "sources": ["PAY01"],
        },
    )

    assert result["statut"] == "information_insuffisante"
    assert result["sources"] == ["PAY01"]


def test_invalid_citation_is_not_reported_as_service_outage() -> None:
    documents = [
        {
            "id": "PAY01",
            "categorie": "paiements",
            "titre": "Factures",
            "texte": "Contactez la comptabilité.",
            "score": 0.7,
        }
    ]

    def invalid_generator(*_: object) -> Never:
        raise InvalidCitationError("Citation incorrecte.")

    result = analyse_request(
        "Comment obtenir une facture ?",
        classifier=confident_classification,
        retriever=lambda *_, **__: documents,
        generator=invalid_generator,
    )

    assert result["statut"] == "reponse_non_fiable"
