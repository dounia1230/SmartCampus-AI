"""Business rules and end-to-end request orchestration."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any

from smartcampus_ai.classification import ModelNotTrainedError, classify_request
from smartcampus_ai.generation import (
    GenerationServiceError,
    InvalidCitationError,
    generate_answer,
)
from smartcampus_ai.retrieval import search_documents

CLASSIFICATION_THRESHOLD = 0.42
RETRIEVAL_THRESHOLD = 0.45
MAX_QUESTION_LENGTH = 1_000

Classifier = Callable[[str], dict[str, float | str]]
Retriever = Callable[..., list[dict[str, str | float]]]
Generator = Callable[..., dict[str, bool | str | list[str]]]


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _is_refund_request(question: str) -> bool:
    normalized = _normalize(question)
    patterns = (
        r"\brembours\w*",
        r"\brecuper\w*\s+(?:(?:mon|mes|l[' ]?)\s*)?argent\b",
        r"\brendre\s+(?:(?:mon|mes|l[' ]?)\s*)?argent\b",
        r"\bannul\w*\s+(?:un|le|mon)?\s*paiement\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def _requires_authentication(question: str) -> bool:
    normalized = _normalize(question)
    mentions_personal_document = re.search(
        r"\b(certificat|attestation|justificatif)\w*\b", normalized
    )
    requests_document = re.search(
        r"\b(obtenir|demander|recevoir|retirer|telecharger|recuperer|avoir|veux|souhaite|"
        r"besoin|fournir|mon|ma|mes)\b",
        normalized,
    )
    return bool(mentions_personal_document and requests_document)


def _conversation_reply(question: str) -> str | None:
    """Handle short social messages before administrative intent detection."""
    normalized = _normalize(question)
    compact = " ".join(re.sub(r"[^\w\s']", " ", normalized).split())
    welcome_echo = (
        "bonjour decrivez votre demande concernant une inscription une attestation ou un paiement"
    )
    greeting_pattern = r"(bonjour|bonsoir|salut|hello|coucou)( comment (ca va|allez vous))?"

    if compact == welcome_echo or re.fullmatch(greeting_pattern, compact):
        return (
            "Bonjour ! Comment puis-je vous aider concernant une inscription, "
            "une attestation ou un paiement ?"
        )
    if re.fullmatch(r"merci( beaucoup)?", compact):
        return "Avec plaisir ! Avez-vous une autre question administrative ?"
    if re.fullmatch(r"(au revoir|a bientot|bonne journee)", compact):
        return "Au revoir et bonne journée !"
    return None


def analyse_request(
    question: str,
    authenticated: bool = False,
    *,
    classifier: Classifier = classify_request,
    retriever: Retriever = search_documents,
    generator: Generator = generate_answer,
) -> dict[str, Any]:
    """Apply safety rules, classification, retrieval and grounded generation."""
    clean_question = question.strip()
    if not clean_question:
        return {"statut": "vide", "reponse": "Veuillez saisir une demande."}
    if len(clean_question) > MAX_QUESTION_LENGTH:
        return {
            "statut": "entree_invalide",
            "reponse": f"Votre demande doit contenir au maximum {MAX_QUESTION_LENGTH} caractères.",
        }

    if conversation_reply := _conversation_reply(clean_question):
        return {"statut": "conversation", "reponse": conversation_reply}

    if _is_refund_request(clean_question):
        return {
            "statut": "validation_humaine",
            "reponse": (
                "Toute demande de remboursement nécessite une validation humaine par le "
                "service financier. Aucune opération n'a été effectuée."
            ),
            "sources": ["PAY02"],
        }
    if _requires_authentication(clean_question) and not authenticated:
        return {
            "statut": "authentification_requise",
            "reponse": (
                "Vous devez être authentifié avant de demander un document personnel. "
                "Aucune demande n'a été transmise."
            ),
        }

    try:
        classification = classifier(clean_question)
    except ModelNotTrainedError as error:
        return {"statut": "modele_indisponible", "reponse": str(error)}

    category = str(classification["categorie"])
    confidence = float(classification["confiance"])
    if confidence < CLASSIFICATION_THRESHOLD:
        return {
            "statut": "clarification_requise",
            "reponse": (
                "Je ne suis pas assez certain de votre demande. Précisez si elle concerne "
                "une inscription, une attestation ou un paiement."
            ),
            "categorie": category,
            "confiance": confidence,
        }

    documents = retriever(
        clean_question,
        category,
        top_k=2,
        threshold=RETRIEVAL_THRESHOLD,
    )
    if not documents:
        return {
            "statut": "information_insuffisante",
            "reponse": "Je ne trouve aucune procédure suffisamment pertinente pour répondre.",
            "categorie": category,
            "confiance": confidence,
        }

    try:
        generated = generator(clean_question, documents)
    except InvalidCitationError as error:
        return {
            "statut": "reponse_non_fiable",
            "reponse": str(error),
            "categorie": category,
            "confiance": confidence,
        }
    except GenerationServiceError as error:
        return {
            "statut": "service_indisponible",
            "reponse": str(error),
            "categorie": category,
            "confiance": confidence,
        }

    if not generated["answerable"]:
        return {
            "statut": "information_insuffisante",
            "reponse": generated["reponse"],
            "sources": generated["sources"],
            "categorie": category,
            "confiance": confidence,
            "documents": documents,
        }

    return {
        "statut": "termine",
        "reponse": generated["reponse"],
        "sources": generated["sources"],
        "categorie": category,
        "confiance": confidence,
        "documents": documents,
    }
