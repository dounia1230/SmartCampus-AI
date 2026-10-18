"""Validated knowledge-base loading and semantic retrieval."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from smartcampus_ai.paths import PROCEDURES_PATH

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingModel(Protocol):
    def encode_document(self, sentences: list[str], **kwargs: Any) -> Any: ...

    def encode_query(self, sentences: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class Procedure:
    id: str
    categorie: str
    titre: str
    texte: str


@dataclass(frozen=True)
class RetrievedProcedure:
    id: str
    categorie: str
    titre: str
    texte: str
    score: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def load_procedures(path: Path = PROCEDURES_PATH) -> list[Procedure]:
    """Load procedures from JSON and fail early on malformed records."""
    with path.open(encoding="utf-8") as file:
        raw_procedures = json.load(file)

    if not isinstance(raw_procedures, list) or not raw_procedures:
        raise ValueError("La base de procédures doit être une liste non vide.")

    procedures: list[Procedure] = []
    seen_ids: set[str] = set()
    required = {"id", "categorie", "titre", "texte"}
    for index, raw in enumerate(raw_procedures):
        if not isinstance(raw, dict):
            raise ValueError(f"La procédure #{index + 1} doit être un objet JSON.")
        missing = required.difference(raw)
        if missing:
            raise ValueError(
                f"Champs manquants dans la procédure #{index + 1}: {', '.join(sorted(missing))}"
            )
        values = {key: str(raw[key]).strip() for key in required}
        if any(not value for value in values.values()):
            raise ValueError(f"La procédure #{index + 1} contient un champ vide.")
        if values["id"] in seen_ids:
            raise ValueError(f"Identifiant de procédure dupliqué: {values['id']}")
        seen_ids.add(values["id"])
        procedures.append(Procedure(**values))
    return procedures


@lru_cache(maxsize=2)
def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> EmbeddingModel:
    """Load the pretrained model only when retrieval is first requested."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SemanticRetriever:
    """In-memory cosine-similarity index for the small procedure corpus."""

    def __init__(
        self,
        procedures: list[Procedure] | None = None,
        model: EmbeddingModel | None = None,
    ) -> None:
        self.procedures = procedures or load_procedures()
        self.model = model or load_embedding_model()
        texts = [f"{procedure.titre}. {procedure.texte}" for procedure in self.procedures]
        self.document_vectors = np.asarray(
            self.model.encode_document(texts, normalize_embeddings=True),
            dtype=float,
        )
        if self.document_vectors.shape[0] != len(self.procedures):
            raise ValueError("Le modèle n'a pas produit un vecteur par procédure.")

    def search(
        self,
        question: str,
        category: str,
        *,
        top_k: int = 2,
        threshold: float = 0.45,
    ) -> list[RetrievedProcedure]:
        """Return the best procedures in ``category`` above ``threshold``."""
        if not question.strip():
            raise ValueError("La question ne peut pas être vide.")
        if top_k < 1:
            raise ValueError("top_k doit être supérieur ou égal à 1.")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("Le seuil doit être compris entre -1 et 1.")

        candidate_indices = [
            index
            for index, procedure in enumerate(self.procedures)
            if procedure.categorie == category
        ]
        if not candidate_indices:
            return []

        query_vector = np.asarray(
            self.model.encode_query(question.strip(), normalize_embeddings=True),
            dtype=float,
        ).reshape(-1)
        candidate_vectors = self.document_vectors[candidate_indices]
        scores = candidate_vectors @ query_vector
        ranked = sorted(
            zip(candidate_indices, scores, strict=True), key=lambda item: item[1], reverse=True
        )

        results: list[RetrievedProcedure] = []
        for index, score in ranked:
            if float(score) < threshold:
                continue
            procedure = self.procedures[index]
            results.append(RetrievedProcedure(**asdict(procedure), score=float(score)))
            if len(results) == top_k:
                break
        return results


@lru_cache(maxsize=1)
def get_retriever() -> SemanticRetriever:
    return SemanticRetriever()


def search_documents(
    question: str,
    category: str,
    *,
    top_k: int = 2,
    threshold: float = 0.45,
) -> list[dict[str, str | float]]:
    """Convenience API returning serializable retrieval results."""
    return [
        result.to_dict()
        for result in get_retriever().search(
            question,
            category,
            top_k=top_k,
            threshold=threshold,
        )
    ]
