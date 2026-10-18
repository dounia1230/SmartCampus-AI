"""Streamlit interface for SmartCampus AI."""

from __future__ import annotations

from typing import Any

import streamlit as st

from smartcampus_ai.orchestrator import analyse_request
from smartcampus_ai.paths import CLASSIFIER_PATH

STATUS_LABELS = {
    "termine": "Réponse documentée",
    "conversation": "Conversation",
    "validation_humaine": "Validation humaine requise",
    "authentification_requise": "Authentification requise",
    "clarification_requise": "Précision nécessaire",
    "information_insuffisante": "Information insuffisante",
    "service_indisponible": "Service local indisponible",
    "reponse_non_fiable": "Réponse non fiable",
    "modele_indisponible": "Modèle non entraîné",
    "entree_invalide": "Demande trop longue",
    "vide": "Demande vide",
}


def configure_page() -> None:
    st.set_page_config(
        page_title="SmartCampus AI",
        page_icon=":material/school:",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --sc-primary: #1d4ed8;
            --sc-background: #f8fafc;
            --sc-text: #172033;
            --sc-muted: #475569;
            --sc-border: #cbd5e1;
        }
        .stApp { background: var(--sc-background); color: var(--sc-text); }
        .block-container { max-width: 780px; padding-top: 2.5rem; }
        h1, h2, h3 { color: var(--sc-text); letter-spacing: -0.02em; }
        .sc-kicker {
            color: var(--sc-primary); font-size: 0.82rem; font-weight: 700;
            letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.35rem;
        }
        .sc-intro { color: var(--sc-muted); font-size: 1.05rem; max-width: 65ch; }
        .sc-notice {
            background: #eff6ff; border: 1px solid #93c5fd; border-radius: 0.7rem;
            color: #1e3a8a; padding: 0.85rem 1rem; margin: 1rem 0 1.4rem;
        }
        .sc-status {
            border: 1px solid var(--sc-border); border-radius: 999px; color: var(--sc-muted);
            display: inline-block; font-size: 0.78rem; font-weight: 700;
            margin-top: 0.5rem; padding: 0.2rem 0.65rem;
        }
        div[data-testid="stChatMessage"] {
            background: #ffffff; border: 1px solid #e2e8f0; border-radius: 0.85rem;
            padding: 0.25rem 0.45rem;
        }
        div[data-testid="stChatInput"] textarea { font-size: 1rem; }
        button:focus-visible, textarea:focus-visible, input:focus-visible {
            outline: 3px solid #f97316 !important; outline-offset: 2px !important;
        }
        @media (max-width: 640px) {
            .block-container { padding: 1.25rem 1rem 6rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important;
                transition: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> bool:
    with st.sidebar:
        st.header("Session étudiante")
        authenticated = st.checkbox(
            "Je suis authentifié(e)",
            value=False,
            help="Simulation pédagogique : aucune identité réelle n'est vérifiée.",
        )
        st.caption("L'authentification est simulée dans ce prototype.")
        st.divider()
        st.subheader("État du système")
        if CLASSIFIER_PATH.exists():
            st.success("Classifieur prêt")
        else:
            st.error("Classifieur à entraîner")
            st.code("uv run smartcampus-train", language="powershell")
        st.caption("Ollama est sollicité uniquement après classification et recherche.")
        st.divider()
        st.caption("Procédures fictives — démonstration universitaire locale.")
    return authenticated


def render_result(result: dict[str, Any]) -> None:
    st.markdown(str(result["reponse"]))
    status = str(result.get("statut", "information_insuffisante"))
    label = STATUS_LABELS.get(status, status.replace("_", " ").capitalize())
    st.markdown(f'<span class="sc-status">{label}</span>', unsafe_allow_html=True)

    sources = result.get("sources", [])
    documents = result.get("documents", [])
    if sources:
        st.caption("Sources : " + " · ".join(f"[{source}]" for source in sources))
    if documents:
        with st.expander("Voir les procédures retrouvées"):
            for document in documents:
                score = float(document["score"])
                st.markdown(f"**[{document['id']}] {document['titre']}** — pertinence {score:.0%}")
                st.write(document["texte"])

    confidence = result.get("confiance")
    category = result.get("categorie")
    if confidence is not None and category:
        st.caption(f"Catégorie : {category} · confiance du classifieur : {float(confidence):.0%}")


def initialize_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Bonjour. Décrivez votre demande concernant une inscription, "
                    "une attestation ou un paiement."
                ),
            }
        ]


def main() -> None:
    configure_page()
    authenticated = render_sidebar()

    st.markdown(
        '<div class="sc-kicker">Assistant universitaire local</div>', unsafe_allow_html=True
    )
    st.title("SmartCampus AI")
    st.markdown(
        '<p class="sc-intro">Trouvez la procédure administrative adaptée et vérifiez '
        "immédiatement les sources utilisées.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sc-notice"><strong>Important :</strong> ce prototype utilise des '
        "procédures fictives et n'effectue aucune opération administrative.</div>",
        unsafe_allow_html=True,
    )

    initialize_history()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "result" in message:
                render_result(message["result"])
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("Ex. : Comment obtenir une facture ?", max_chars=1_000):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyse de la demande et recherche des procédures…"):
                result = analyse_request(prompt, authenticated=authenticated)
            render_result(result)
        st.session_state.messages.append(
            {"role": "assistant", "content": result["reponse"], "result": result}
        )


if __name__ == "__main__":
    main()
