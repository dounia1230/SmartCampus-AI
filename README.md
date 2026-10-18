# SmartCampus AI

Assistant administratif universitaire en français, construit comme un projet pédagogique
de bout en bout : classification supervisée, recherche sémantique (RAG), règles métier,
génération locale avec Ollama et interface Streamlit.

> Les procédures incluses sont fictives. Ce prototype ne doit pas être utilisé comme source
> officielle ni exécuter de véritables opérations administratives.

## Prérequis

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/)
- Ollama (facultatif jusqu'à l'étape de génération)

## Démarrage rapide

```powershell
uv sync
uv run smartcampus-train
uv run streamlit run app.py
```

Le modèle de classification est généré dans `models/classifier.joblib`. Le premier lancement
de la recherche sémantique télécharge le modèle d'embeddings configuré.

Dans la barre latérale, l'option d'authentification est une simulation pédagogique. Elle ne doit
jamais être confondue avec un véritable contrôle d'identité.

## Commandes utiles

```powershell
# Entraîner le classifieur
uv run smartcampus-train

# Lancer les tests rapides (sans téléchargement de modèle)
uv run pytest

# Vérifier le style
uv run ruff check .

# Évaluer le classifieur et produire la matrice de confusion
uv run python -m evaluation.evaluate
```

Consultez [docs/SETUP.md](docs/SETUP.md) pour l'installation détaillée et
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) pour comprendre le flux de traitement.
