# Installation locale

## 1. Synchroniser l'environnement

Depuis la racine du dépôt :

```powershell
uv sync
```

`uv.lock` est la source reproductible des versions résolues. Il ne faut pas maintenir un
`requirements.txt` séparé.

## 2. Entraîner le classifieur

```powershell
uv run smartcampus-train
```

Le fichier `models/classifier.joblib` est un artefact local généré et ignoré par Git.

## 3. Préparer Ollama

Installez Ollama, démarrez son service puis récupérez le modèle configuré :

```powershell
ollama pull llama3.2:3b
```

L'application restera capable d'expliquer clairement une indisponibilité d'Ollama.

## 4. Lancer et vérifier

```powershell
uv run streamlit run app.py
uv run ruff check .
uv run pytest
```
