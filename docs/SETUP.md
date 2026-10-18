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

Installez Ollama et démarrez son service. Le modèle utilisé par défaut est `qwen3.5:4b`, déjà
présent sur la machine de développement. Vérifiez les modèles locaux avec :

```powershell
ollama list
```

Pour utiliser un autre modèle installé sans modifier le code, définissez la variable avant de
lancer l'application :

```powershell
$env:SMARTCAMPUS_OLLAMA_MODEL = "Qwen3:8b"
uv run streamlit run app.py
```

La variable ne vaut que pour la session PowerShell courante. Sans cette variable, l'application
utilise `qwen3.5:4b`. Elle reste capable de signaler clairement une indisponibilité d'Ollama.

## 4. Lancer et vérifier

```powershell
uv run streamlit run app.py
uv run ruff check .
uv run pytest
```
