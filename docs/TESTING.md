# Tests et évaluation

## Vérifications rapides

```powershell
uv run ruff check .
uv run pytest
```

Les tests unitaires remplacent le modèle d'embeddings et Ollama par de petits doubles contrôlés.
Ils restent donc reproductibles, rapides et utilisables hors ligne. Ils vérifient notamment que
les décisions de refus interviennent avant les appels génératifs.

## Évaluation du classifieur

Après l'entraînement :

```powershell
uv run python -m evaluation.evaluate
```

Cette commande utilise 15 formulations inédites et équilibrées, calcule l'exactitude, le F1
macro et le rapport par classe, puis écrit dans `evaluation/artifacts/` :

- `classifier_metrics.json` ;
- `confusion_matrix.png`.

Six questions hors domaine servent à comparer les seuils de confiance. Le diagnostic affiche la
couverture des questions pertinentes, l'exactitude des réponses acceptées et le taux de rejet hors
domaine. Ce petit échantillon sert à illustrer la méthode ; il n'est pas assez grand pour certifier
un déploiement réel.

### Baseline du 18 octobre 2026

Avec 60 exemples d'entraînement équilibrés et 15 exemples de test distincts :

- exactitude : `1.000` ;
- F1 macro : `1.000` ;
- confiance moyenne : `0.467` ;
- au seuil `0.42` : couverture pertinente `66.7 %`, rejet hors domaine `66.7 %`.

Ce score parfait porte sur un jeu minuscule et proche du domaine d'entraînement. Il démontre que
les trois intentions sont séparables dans ce prototype, pas que le modèle généralise à une vraie
population étudiante. La prochaine évaluation devrait employer un jeu annoté par une autre personne,
plus de paraphrases, des fautes de frappe et davantage de demandes ambiguës.

Un test réel du modèle `paraphrase-multilingual-MiniLM-L12-v2` a également classé la question
« Comment obtenir une facture ? » vers `PAY01` avec une similarité cosinus de `0.556`.

## Scénarios métier couverts

- remboursement envoyé vers validation humaine avant tout appel IA ;
- document personnel refusé sans authentification ;
- faible confiance transformée en demande de précision ;
- absence de procédure transformée en abstention ;
- documents proches mais insuffisants transformés en `information_insuffisante` par `answerable` ;
- panne Ollama signalée comme indisponibilité ;
- citation absente ou inventée rejetée ;
- pipeline nominal avec catégorie, confiance, documents et sources.
