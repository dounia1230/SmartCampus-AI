# Architecture

SmartCampus AI est une application locale, organisée comme un pipeline explicable :

1. les règles déterministes traitent les entrées vides, l'authentification et les remboursements ;
2. le classifieur TF-IDF + régression logistique détermine la catégorie et sa confiance ;
3. la recherche sémantique limite le contexte aux procédures de cette catégorie ;
4. Ollama formule une réponse à partir de ce contexte uniquement ;
5. l'orchestrateur vérifie les références avant de les rendre à l'interface Streamlit.

Le code métier vit dans `src/smartcampus_ai`. Les données fictives restent dans `data`, les
modèles générés dans `models`, et les sorties d'évaluation dans `evaluation/artifacts`.

La recherche utilise `encode_query` et `encode_document`, puis un produit scalaire sur des
vecteurs normalisés (équivalent à la similarité cosinus). Le corpus étant volontairement petit,
un index en mémoire est plus clair et suffisant ; une base vectorielle serait prématurée.

## Principes

- Les règles de refus passent avant tout appel coûteux ou génératif.
- Une faible confiance ou un contexte insuffisant produit une abstention explicite.
- L'interface ne contient pas de logique métier.
- Les dépendances lourdes et modèles sont chargés paresseusement.

## États de sortie

L'orchestrateur renvoie un `statut` stable pour que l'interface ne déduise jamais l'état depuis
le texte : `vide`, `entree_invalide`, `validation_humaine`, `authentification_requise`,
`modele_indisponible`, `clarification_requise`, `information_insuffisante`,
`service_indisponible` ou `termine`.

Une réponse générée n'est acceptée que si elle contient au moins une citation issue des documents
retrouvés et aucune citation inconnue. Cette validation limite les erreurs de provenance, mais ne
prouve pas à elle seule que chaque phrase est fidèlement supportée.
