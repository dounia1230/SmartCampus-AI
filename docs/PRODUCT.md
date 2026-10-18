# Produit

## Objectif

SmartCampus AI aide un étudiant francophone à trouver une procédure fictive liée aux inscriptions,
aux attestations ou aux paiements. Le projet démontre clairement la combinaison de classification
supervisée, embeddings, RAG, règles métier et génération locale.

## Utilisateur principal

Un étudiant qui formule une question courte dans le chat et veut une réponse compréhensible,
accompagnée des identifiants des procédures utilisées.

## Périmètre du MVP

- trois catégories et six procédures fictives ;
- authentification simulée ;
- réponses en français produites localement avec Ollama ;
- abstention en cas d'incertitude ou de contexte insuffisant ;
- validation humaine obligatoire pour les remboursements ;
- mesures reproductibles du classifieur et tests des règles métier.

## Hors périmètre

- connexion à un système universitaire réel ;
- stockage de données personnelles ou historique persistant ;
- réalisation d'un paiement, remboursement ou dépôt de dossier ;
- garantie juridique ou administrative ;
- authentification réelle et contrôle d'accès multi-utilisateur.

## Critères de réussite pédagogique

- chaque composant peut être expliqué et testé séparément ;
- une réponse générée ne peut citer qu'une procédure récupérée ;
- les décisions sensibles ne dépendent pas du LLM ;
- l'évaluation présente aussi ses limites et ses cas de refus.
