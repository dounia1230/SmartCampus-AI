"""Held-out examples used by the reproducible classifier evaluation."""

X_TEST = [
    "Je veux intégrer cette université",
    "Où remettre mon dossier de candidature ?",
    "Comment confirmer mon admission ?",
    "Quelles démarches faut-il suivre pour devenir étudiant ?",
    "Le statut de ma candidature a-t-il changé ?",
    "Je cherche une preuve officielle de mes études",
    "Comment avoir un document confirmant ma réussite ?",
    "Je souhaite retirer mon justificatif étudiant",
    "Où demander une preuve de scolarité ?",
    "Mon attestation est-elle disponible ?",
    "Je dois régler ma mensualité",
    "Puis-je récupérer l'argent versé ?",
    "Comment recevoir mon reçu financier ?",
    "À quelle date payer les frais ?",
    "Je souhaite connaître mon solde",
]

Y_TEST = [
    "inscriptions",
    "inscriptions",
    "inscriptions",
    "inscriptions",
    "inscriptions",
    "attestations",
    "attestations",
    "attestations",
    "attestations",
    "attestations",
    "paiements",
    "paiements",
    "paiements",
    "paiements",
    "paiements",
]

OUT_OF_DOMAIN = [
    "Quel est le menu du restaurant universitaire ?",
    "À quelle heure passe le prochain bus ?",
    "Qui a gagné le match hier ?",
    "Peux-tu corriger mon exercice de mathématiques ?",
    "Quelle météo fera-t-il demain ?",
    "Où se trouve la bibliothèque municipale ?",
]
