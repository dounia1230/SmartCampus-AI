"""Train and persist the administrative request classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from smartcampus_ai.paths import CLASSIFIER_PATH, DATA_DIR

REQUIRED_COLUMNS = {"question", "categorie"}


def load_training_data(path: Path) -> pd.DataFrame:
    """Load and validate the labeled training examples."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {', '.join(sorted(missing))}")

    frame = frame.loc[:, ["question", "categorie"]].dropna().copy()
    frame["question"] = frame["question"].astype(str).str.strip()
    frame["categorie"] = frame["categorie"].astype(str).str.strip()
    frame = frame[(frame["question"] != "") & (frame["categorie"] != "")]

    if frame["categorie"].nunique() < 2:
        raise ValueError("Le jeu d'entraînement doit contenir au moins deux catégories.")
    if frame.duplicated(subset=["question"]).any():
        raise ValueError("Le jeu d'entraînement contient des questions dupliquées.")
    return frame


def build_pipeline() -> Pipeline:
    """Create a deterministic TF-IDF and logistic-regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, random_state=42),
            ),
        ]
    )


def train_classifier(data_path: Path, output_path: Path) -> Pipeline:
    """Train the pipeline and save it to ``output_path``."""
    frame = load_training_data(data_path)
    pipeline = build_pipeline()
    pipeline.fit(frame["question"], frame["categorie"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîner le classifieur SmartCampus AI.")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "demandes.csv")
    parser.add_argument("--output", type=Path, default=CLASSIFIER_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = train_classifier(args.data, args.output)
    prediction = model.predict(["Comment obtenir une facture ?"])[0]
    confidence = float(model.predict_proba(["Comment obtenir une facture ?"])[0].max())
    print(f"Modèle enregistré dans {args.output}")
    print(f"Test: categorie={prediction}, confiance={confidence:.3f}")


if __name__ == "__main__":
    main()
