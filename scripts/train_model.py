"""Entraînement du classifieur de sentiment (TF-IDF + régression logistique).

Lancer depuis la racine du projet :  python scripts/train_model.py
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from app.services.sentiment import (
    CHEMIN_MODELE,
    SENTIMENTS_VALIDES,
    STOP_WORDS_FR,
    nettoyer_texte,
)

sys.stdout.reconfigure(encoding="utf-8")  # accents corrects dans PowerShell
DOSSIER_DATA = RACINE / "data"
GRAINE = 42


def charger_csv(nom_fichier: str) -> pd.DataFrame:
    """Lit un CSV (le vrai format a été vérifié : CSV UTF-8 classique)."""
    return pd.read_csv(DOSSIER_DATA / nom_fichier, encoding="utf-8")


def preparer_donnees(df: pd.DataFrame, col_texte: str, col_sentiment: str) -> pd.DataFrame:
    """Supprime les lignes invalides et les doublons, normalise les étiquettes."""
    df = df[[col_texte, col_sentiment]].copy()
    df.columns = ["texte", "sentiment"]
    df = df.dropna()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["texte"] = df["texte"].astype(str).str.strip()
    df = df[(df["texte"] != "") & df["sentiment"].isin(SENTIMENTS_VALIDES)]
    return df.drop_duplicates(subset="texte").reset_index(drop=True)


def construire_pipeline() -> Pipeline:
    """Un seul objet : nettoyage + TF-IDF + régression logistique."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=nettoyer_texte,  # minuscules, sans accents, sans ponctuation
                    stop_words=STOP_WORDS_FR,  # stop-words français SANS les négations
                    ngram_range=(1, 2),  # mots seuls et paires de mots ("pas bien")
                ),
            ),
            ("classifieur", LogisticRegression(max_iter=1000, C=10, random_state=GRAINE)),
        ]
    )


def evaluer(modele, textes, vrais, titre: str):
    predits = modele.predict(textes)
    print(f"\n=== {titre} ({len(textes)} exemples) ===")
    print("Précision (accuracy) :", round(accuracy_score(vrais, predits), 4))
    print("F1-score macro       :", round(f1_score(vrais, predits, average="macro"), 4))
    print("Matrice de confusion (lignes = vrai, colonnes = prédit) :")
    print("Ordre des classes    :", list(modele.classes_))
    print(confusion_matrix(vrais, predits, labels=modele.classes_))
    print(classification_report(vrais, predits, labels=modele.classes_, zero_division=0))


def main():
    df_brut = charger_csv("textes.csv")
    print("Colonnes de textes.csv :", list(df_brut.columns))
    print("Lignes brutes :", len(df_brut))
    df = preparer_donnees(df_brut, "texte", "sentiment_ref")
    print("Lignes valides après nettoyage :", len(df))
    print("Répartition des classes :\n", df["sentiment"].value_counts().to_string())

    # Séparation stratifiée : chaque classe est présente dans train et test.
    x_train, x_test, y_train, y_test = train_test_split(
        df["texte"], df["sentiment"], test_size=0.25, stratify=df["sentiment"], random_state=GRAINE
    )
    print(f"Entraînement : {len(x_train)} | Test : {len(x_test)}")

    modele = construire_pipeline()
    modele.fit(x_train, y_train)
    evaluer(modele, x_test, y_test, "Jeu de test (textes)")

    # Le jeu de test est minuscule : la validation croisée donne une estimation plus stable.
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=GRAINE)
    scores = cross_val_score(construire_pipeline(), df["texte"], df["sentiment"], cv=cv, scoring="accuracy")
    print("\nValidation croisée 4 plis (accuracy) :", scores.round(3), "moyenne =", round(scores.mean(), 4))

    # Test supplémentaire "cascade" : textes de référence des images et des audios
    # (jamais utilisés pour l'entraînement). Cela simule un OCR/ASR parfait.
    for fichier, col in [("images.csv", "texte_attendu_ocr"), ("audios.csv", "transcription_reference")]:
        extra = preparer_donnees(charger_csv(fichier), col, "sentiment_ref")
        evaluer(modele, extra["texte"], extra["sentiment"], f"Textes de référence de {fichier}")

    # Le dataset est très petit : le modèle FINAL est ré-entraîné sur les 24 textes
    # (les chiffres ci-dessus restent ceux du modèle entraîné sur 18 textes seulement).
    modele_final = construire_pipeline()
    modele_final.fit(df["texte"], df["sentiment"])
    print("\nModèle final ré-entraîné sur tous les textes :", len(df))
    for fichier, col in [("images.csv", "texte_attendu_ocr"), ("audios.csv", "transcription_reference")]:
        extra = preparer_donnees(charger_csv(fichier), col, "sentiment_ref")
        precision = accuracy_score(extra["sentiment"], modele_final.predict(extra["texte"]))
        print(f"Modèle final - accuracy sur textes de référence de {fichier} : {precision:.4f}")

    CHEMIN_MODELE.parent.mkdir(exist_ok=True)
    joblib.dump(modele_final, CHEMIN_MODELE)
    modele = modele_final
    print("\nModèle sauvegardé :", CHEMIN_MODELE)


if __name__ == "__main__":
    main()
