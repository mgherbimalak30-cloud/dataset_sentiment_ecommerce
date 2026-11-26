"""Compare quelques réglages du modèle avec une validation croisée STRATIFIÉE répétée sur les 24 textes.
Les textes de référence images/audios restent un test externe (jamais utilisés pour choisir).
Lancer : python scripts/compare_models.py
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from app.services.sentiment import STOP_WORDS_FR, nettoyer_texte

df = pd.read_csv(RACINE / "data/textes.csv")
ext = pd.concat(
    [
        pd.read_csv(RACINE / "data/images.csv").rename(columns={"texte_attendu_ocr": "texte"}),
        pd.read_csv(RACINE / "data/audios.csv").rename(columns={"transcription_reference": "texte"}),
    ]
)


def pipeline(ngram, stop, C):
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(preprocessor=nettoyer_texte, stop_words=STOP_WORDS_FR if stop else None, ngram_range=ngram)),
            ("clf", LogisticRegression(max_iter=1000, C=C)),
        ]
    )


cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=42)
print("ngram | stopwords | C | CV accuracy (24 textes, 4 plis x 10) | accuracy externe (18 textes img+audio)")
for ngram in [(1, 1), (1, 2)]:
    for stop in [True, False]:
        for C in [1, 10, 100]:
            cv_score = cross_val_score(pipeline(ngram, stop, C), df["texte"], df["sentiment_ref"], cv=cv).mean()
            m = pipeline(ngram, stop, C).fit(df["texte"], df["sentiment_ref"])
            ext_score = (m.predict(ext["texte"]) == ext["sentiment_ref"]).mean()
            print(f"{ngram} | {stop} | {C} | {cv_score:.3f} | {ext_score:.3f}")
