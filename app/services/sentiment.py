"""Service de sentiment : UN SEUL classifieur utilisé par texte, image (OCR) et audio (ASR).

Ce fichier contient :
- le nettoyage du texte (utilisé à l'entraînement ET à la prédiction),
- la liste de stop-words français (sans les mots de négation),
- le chargement du modèle sauvegardé et la fonction `predire_sentiment`.
"""
import re
import unicodedata
from pathlib import Path

import joblib

RACINE_PROJET = Path(__file__).resolve().parents[2]
CHEMIN_MODELE = RACINE_PROJET / "models" / "sentiment_model.pkl"

# Mots de négation / restriction : on les GARDE car ils changent le sens d'une phrase.
MOTS_NEGATION = {"pas", "jamais", "aucun", "aucune", "ne", "n", "non", "sans", "rien", "ni", "plus"}

# Stop-words français (écrits sans accents car le nettoyage supprime les accents).
STOP_WORDS_FR = sorted(
    {
        "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "et", "ou", "en", "au", "aux",
        "a", "ce", "cet", "cette", "ces", "ca", "c", "je", "j", "tu", "il", "elle", "on", "nous",
        "vous", "ils", "elles", "me", "m", "te", "se", "s", "mon", "ma", "mes", "ton", "son", "sa",
        "ses", "notre", "votre", "leur", "leurs", "que", "qu", "qui", "quoi", "dont", "dans", "sur",
        "sous", "par", "pour", "avec", "chez", "est", "es", "sont", "suis", "etait", "ete", "etre",
        "ai", "as", "avons", "avez", "ont", "avait", "y", "si", "mais", "donc", "car", "or", "tout",
        "toute", "tous", "toutes", "ainsi", "comme", "lors", "meme", "depuis", "apres", "dans",
    }
    - MOTS_NEGATION
)

SENTIMENTS_VALIDES = ["negatif", "neutre", "positif"]


def nettoyer_texte(texte: str) -> str:
    """Minuscules, suppression des accents, de la ponctuation et des espaces en trop."""
    texte = str(texte).lower()
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    texte = re.sub(r"[^a-z0-9\s]", " ", texte)  # on remplace la ponctuation par un espace
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte


class ModeleIndisponible(Exception):
    """Levée quand le fichier du modèle n'existe pas (il faut lancer l'entraînement)."""


_modele_en_cache = None


def charger_modele():
    """Charge le pipeline (TF-IDF + régression logistique) une seule fois."""
    global _modele_en_cache
    if _modele_en_cache is None:
        if not CHEMIN_MODELE.exists():
            raise ModeleIndisponible(
                "Modèle introuvable. Lancez d'abord : python scripts/train_model.py"
            )
        _modele_en_cache = joblib.load(CHEMIN_MODELE)
    return _modele_en_cache


def modele_disponible() -> bool:
    try:
        charger_modele()
        return True
    except Exception:
        return False


def predire_sentiment(texte: str) -> dict:
    """Retourne {'sentiment_predit': ..., 'confiance': ...} pour un texte."""
    modele = charger_modele()
    probabilites = modele.predict_proba([texte])[0]
    indice = int(probabilites.argmax())
    return {
        "sentiment_predit": str(modele.classes_[indice]),
        "confiance": round(float(probabilites[indice]), 4),
    }
