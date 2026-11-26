"""Schémas Pydantic : forme des requêtes et des réponses de l'API."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TexteEntree(BaseModel):
    text: str = Field(..., description="Texte de l'avis client", examples=["Le produit est excellent"])

    @field_validator("text")
    @classmethod
    def texte_non_vide(cls, valeur: str) -> str:
        valeur = valeur.strip()
        if not valeur:
            raise ValueError("Le texte ne doit pas être vide.")
        if len(valeur) > 5000:
            raise ValueError("Le texte est trop long (maximum 5000 caractères).")
        return valeur


class PredictionReponse(BaseModel):
    source: Literal["text", "image", "audio"]
    texte_extrait: str
    sentiment_predit: Literal["positif", "negatif", "neutre"]
    confiance: float = Field(..., ge=0, le=1)


class SanteReponse(BaseModel):
    status: str
    modele_charge: bool
    ocr_disponible: bool
    asr_disponible: bool
