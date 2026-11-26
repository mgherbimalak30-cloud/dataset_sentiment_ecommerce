"""Tests de l'API. OCR et ASR sont remplacés par des mocks UNIQUEMENT ici.

Le vrai modèle de sentiment (models/sentiment_model.pkl) est utilisé : lancez d'abord
`python scripts/train_model.py`. Aucun accès Internet ni Whisper/Tesseract n'est nécessaire.
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services import asr, ocr
from app.services.sentiment import SENTIMENTS_VALIDES

client = TestClient(app)

TEXTE_POSITIF = "Produit excellent, livraison très rapide, je recommande vivement !"
TEXTE_NEGATIF = "Article reçu cassé et service client injoignable, très déçu."


# ---------- Fabriques de fichiers ----------
def fabriquer_png() -> bytes:
    tampon = io.BytesIO()
    Image.new("RGB", (60, 30), "white").save(tampon, format="PNG")
    return tampon.getvalue()


def fabriquer_audio() -> bytes:
    return b"ID3" + b"\x00" * 200  # signature MP3 : suffisant pour passer la validation du format


class FauxSegment:
    def __init__(self, text):
        self.text = text


class FauxWhisper:
    def __init__(self, texte):
        self.texte = texte

    def transcribe(self, *_args, **_kwargs):
        segments = [FauxSegment(self.texte)] if self.texte else []
        return segments, None


@pytest.fixture
def ocr_mock(monkeypatch):
    """Simule Tesseract : retourne un texte fixe, modifiable par le test."""
    etat = {"texte": TEXTE_POSITIF}
    monkeypatch.setattr(ocr, "configurer_tesseract", lambda: None)
    monkeypatch.setattr(ocr, "choisir_langue", lambda: "fra")
    monkeypatch.setattr(ocr.pytesseract, "image_to_string", lambda *_a, **_k: etat["texte"])
    return etat


@pytest.fixture
def asr_mock(monkeypatch):
    etat = {"texte": TEXTE_NEGATIF}
    monkeypatch.setattr(asr, "charger_whisper", lambda: FauxWhisper(etat["texte"]))
    return etat


def verifier_structure(reponse_json: dict, source: str):
    assert set(reponse_json) == {"source", "texte_extrait", "sentiment_predit", "confiance"}
    assert reponse_json["source"] == source
    assert reponse_json["sentiment_predit"] in SENTIMENTS_VALIDES
    assert 0 <= reponse_json["confiance"] <= 1


# ---------- Santé ----------
def test_accueil_est_l_interface():
    reponse = client.get("/")
    assert reponse.status_code == 200
    assert "E-commerce Customer Voice" in reponse.text


def test_route_api_verifie_application():
    assert client.get("/api").status_code == 200


def test_health_modele_disponible():
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json()["modele_charge"] is True
    assert reponse.json()["status"] == "ok"


# ---------- Texte ----------
def test_predict_text_positif():
    reponse = client.post("/predict/text", json={"text": TEXTE_POSITIF})
    assert reponse.status_code == 200
    donnees = reponse.json()
    verifier_structure(donnees, "text")
    assert donnees["sentiment_predit"] == "positif"
    assert donnees["texte_extrait"] == TEXTE_POSITIF


def test_predict_text_negatif():
    reponse = client.post("/predict/text", json={"text": TEXTE_NEGATIF})
    assert reponse.status_code == 200
    assert reponse.json()["sentiment_predit"] == "negatif"


@pytest.mark.parametrize("corps", [{"text": ""}, {"text": "   "}, {}, {"texte": "mauvais champ"}, {"text": 123}])
def test_predict_text_requete_invalide(corps):
    assert client.post("/predict/text", json=corps).status_code == 422


# ---------- Image ----------
def test_predict_image(ocr_mock):
    reponse = client.post("/predict/image", files={"file": ("avis.png", fabriquer_png(), "image/png")})
    assert reponse.status_code == 200
    donnees = reponse.json()
    verifier_structure(donnees, "image")
    assert donnees["texte_extrait"] == TEXTE_POSITIF
    assert donnees["sentiment_predit"] == "positif"


def test_image_sans_texte_retourne_422(ocr_mock):
    ocr_mock["texte"] = "  \n "
    reponse = client.post("/predict/image", files={"file": ("vide.png", fabriquer_png(), "image/png")})
    assert reponse.status_code == 422
    assert "texte" in reponse.json()["detail"].lower()


def test_image_illisible_retourne_422(ocr_mock):
    reponse = client.post("/predict/image", files={"file": ("faux.png", b"ceci n'est pas une image", "image/png")})
    assert reponse.status_code == 422


def test_image_vide_retourne_422(ocr_mock):
    assert client.post("/predict/image", files={"file": ("a.png", b"", "image/png")}).status_code == 422


def test_image_mime_non_supporte(ocr_mock):
    reponse = client.post("/predict/image", files={"file": ("a.txt", b"bonjour", "text/plain")})
    assert reponse.status_code == 415


def test_image_extension_non_supportee(ocr_mock):
    reponse = client.post("/predict/image", files={"file": ("a.gif", fabriquer_png(), "image/png")})
    assert reponse.status_code == 415


def test_image_trop_volumineuse(ocr_mock):
    gros = fabriquer_png() + b"\x00" * (ocr.TAILLE_MAX_IMAGE + 10)
    reponse = client.post("/predict/image", files={"file": ("gros.png", gros, "image/png")})
    assert reponse.status_code == 413


def test_ocr_indisponible_retourne_503(monkeypatch):
    def tesseract_absent():
        raise ocr.ServiceIndisponible("Tesseract OCR est introuvable.")

    monkeypatch.setattr(ocr, "configurer_tesseract", lambda: None)
    monkeypatch.setattr(ocr, "choisir_langue", tesseract_absent)
    reponse = client.post("/predict/image", files={"file": ("a.png", fabriquer_png(), "image/png")})
    assert reponse.status_code == 503


# ---------- Audio ----------
def test_predict_audio(asr_mock):
    reponse = client.post("/predict/audio", files={"file": ("avis.mp3", fabriquer_audio(), "audio/mpeg")})
    assert reponse.status_code == 200
    donnees = reponse.json()
    verifier_structure(donnees, "audio")
    assert donnees["texte_extrait"] == TEXTE_NEGATIF
    assert donnees["sentiment_predit"] == "negatif"


def test_audio_sans_parole_retourne_422(asr_mock):
    asr_mock["texte"] = ""
    reponse = client.post("/predict/audio", files={"file": ("silence.mp3", fabriquer_audio(), "audio/mpeg")})
    assert reponse.status_code == 422


def test_audio_vide_retourne_422(asr_mock):
    assert client.post("/predict/audio", files={"file": ("a.mp3", b"", "audio/mpeg")}).status_code == 422


def test_audio_faux_contenu_retourne_422(asr_mock):
    reponse = client.post("/predict/audio", files={"file": ("a.mp3", b"pas de l'audio du tout", "audio/mpeg")})
    assert reponse.status_code == 422


def test_audio_mime_non_supporte(asr_mock):
    reponse = client.post("/predict/audio", files={"file": ("a.txt", b"bonjour", "text/plain")})
    assert reponse.status_code == 415


def test_audio_extension_non_supportee(asr_mock):
    reponse = client.post("/predict/audio", files={"file": ("a.xyz", fabriquer_audio(), "audio/mpeg")})
    assert reponse.status_code == 415


def test_audio_trop_volumineux(asr_mock):
    gros = fabriquer_audio() + b"\x00" * (asr.TAILLE_MAX_AUDIO + 10)
    reponse = client.post("/predict/audio", files={"file": ("gros.mp3", gros, "audio/mpeg")})
    assert reponse.status_code == 413


def test_asr_indisponible_retourne_503(monkeypatch):
    def whisper_absent():
        raise asr.ServiceIndisponible("Modèle Whisper indisponible.")

    monkeypatch.setattr(asr, "charger_whisper", whisper_absent)
    reponse = client.post("/predict/audio", files={"file": ("a.mp3", fabriquer_audio(), "audio/mpeg")})
    assert reponse.status_code == 503


def test_audio_non_decodable_retourne_422(monkeypatch):
    class WhisperCasse:
        def transcribe(self, *_a, **_k):
            raise RuntimeError("décodage impossible")

    monkeypatch.setattr(asr, "charger_whisper", lambda: WhisperCasse())
    reponse = client.post("/predict/audio", files={"file": ("a.mp3", fabriquer_audio(), "audio/mpeg")})
    assert reponse.status_code == 422


# ---------- Détection automatique /predict ----------
def test_predict_auto_texte_json():
    reponse = client.post("/predict", json={"text": TEXTE_POSITIF})
    assert reponse.status_code == 200
    verifier_structure(reponse.json(), "text")


def test_predict_auto_image(ocr_mock):
    reponse = client.post("/predict", files={"file": ("avis.png", fabriquer_png(), "image/png")})
    assert reponse.status_code == 200
    verifier_structure(reponse.json(), "image")


def test_predict_auto_audio(asr_mock):
    reponse = client.post("/predict", files={"file": ("avis.mp3", fabriquer_audio(), "audio/mpeg")})
    assert reponse.status_code == 200
    verifier_structure(reponse.json(), "audio")


def test_predict_auto_json_invalide():
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", content=b"{pas du json", headers={"content-type": "application/json"}).status_code == 422


def test_predict_auto_requete_vide():
    reponse = client.post("/predict", data={"autre": "x"}, files={"inutile": ("a.txt", b"x", "text/plain")})
    assert reponse.status_code == 422


def test_predict_auto_type_ambigu(ocr_mock):
    reponse = client.post(
        "/predict", data={"text": "bonjour"}, files={"file": ("avis.png", fabriquer_png(), "image/png")}
    )
    assert reponse.status_code == 422


def test_predict_auto_contenu_different_du_type_annonce(ocr_mock, asr_mock):
    # une image envoyée en prétendant que c'est de l'audio
    reponse = client.post("/predict", files={"file": ("avis.mp3", fabriquer_png(), "audio/mpeg")})
    assert reponse.status_code == 422


def test_predict_auto_format_inconnu():
    reponse = client.post("/predict", files={"file": ("a.txt", b"juste du texte", "text/plain")})
    assert reponse.status_code == 415


def test_predict_auto_content_type_non_supporte():
    reponse = client.post("/predict", content=b"abc", headers={"content-type": "text/plain"})
    assert reponse.status_code == 415


# ---------- Même classifieur pour les trois modalités ----------
def test_meme_classifieur_pour_les_trois_modalites(ocr_mock, asr_mock):
    ocr_mock["texte"] = TEXTE_POSITIF
    asr_mock["texte"] = TEXTE_POSITIF
    texte = client.post("/predict/text", json={"text": TEXTE_POSITIF}).json()
    image = client.post("/predict/image", files={"file": ("a.png", fabriquer_png(), "image/png")}).json()
    audio = client.post("/predict/audio", files={"file": ("a.mp3", fabriquer_audio(), "audio/mpeg")}).json()
    assert texte["sentiment_predit"] == image["sentiment_predit"] == audio["sentiment_predit"]
    assert texte["confiance"] == image["confiance"] == audio["confiance"]
