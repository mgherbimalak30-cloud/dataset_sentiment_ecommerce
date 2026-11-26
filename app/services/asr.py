"""ASR : audio -> texte (Whisper via faster-whisper), puis sentiment avec le MÊME classifieur.

faster-whisper est une version légère de Whisper (CPU). Elle lit les MP3/WAV elle-même
(bibliothèque PyAV), donc FFmpeg n'est pas obligatoire.
Au premier lancement, le modèle est téléchargé depuis Internet (~150 Mo pour 'base').
"""
import io
import os
from pathlib import Path

from app.services.erreurs import ContenuInvalide, FichierTropVolumineux, FormatNonSupporte, ServiceIndisponible
from app.services.sentiment import predire_sentiment

EXTENSIONS_AUDIO = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
TAILLE_MAX_AUDIO = 10 * 1024 * 1024  # 10 Mo
LANGUE_AUDIO = "fr"  # les audios du dataset sont en français
NOM_MODELE_WHISPER = os.environ.get("WHISPER_MODEL", "base")

_modele_whisper = None


def format_audio_reel(contenu: bytes):
    """Reconnaît le vrai format grâce aux premiers octets (on ne se fie pas à l'extension)."""
    if contenu[:3] == b"ID3" or (len(contenu) > 1 and contenu[0] == 0xFF and (contenu[1] & 0xE0) == 0xE0):
        return "mp3"
    if contenu[:4] == b"RIFF" and contenu[8:12] == b"WAVE":
        return "wav"
    if contenu[:4] == b"OggS":
        return "ogg"
    if contenu[:4] == b"fLaC":
        return "flac"
    if contenu[4:8] == b"ftyp":
        return "m4a"
    return None


def verifier_audio(contenu: bytes, nom_fichier: str):
    if len(contenu) == 0:
        raise ContenuInvalide("Le fichier audio est vide.")
    if len(contenu) > TAILLE_MAX_AUDIO:
        raise FichierTropVolumineux(f"Audio trop volumineux (maximum {TAILLE_MAX_AUDIO // (1024 * 1024)} Mo).")
    if Path(nom_fichier).suffix.lower() not in EXTENSIONS_AUDIO:
        raise FormatNonSupporte("Extension audio non prise en charge (mp3, wav, ogg, flac, m4a).")
    if format_audio_reel(contenu) is None:
        raise ContenuInvalide("Le contenu n'est pas un fichier audio reconnu.")


def charger_whisper():
    global _modele_whisper
    if _modele_whisper is None:
        try:
            from faster_whisper import WhisperModel

            _modele_whisper = WhisperModel(NOM_MODELE_WHISPER, device="cpu", compute_type="int8")
        except Exception as erreur:
            raise ServiceIndisponible(
                f"Impossible de charger le modèle Whisper '{NOM_MODELE_WHISPER}' "
                "(connexion Internet nécessaire au premier lancement)."
            ) from erreur
    return _modele_whisper


def transcrire_audio(contenu: bytes, nom_fichier: str) -> str:
    """Retourne la transcription (erreur si audio illisible ou sans parole)."""
    verifier_audio(contenu, nom_fichier)
    modele = charger_whisper()
    try:
        segments, _info = modele.transcribe(io.BytesIO(contenu), language=LANGUE_AUDIO, vad_filter=True)
        texte = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as erreur:
        raise ContenuInvalide("Audio illisible : impossible de le décoder.") from erreur
    if not texte:
        raise ContenuInvalide("Aucune parole détectée dans l'audio.")
    return texte


def analyser_audio(contenu: bytes, nom_fichier: str) -> dict:
    texte = transcrire_audio(contenu, nom_fichier)
    return {"source": "audio", "texte_extrait": texte, **predire_sentiment(texte)}
