"""OCR : image -> texte (Pillow + Pytesseract), puis sentiment avec le MÊME classifieur.

Tesseract est un programme Windows à installer séparément (voir README).
On peut indiquer son chemin avec la variable d'environnement TESSERACT_CMD.
"""
import io
import os
from pathlib import Path

import pytesseract
from PIL import Image, UnidentifiedImageError

from app.services.erreurs import ContenuInvalide, FichierTropVolumineux, FormatNonSupporte, ServiceIndisponible
from app.services.sentiment import predire_sentiment

EXTENSIONS_IMAGE = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
FORMATS_PILLOW = {"PNG", "JPEG", "BMP", "WEBP"}
TAILLE_MAX_IMAGE = 5 * 1024 * 1024  # 5 Mo

CHEMINS_TESSERACT_WINDOWS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
]


def configurer_tesseract():
    """Trouve tesseract.exe (variable d'environnement, sinon chemins habituels, sinon PATH)."""
    chemin = os.environ.get("TESSERACT_CMD")
    if chemin:
        pytesseract.pytesseract.tesseract_cmd = chemin
        return
    for candidat in CHEMINS_TESSERACT_WINDOWS:
        if Path(candidat).exists():
            pytesseract.pytesseract.tesseract_cmd = candidat
            return


def choisir_langue() -> str:
    """Français si le pack de langue 'fra' est installé, sinon anglais (moins précis pour les accents)."""
    try:
        langues = pytesseract.get_languages(config="")
    except Exception as erreur:
        raise ServiceIndisponible(
            "Tesseract OCR est introuvable. Installez-le et configurez TESSERACT_CMD (voir README)."
        ) from erreur
    return "fra" if "fra" in langues else "eng"


def ouvrir_image(contenu: bytes, nom_fichier: str) -> Image.Image:
    """Vérifie taille, extension et vrai format (Pillow), puis retourne l'image."""
    if len(contenu) == 0:
        raise ContenuInvalide("Le fichier image est vide.")
    if len(contenu) > TAILLE_MAX_IMAGE:
        raise FichierTropVolumineux(f"Image trop volumineuse (maximum {TAILLE_MAX_IMAGE // (1024 * 1024)} Mo).")
    if Path(nom_fichier).suffix.lower() not in EXTENSIONS_IMAGE:
        raise FormatNonSupporte("Extension d'image non prise en charge (png, jpg, jpeg, bmp, webp).")
    try:
        image = Image.open(io.BytesIO(contenu))
        image.verify()  # vérifie que le fichier n'est pas corrompu
        image = Image.open(io.BytesIO(contenu))  # verify() rend l'image inutilisable : on la rouvre
        format_reel = image.format
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as erreur:
        raise ContenuInvalide("Image illisible ou contenu qui n'est pas une image.") from erreur
    if format_reel not in FORMATS_PILLOW:
        raise FormatNonSupporte(f"Format d'image non pris en charge : {format_reel}.")
    return image.convert("RGB")


def extraire_texte_image(contenu: bytes, nom_fichier: str) -> str:
    """Retourne le texte lu dans l'image (erreur si aucun texte)."""
    image = ouvrir_image(contenu, nom_fichier)
    configurer_tesseract()
    langue = choisir_langue()
    try:
        texte = pytesseract.image_to_string(image, lang=langue)
    except pytesseract.TesseractNotFoundError as erreur:
        raise ServiceIndisponible("Tesseract OCR est introuvable (voir README).") from erreur
    except pytesseract.TesseractError as erreur:
        raise ServiceIndisponible(f"Erreur Tesseract : {erreur}") from erreur
    texte = " ".join(texte.split())
    if not texte:
        raise ContenuInvalide("Aucun texte détecté dans l'image.")
    return texte


def analyser_image(contenu: bytes, nom_fichier: str) -> dict:
    texte = extraire_texte_image(contenu, nom_fichier)
    return {"source": "image", "texte_extrait": texte, **predire_sentiment(texte)}
