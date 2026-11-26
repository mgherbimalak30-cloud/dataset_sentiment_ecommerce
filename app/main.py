"""API FastAPI : texte, image (OCR) et audio (ASR) -> même classifieur de sentiment."""
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.schemas import PredictionReponse, SanteReponse, TexteEntree
from app.services import asr, ocr
from app.services.erreurs import ContenuInvalide, ErreurService, FormatNonSupporte
from app.services.sentiment import ModeleIndisponible, modele_disponible, predire_sentiment

DOSSIER_STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="E-commerce Customer Voice",
    description="Classifieur de sentiment multi-source (texte, image, audio).",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=DOSSIER_STATIC), name="static")


# ---------- Gestion des erreurs : jamais de fausse prédiction ----------
@app.exception_handler(ErreurService)
async def gerer_erreur_service(_request: Request, erreur: ErreurService):
    return JSONResponse(status_code=erreur.code_http, content={"detail": erreur.message})


@app.exception_handler(ModeleIndisponible)
async def gerer_modele_absent(_request: Request, erreur: ModeleIndisponible):
    return JSONResponse(status_code=503, content={"detail": str(erreur)})


# ---------- Outils ----------
def signature_image(contenu: bytes) -> bool:
    return (
        contenu[:8] == b"\x89PNG\r\n\x1a\n"
        or contenu[:3] == b"\xff\xd8\xff"
        or contenu[:2] == b"BM"
        or (contenu[:4] == b"RIFF" and contenu[8:12] == b"WEBP")
    )


async def lire_fichier(fichier: UploadFile, taille_max: int) -> bytes:
    """Lit le fichier mais s'arrête dès qu'il dépasse la taille maximale."""
    contenu = await fichier.read(taille_max + 1)
    return contenu


def predire_texte(texte: str) -> dict:
    return {"source": "text", "texte_extrait": texte, **predire_sentiment(texte)}


# ---------- Routes ----------
@app.get("/", include_in_schema=False)
def accueil():
    """Page d'accueil : l'interface web."""
    return FileResponse(DOSSIER_STATIC / "index.html")


@app.get("/api")
def verifier_application():
    return {"message": "E-commerce Customer Voice : l'API fonctionne.", "docs": "/docs"}


@app.get("/health", response_model=SanteReponse)
def sante():
    from app.services.ocr import configurer_tesseract, choisir_langue

    configurer_tesseract()
    try:
        choisir_langue()
        ocr_ok = True
    except Exception:
        ocr_ok = False
    try:
        import faster_whisper  # noqa: F401

        asr_ok = True
    except ImportError:
        asr_ok = False
    ok = modele_disponible()
    return SanteReponse(
        status="ok" if ok else "degrade", modele_charge=ok, ocr_disponible=ocr_ok, asr_disponible=asr_ok
    )


@app.post("/predict/text", response_model=PredictionReponse)
def predict_text(entree: TexteEntree):
    return predire_texte(entree.text)


@app.post("/predict/image", response_model=PredictionReponse)
async def predict_image(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise FormatNonSupporte("Le type MIME doit être image/* (png, jpeg, bmp, webp).")
    contenu = await lire_fichier(file, ocr.TAILLE_MAX_IMAGE)
    return ocr.analyser_image(contenu, file.filename or "")


@app.post("/predict/audio", response_model=PredictionReponse)
async def predict_audio(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("audio/"):
        raise FormatNonSupporte("Le type MIME doit être audio/* (mp3, wav, ogg, flac, m4a).")
    contenu = await lire_fichier(file, asr.TAILLE_MAX_AUDIO)
    return asr.analyser_audio(contenu, file.filename or "")


@app.post("/predict", response_model=PredictionReponse)
async def predict_auto(request: Request):
    """Détecte le type : JSON -> texte ; multipart -> image ou audio (selon le contenu réel)."""
    type_contenu = request.headers.get("content-type", "")

    if type_contenu.startswith("application/json"):
        try:
            donnees = await request.json()
            entree = TexteEntree(**donnees)
        except (ValueError, TypeError, ValidationError) as erreur:
            raise RequestValidationError([{"loc": ("body",), "msg": f"JSON invalide : {erreur}", "type": "value_error"}])
        return predire_texte(entree.text)

    if type_contenu.startswith("multipart/form-data"):
        formulaire = await request.form()
        fichier = formulaire.get("file")
        texte = formulaire.get("text")
        if fichier is not None and texte:
            raise ContenuInvalide("Entrée ambiguë : envoyez soit un fichier, soit un texte, pas les deux.")
        if fichier is None and texte:
            return predire_texte(TexteEntree(text=str(texte)).text)
        if fichier is None or not hasattr(fichier, "read"):
            raise ContenuInvalide("Aucune entrée : envoyez un champ 'file' ou un JSON {\"text\": ...}.")

        taille_max = max(ocr.TAILLE_MAX_IMAGE, asr.TAILLE_MAX_AUDIO)
        contenu = await fichier.read(taille_max + 1)
        mime = (fichier.content_type or "").lower()
        nom = fichier.filename or ""

        if signature_image(contenu):
            categorie_reelle = "image"
        elif asr.format_audio_reel(contenu):
            categorie_reelle = "audio"
        elif len(contenu) == 0:
            raise ContenuInvalide("Le fichier envoyé est vide.")
        else:
            raise FormatNonSupporte("Type de fichier non reconnu (image ou audio attendu).")

        categorie_annoncee = mime.split("/")[0]
        if categorie_annoncee not in ("image", "audio"):
            raise FormatNonSupporte(f"Type MIME non pris en charge : {mime or 'absent'}.")
        if categorie_annoncee != categorie_reelle:
            raise ContenuInvalide(
                f"Le contenu est de type {categorie_reelle} mais le type annoncé est {categorie_annoncee}."
            )
        if categorie_reelle == "image":
            return ocr.analyser_image(contenu, nom)
        return asr.analyser_audio(contenu, nom)

    raise FormatNonSupporte("Content-Type non pris en charge : utilisez application/json ou multipart/form-data.")
