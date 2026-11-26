"""Erreurs communes aux services OCR / ASR. Chacune correspond à un code HTTP dans main.py."""


class ErreurService(Exception):
    code_http = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class FormatNonSupporte(ErreurService):
    code_http = 415


class FichierTropVolumineux(ErreurService):
    code_http = 413


class ContenuInvalide(ErreurService):
    """Fichier vide, illisible, sans texte, ou qui ne correspond pas à son extension."""

    code_http = 422


class ServiceIndisponible(ErreurService):
    """Tesseract ou Whisper non installé / non utilisable."""

    code_http = 503
