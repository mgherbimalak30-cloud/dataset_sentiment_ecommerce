"""Test réel (sans mock) des 4 routes avec les fichiers du dataset. Lancer : python scripts/smoke_test.py"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
print("health :", client.get("/health").json())

r = client.post("/predict/text", json={"text": "Le produit est excellent"})
print("/predict/text :", r.status_code, r.json())
r = client.post("/predict", json={"text": "Colis cassé, très déçu"})
print("/predict (json) :", r.status_code, r.json())

audio = (RACINE / "data/audios/A01.mp3").read_bytes()
r = client.post("/predict/audio", files={"file": ("A01.mp3", audio, "audio/mpeg")})
print("/predict/audio :", r.status_code, r.json())
r = client.post("/predict", files={"file": ("A02.mp3", (RACINE / "data/audios/A02.mp3").read_bytes(), "audio/mpeg")})
print("/predict (audio) :", r.status_code, r.json())

image = (RACINE / "data/images/I01.png").read_bytes()
r = client.post("/predict/image", files={"file": ("I01.png", image, "image/png")})
print("/predict/image :", r.status_code, r.json())
r = client.post("/predict", files={"file": ("I02.png", (RACINE / "data/images/I02.png").read_bytes(), "image/png")})
print("/predict (image) :", r.status_code, r.json())
