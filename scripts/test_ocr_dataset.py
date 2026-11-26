"""Test réel de l'OCR sur les 9 images du dataset. Lancer : python scripts/test_ocr_dataset.py"""
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient

from app.main import app
from app.services.sentiment import nettoyer_texte

client = TestClient(app)
df = pd.read_csv(RACINE / "data" / "images.csv", encoding="utf-8")
bons = 0
for _, ligne in df.iterrows():
    contenu = (RACINE / "data" / "images" / ligne["fichier"]).read_bytes()
    r = client.post("/predict/image", files={"file": (ligne["fichier"], contenu, "image/png")})
    if r.status_code != 200:
        print(ligne["fichier"], r.status_code, r.json())
        continue
    d = r.json()
    ok = d["sentiment_predit"] == ligne["sentiment_ref"]
    bons += ok
    mots_attendus = set(nettoyer_texte(ligne["texte_attendu_ocr"]).split())
    mots_lus = set(nettoyer_texte(d["texte_extrait"]).split())
    rappel = len(mots_attendus & mots_lus) / len(mots_attendus)
    print(f"{ligne['fichier']} attendu={ligne['sentiment_ref']} predit={d['sentiment_predit']} "
          f"conf={d['confiance']} {'OK' if ok else 'ERREUR'} | mots OCR corrects={rappel:.0%}")
    print("   OCR     :", d["texte_extrait"])
    print("   attendu :", ligne["texte_attendu_ocr"])
print(f"Sentiments corrects : {bons}/{len(df)}")
