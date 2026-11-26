# Guide de démonstration (≈ 5 minutes)

## Explications simples
- **Problème métier** : un site e-commerce reçoit des avis sous plusieurs formes (messages, captures d'écran, messages vocaux). On veut savoir automatiquement si le client est content, mécontent ou neutre.
- **Pourquoi 3 modalités** : les clients n'écrivent pas toujours du texte. Image et audio doivent être convertis en texte.
- **TF-IDF** : transforme un texte en nombres. Un mot compte beaucoup s'il est fréquent dans le texte mais rare dans les autres (ex. « cassé »), peu s'il est partout (ex. « le »).
- **Régression logistique** : apprend un poids par mot et calcule une probabilité pour chaque classe ; la plus haute est la prédiction, sa probabilité est la « confiance ».
- **OCR (Tesseract)** : lit le texte dans une image.
- **ASR (Whisper)** : transcrit la parole en texte.
- **Même classifieur partout** : le sentiment dépend du texte. Un seul modèle = plus simple, une seule évaluation, résultats cohérents. Un test automatique vérifie : même texte → même résultat par les 3 routes.
- **API** : FastAPI reçoit JSON ou fichier, Pydantic valide, les services OCR/ASR extraient le texte, le modèle prédit. Erreurs claires (422, 413, 415, 503), jamais de fausse prédiction.
- **Tests** : 37 tests Pytest vérifient routes, erreurs et formats. OCR/ASR y sont simulés (mocks) pour fonctionner hors ligne ; la qualité réelle est mesurée à part par `scripts/test_ocr_dataset.py` et `scripts/smoke_test.py`.
- **Limites** : voir plus bas.

## Préparation
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload
```
Ouvrir http://127.0.0.1:8000/ . Faire un premier essai audio avant la démo (le modèle Whisper se télécharge au premier appel, ~150 Mo, puis c'est rapide). Whisper met quelques secondes par audio, l'OCR 1 à 2 secondes.

## Scénario (vérifié sur cette machine)
1. **(0:30) Contexte + architecture** : schéma en cascade du README.
2. **(1:00) Texte** : `Excellente qualité de fabrication, le rapport qualité-prix est imbattable.` → positif 85,2 %. Puis `Article reçu cassé et le service client ne répond pas depuis une semaine. Très déçu.` Précaution : ce sont des textes d'entraînement, donc la confiance est flatteuse. Pour être honnête, saisir aussi une phrase nouvelle et commenter le résultat.
3. **(1:00) Image** : `data/images/I02.png` → négatif (39,6 %) ; `I03.png` → neutre (62,1 %). Montrer le texte extrait (le bandeau « CAPTURE AVIS CLIENT… » de l'image est lu aussi).
4. **(1:00) Audio** : `data/audios/A04.mp3` → positif (77,0 %), transcription fidèle ; `A02.mp3` → négatif (64,5 %).
5. **(0:45) Swagger** : http://127.0.0.1:8000/docs , tester `/predict` ; montrer une erreur (fichier .txt → 415).
6. **(0:30) Tests** : `python -m pytest -v` → 37 passed.
7. **(0:15) Limites** : être transparent (voir ci-dessous).

## Résultats réels mesurés
| Mesure | Résultat |
|---|---|
| Tests Pytest | 37 réussis |
| Images (OCR réel, 9 fichiers) | 7/9 sentiments corrects ; erreurs : I01 (positif→neutre), I09 (neutre→négatif) |
| Audios (Whisper réel, 9 fichiers) | 8/9 ; erreur : A05 (négatif→positif) |
| Jeu de test texte (6 textes) | accuracy 0,50 ; F1 macro 0,43 |
| Validation croisée texte | ≈ 0,29 (sous le hasard de 0,33) |

Détail OCR : `docs/resultats_ocr_images.txt`. Détail tests : `docs/resultats_tests.txt`.

## Limites à annoncer au professeur
- Seulement 24 textes d'entraînement : le modèle est peu fiable et peu sûr de lui (confiances souvent 40-60 %).
- Exemple d'erreur : « Le produit est excellent » est classé neutre (45 %).
- La classe neutre est la plus difficile.
- Étude d'amélioration (`scripts/compare_models.py`) : 12 réglages testés, aucun ne fait mieux de façon fiable (0,26-0,30) ; il faudrait plus de données annotées.
- OCR et ASR ajoutent leurs propres erreurs (accents perdus, « Collier » au lieu de « Colis »).
