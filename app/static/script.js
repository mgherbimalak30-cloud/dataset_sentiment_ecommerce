// Interface : envoie la requête à la VRAIE API et affiche la réponse reçue.
let modeActuel = "text";
const $ = (id) => document.getElementById(id);

const ICONES = { positif: "😊", negatif: "😞", neutre: "😐" };
const LIBELLES = { positif: "Positif", negatif: "Négatif", neutre: "Neutre" };
const ROUTES = { text: "/predict/text", image: "/predict/image", audio: "/predict/audio" };

// --- Changement de mode ---
document.querySelectorAll(".onglet").forEach((bouton) => {
  bouton.addEventListener("click", () => {
    modeActuel = bouton.dataset.mode;
    document.querySelectorAll(".onglet").forEach((b) => b.classList.toggle("actif", b === bouton));
    ["text", "image", "audio"].forEach((m) => $("zone-" + m).classList.toggle("cachee", m !== modeActuel));
    afficherErreur("");
    $("resultat").classList.add("cachee");
  });
});

// --- Sélection des fichiers ---
$("bouton-image").addEventListener("click", () => $("champ-image").click());
$("bouton-audio").addEventListener("click", () => $("champ-audio").click());

$("champ-image").addEventListener("change", (e) => {
  const fichier = e.target.files[0];
  $("nom-image").textContent = fichier ? fichier.name : "Aucun fichier choisi";
  if (fichier) { $("apercu-image").src = URL.createObjectURL(fichier); $("apercu-image").classList.remove("cachee"); }
});
$("champ-audio").addEventListener("change", (e) => {
  const fichier = e.target.files[0];
  $("nom-audio").textContent = fichier ? fichier.name : "Aucun fichier choisi";
  if (fichier) { $("lecteur-audio").src = URL.createObjectURL(fichier); $("lecteur-audio").classList.remove("cachee"); }
});

function afficherErreur(message) {
  $("erreur").textContent = message;
  $("erreur").classList.toggle("cachee", !message);
}

// Transforme la réponse d'erreur FastAPI en message lisible
function messageErreur(donnees, status) {
  if (donnees && typeof donnees.detail === "string") return donnees.detail;
  if (donnees && Array.isArray(donnees.detail)) return donnees.detail.map((d) => d.msg).join(" ; ");
  return "Erreur " + status;
}

function afficherResultat(r) {
  const badge = $("badge-sentiment");
  badge.className = "sentiment " + r.sentiment_predit;
  $("icone-sentiment").textContent = ICONES[r.sentiment_predit] || "";
  $("libelle-sentiment").textContent = LIBELLES[r.sentiment_predit] || r.sentiment_predit;
  const pourcent = (r.confiance * 100).toFixed(1);
  $("valeur-confiance").textContent = pourcent + " %";
  $("barre-confiance").style.width = pourcent + "%";
  $("titre-texte").textContent =
    r.source === "image" ? "Texte extrait (OCR)" : r.source === "audio" ? "Transcription (Whisper)" : "Texte analysé";
  $("texte-extrait").textContent = r.texte_extrait;
  $("resultat").classList.remove("cachee");
}

// --- Analyse ---
$("bouton-analyser").addEventListener("click", async () => {
  afficherErreur("");
  $("resultat").classList.add("cachee");
  let requete;

  if (modeActuel === "text") {
    const texte = $("champ-texte").value.trim();
    if (!texte) return afficherErreur("Veuillez saisir un texte.");
    requete = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: texte }) };
  } else {
    const fichier = $(modeActuel === "image" ? "champ-image" : "champ-audio").files[0];
    if (!fichier) return afficherErreur("Veuillez sélectionner un fichier " + (modeActuel === "image" ? "image." : "audio."));
    const formulaire = new FormData();
    formulaire.append("file", fichier);
    requete = { method: "POST", body: formulaire };
  }

  $("chargement").classList.remove("cachee");
  $("bouton-analyser").disabled = true;
  try {
    const reponse = await fetch(ROUTES[modeActuel], requete);
    const donnees = await reponse.json().catch(() => null);
    if (!reponse.ok) return afficherErreur(messageErreur(donnees, reponse.status));
    afficherResultat(donnees);
  } catch (e) {
    afficherErreur("Impossible de contacter l'API. Le serveur est-il lancé ?");
  } finally {
    $("chargement").classList.add("cachee");
    $("bouton-analyser").disabled = false;
  }
});
