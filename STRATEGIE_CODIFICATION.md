# Stratégie de Codification Automatique CITP-08 (ISCO)

Ce document détaille la méthodologie de traitement de données et l'architecture d'intelligence artificielle utilisées pour automatiser la classification des métiers selon le référentiel international **CITP-08**.

## 1. Vision d'ensemble
L'objectif est de transformer un intitulé de métier saisi librement (ex: "Ingénieur en mécanique") en un code numérique standardisé, tout en filtrant les données non exploitables (bruit).



---

## 2. Phase 1 : Pipeline de Nettoyage et Qualité (Data Cleaning à partir de clean1_numerique.py à clean9_space_letter.py)
Avant d'entraîner l'intelligence artificielle, les données passent par **9 étapes de nettoyage** automatisées. Cette étape est cruciale car elle garantit que le modèle n'apprend que sur des données de qualité.

| Étape | Action Technologique | Type de Bruit Éliminé |
| :--- | :--- | :--- |
| **01** | Filtrage Numérique | Supprime les codes ID ou chiffres isolés saisis par erreur. |
| **02** | Filtrage Caractères | Élimine les lignes contenant uniquement des symboles (`!!!`, `???`). |
| **03** | Nettoyage des Bords | Supprime les symboles parasites au début ou à la fin d'un mot. |
| **04** | Filtrage Mono-lettre | Élimine les saisies trop courtes pour être un métier (ex: "A", "Z"). |
| **05 & 06** | Détection de Patterns | Identifie les suites de lettres sans sens (ex: "ABC", "XYZ"). |
| **07** | Filtrage Abréviations | Détecte les codes de service ou abréviations internes (ex: "T. DE S"). |
| **08** | **Normalisation** | Passage en minuscules et suppression des accents (Unicode). |
| **09** | Filtrage Espaces | Élimine les suites de lettres séparées par des espaces (ex: "s d o"). |

---

## 3. Phase 2 : Architecture de l'Intelligence Artificielle avec train.py, fasttext_citp_v1.pt et cc.fr.300.bin

### A. Vectorisation Sémantique (FastText)
Nous utilisons la technologie **FastText** (développée par Meta AI) pour transformer chaque mot en un vecteur mathématique de **300 dimensions**. 
* **Avantage :** Le système comprend que "Médecin" et "Docteur" sont sémantiquement proches, même s'ils s'écrivent différemment.

### B. Le Classifieur (Deep Learning - PyTorch)
Une fois vectorisée, la donnée entre dans un **réseau de neurones multicouches** conçu sur mesure (`CITPClassifier`) :
1. **Input Layer** : 300 neurones.
2. **Hidden Layers** : Deux couches denses (512 et 256 neurones) pour capter la complexité des métiers.
3. **Optimisation** : Utilisation du **Batch Normalization** (pour la stabilité) et du **Dropout** (pour éviter que l'IA n'apprenne par cœur les erreurs).
4. **Output Layer** : Une couche de sortie qui calcule la probabilité pour chaque code du référentiel CITP.



---

## 4. Phase 3 : Résultat (Exécution des prédictions sur fichier Excel avec test_file_resultat.py)
Lorsqu'un utilisateur utilise l'application :
1. Le texte est **normalisé** instantanément (minuscules, sans accents).
2. Le modèle **FastText** génère le vecteur du métier.
3. Le modèle **PyTorch** calcule les probabilités.
4. Le fichier excel s'ouvre et affiche le code ayant le **score de confiance en porcentage**.

---

## 5. Conclusion technique
Cette stratégie permet une codification :
* **Précise** : Grâce au nettoyage granulaire préalable.
* **Intelligente** : Capable de gérer les synonymes et les fautes d'orthographe.
* **Performante** : Temps de réponse inférieur à 100ms.