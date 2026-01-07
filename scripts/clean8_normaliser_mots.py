import pandas as pd
import unicodedata
import re
import os
import sys

def nettoyer_texte(texte):
    if not isinstance(texte, str):
        return ""
    
    # 1. Mise en minuscule
    texte = texte.lower()
    
    # 2. Suppression des accents (Normalisation Unicode)
    texte = unicodedata.normalize('NFD', texte)
    texte = "".join([c for c in texte if unicodedata.category(c) != 'Mn'])
    
    # 3. Suppression de la ponctuation et des caractères spéciaux
    # On ne garde que les lettres et les espaces
    texte = re.sub(r'[^a-z\s]', ' ', texte)
    
    # 4. Suppression des espaces doubles
    texte = " ".join(texte.split())
    
    return texte

# --- CONFIGURATION AUTOMATIQUE DES CHEMINS ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_script_dir)

# Fichier d'entrée (le résultat du script précédent)
PATH_ENTREE = os.path.join(base_dir, "torchTestClassifiers", "data", "entrainer", "CNPS_Patterns_T_DE_S_NC.xlsx")
# Fichier de sortie final pour l'entraînement
PATH_SORTIE = os.path.join(base_dir, "torchTestClassifiers", "data", "entrainer", "entrainer_propre.xlsx")

# --- TRAITEMENT ---
if not os.path.exists(PATH_ENTREE):
    print(f"❌ Erreur : Le fichier source est introuvable : {PATH_ENTREE}")
    sys.exit(1)

print(f"⏳ Lecture du fichier : {os.path.basename(PATH_ENTREE)}")
df = pd.read_excel(PATH_ENTREE)

if 'nomenclature' in df.columns:
    print("🧹 Normalisation (minuscules, accents, ponctuation)...")
    df['nomenclature'] = df['nomenclature'].apply(nettoyer_texte)
    
    # Suppression des lignes vides après nettoyage
    avant = len(df)
    df = df[df['nomenclature'].str.strip() != ""]
    apres = len(df)
    
    if avant != apres:
        print(f"🗑️ {avant - apres} lignes vides supprimées.")

    print(f"💾 Sauvegarde finale : {PATH_SORTIE}")
    df.to_excel(PATH_SORTIE, index=False)
    print("✅ Normalisation terminée avec succès !")
else:
    print("❌ Erreur : Colonne 'nomenclature' introuvable.")
    sys.exit(1)