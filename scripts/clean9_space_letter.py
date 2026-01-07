import pandas as pd
import re
import os
import sys

# --- CONFIGURATION AUTOMATIQUE DES CHEMINS ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_script_dir)

# On utilise le fichier généré par le script de normalisation précédent
file_path = os.path.join(base_dir, "torchTestClassifiers", "data", "entrainer", "entrainer_propre.xlsx")
output_path = os.path.join(base_dir, "torchTestClassifiers", "data", "entrainer", "entrainer2_propre.xlsx")

# --- VÉRIFICATION ---
if not os.path.exists(file_path):
    print(f"❌ ERREUR : Le fichier source est introuvable : {file_path}")
    sys.exit(1)

# 1. Chargement
print(f"⏳ Chargement de : {os.path.basename(file_path)}")
df = pd.read_excel(file_path)

# S'assurer que la colonne code accepte du texte
df['code'] = df['code'].astype(object)

# 2. Définition du Pattern Regex
# Capture les chaînes composées uniquement de groupes de 1 à 3 lettres
# Exemple : "a", "a b c", "aaa", "sdo aa"
nc_pattern = r'^([a-z]{1,3})(\s[a-z]{1,3})*$'

# 3. Identification et Mise à jour
# On travaille sur la colonne 'nomenclature' déjà mise en minuscule par le script précédent
mask = df['nomenclature'].astype(str).str.strip().str.match(nc_pattern, na=False)

nb_updates = mask.sum()
df.loc[mask, 'code'] = 'NC'

# 4. Sauvegarde
print(f"🧹 Nettoyage final : {nb_updates} lignes marquées comme 'NC'.")
df.to_excel(output_path, index=False)

print(f"✅ Fichier final prêt pour l'entraînement : {output_path}")