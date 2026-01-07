import pandas as pd
import os
import re
import sys

# --- CONFIGURATION DES CHEMINS (AUTO-DÉTECTION) ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_script_dir)

# On utilise le fichier généré par l'étape précédente (Patterns Minuscules)
chemin_fichier = os.path.join(base_dir, "torchTestClassifiers", "data", "entrainer", "CNPS_Patterns_Minuscules_NC.xlsx")

# --- VÉRIFICATION ---
if not os.path.exists(chemin_fichier):
    print(f"ERREUR : Le fichier n'existe pas : {chemin_fichier}")
    sys.exit(1)

# Charger le fichier
print(f"Chargement du fichier : {chemin_fichier}")
df = pd.read_excel(chemin_fichier)

# Colonnes importantes
COLONNE_NOMENCLATURE = 'nomenclature'
COLONNE_CODE = 'code'
COLONNE_ID = 'id'

# S'assurer que la colonne code accepte du texte
df[COLONNE_CODE] = df[COLONNE_CODE].astype(object)

# Définir les patterns regex pour détecter les styles spécifiques
patterns_a_detecter = [
    r'^[A-Z]\.\s+[A-Z]{2,3}\s+[A-Z]$',  # T. DE S
    r'^[A-Z]\.\s+[A-Z]$',               # A. B
    r'^[A-Z]\.[A-Z]$',                  # A.B
    r'^[A-Z]\s+[A-Z]$',                 # A B
    r'^[A-Z]{1,3}\.[A-Z]{1,3}$',        # AB.CD
    r'^[A-Z]{2,3}\s+[A-Z]\.$',          # DE T.
    r'^[A-Z]\.[A-Z]{2,3}$',             # T.DE
]

print(f"\nAnalyse de {len(df)} lignes...")
modifications = []
compteur_nc = 0

# --- BOUCLE DE TRAITEMENT ---
for index, row in df.iterrows():
    nomenclature = row[COLONNE_NOMENCLATURE]
    if pd.isna(nomenclature):
        continue
    
    nomenclature_str = str(nomenclature).strip()
    
    if nomenclature_str.isupper():
        pattern_trouve = None
        for i, pattern in enumerate(patterns_a_detecter, 1):
            if re.fullmatch(pattern, nomenclature_str):
                pattern_trouve = i
                break
        
        if pattern_trouve is not None:
            code_original = row.get(COLONNE_CODE, '')
            id_val = row.get(COLONNE_ID, 'N/A')
            ancien_code = str(code_original) if pd.notna(code_original) else 'vide'
            
            # Action : Remplacement par NC
            df.at[index, COLONNE_CODE] = "NC"
            compteur_nc += 1
            
            modifications.append({
                'ligne': index + 2,
                'id': id_val,
                'nomenclature': nomenclature_str,
                'pattern_index': pattern_trouve,
                'ancien_code': ancien_code,
                'nouveau_code': "NC"
            })

# --- SAUVEGARDE DYNAMIQUE ---
dossier_source = os.path.dirname(chemin_fichier)
chemin_sauvegarde = os.path.join(dossier_source, 'CNPS_Patterns_T_DE_S_NC.xlsx')
rapport_path = os.path.join(dossier_source, 'Rapport_Patterns_T_DE_S.csv')

df.to_excel(chemin_sauvegarde, index=False)

if modifications:
    pd.DataFrame(modifications).to_csv(rapport_path, index=False, sep=';', encoding='utf-8-sig')

print(f"\n✅ Terminé : {compteur_nc} abréviations complexes marquées 'NC'.")
print(f"💾 Sortie : {chemin_sauvegarde}")