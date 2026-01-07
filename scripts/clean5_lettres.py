import pandas as pd
import os
import re
import sys

# --- CONFIGURATION DES CHEMINS (AUTO-DÉTECTION) ---
# Détecte le dossier racine du projet (project_torch_classifier)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_script_dir)

# Chemin relatif vers le fichier généré par le script précédent
chemin_fichier = os.path.join(base_dir, "torchTestClassifiers", "data", "entrainer", "CNPS_Lettres_Uniques_NC.xlsx")

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

# Définir les patterns regex
patterns_a_detecter = [
    r'^[A-Z]{2}$',                      # Pattern 1: Deux lettres (AB)
    r'^[A-Z]\.[A-Z]$',                  # Pattern 2: Lettre.Point.Lettre (A.B)
    r'^([A-Z])\1{2}$',                  # Pattern 3: Trois lettres identiques (AAA)
    r'^[A-Z]\.[A-Z]\.[A-Z]$',           # Pattern 4: A.B.C
    r'^([A-Z])\1{3}$',                  # Pattern 5: Quatre lettres identiques (AAAA)
    r'^([A-Z])\1$',                     # Pattern 6: Deux lettres identiques (AA)
    r'^[A-Z]{3}$',                      # Pattern 7: Trois lettres (ABC)
]

print(f"\nAnalyse de {len(df)} lignes...")
modifications = []
compteur_nc = 0

# Dictionnaire pour le nommage des types dans le rapport
types_patterns = {
    1: "deux_lettres",
    2: "lettre_point_lettre", 
    3: "trois_lettres_identiques",
    4: "lettre_point_lettre_point_lettre",
    5: "quatre_lettres_identiques",
    6: "deux_lettres_identiques",
    7: "trois_lettres"
}

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
            
            # Action : Remplacement
            df.at[index, COLONNE_CODE] = "NC"
            compteur_nc += 1
            
            type_pattern = types_patterns.get(pattern_trouve, f"pattern_{pattern_trouve}")
            modifications.append({
                'ligne': index + 2,
                'id': id_val,
                'nomenclature': nomenclature_str,
                'type': type_pattern,
                'ancien_code': ancien_code,
                'nouveau_code': "NC"
            })

# --- SAUVEGARDE DYNAMIQUE ---
dossier_source = os.path.dirname(chemin_fichier)
chemin_sauvegarde = os.path.join(dossier_source, 'CNPS_Patterns_Lettres_NC.xlsx')
rapport_path = os.path.join(dossier_source, 'Rapport_Patterns_Lettres.csv')

df.to_excel(chemin_sauvegarde, index=False)

if modifications:
    rapport_df = pd.DataFrame(modifications)
    rapport_df.to_csv(rapport_path, index=False, sep=';', encoding='utf-8-sig')

print(f"\n✅ Succès ! {compteur_nc} patterns identifiés et marqués 'NC'.")
print(f"💾 Fichier : {chemin_sauvegarde}")
print(f"💾 Rapport : {rapport_path}")