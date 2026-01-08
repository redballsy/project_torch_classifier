import pandas as pd
import os
import sys

# --- CONFIGURATION DES CHEMINS (AUTO-DÉTECTION) ---
# Détecte le dossier racine du projet
current_script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_script_dir)

# Chemin relatif (Indépendant de l'utilisateur Windows)
# On part du fichier généré par le script précédent
chemin_fichier = os.path.join(base_dir, "torchTestClassifiers", "data", "entrainer", "CNPS_Nomenclatures_Nettoyees.xlsx")

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

print(f"\nAnalyse de {len(df)} lignes...")
print("Recherche des nomenclatures qui sont UNIQUEMENT une lettre...\n")

modifications = []
compteur_nc = 0
lettres_trouvees = []

# --- BOUCLE DE TRAITEMENT ---
for index, row in df.iterrows():
    nomenclature = row[COLONNE_NOMENCLATURE]
    if pd.isna(nomenclature):
        continue
    
    nomenclature_str = str(nomenclature).strip()
    
    # Condition : Longueur exactement 1 ET est une lettre alphabétique
    if len(nomenclature_str) == 1 and nomenclature_str.isalpha():
        code_original = row.get(COLONNE_CODE, '')
        id_val = row.get(COLONNE_ID, 'N/A')
        
        if nomenclature_str not in lettres_trouvees:
            lettres_trouvees.append(nomenclature_str)
        
        ancien_code = str(code_original) if pd.notna(code_original) else 'vide'
        
        # Action : Remplacement par NC
        df.at[index, COLONNE_CODE] = "NC"
        compteur_nc += 1
        
        modifications.append({
            'ligne': index + 2,
            'id': id_val,
            'nomenclature': nomenclature_str,
            'ancien_code': ancien_code,
            'nouveau_code': "NC"
        })

# --- SAUVEGARDE DYNAMIQUE ---
dossier_source = os.path.dirname(chemin_fichier)
chemin_sauvegarde = os.path.join(dossier_source, 'CNPS_Lettres_Uniques_NC.xlsx')
rapport_path = os.path.join(dossier_source, 'Rapport_Lettres_Uniques_NC.csv')

df.to_excel(chemin_sauvegarde, index=False)

if modifications:
    rapport_df = pd.DataFrame(modifications)
    # encodage utf-8-sig pour Excel
    rapport_df.to_csv(rapport_path, index=False, sep=';', encoding='utf-8-sig')

print(f"\n✅ Succès ! {compteur_nc} codes remplacés par 'NC'.")
print(f"💾 Fichier : {chemin_sauvegarde}")
if lettres_trouvees:
    print(f"🔤 Lettres trouvées : {', '.join(sorted(lettres_trouvees))}")