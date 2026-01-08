import pandas as pd
import re
import os
import sys

# --- CONFIGURATION AUTOMATIQUE DES CHEMINS ---
# Localise la racine du projet à partir de l'emplacement du script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_script_dir)

# Chemin relatif (Indépendant du nom d'utilisateur Windows)
chemin_fichier = os.path.join(base_dir, "torchTestClassifiers", "data", "entrainer", "CNPS_Special_Chars_NC.xlsx")

# --- VÉRIFICATION DU FICHIER ---
if not os.path.exists(chemin_fichier):
    print(f"ERREUR : Le fichier n'existe pas à l'emplacement : {chemin_fichier}")
    sys.exit(1)

# Charger le fichier
print(f"Chargement du fichier : {chemin_fichier}")
df = pd.read_excel(chemin_fichier)

# Colonnes importantes
COLONNE_NOMENCLATURE = 'nomenclature'
COLONNE_CODE = 'code'
COLONNE_ID = 'id'

# --- LOGIQUE DE NETTOYAGE ---
def nettoyer_bords_detail(texte):
    if pd.isna(texte):
        return texte, "", ""
    
    original = str(texte)
    nettoye = original
    
    # Caractères supprimés au début (tout ce qui n'est pas une lettre)
    match_debut = re.match(r'^([^a-zA-ZÀ-ÿ]+)', nettoye)
    supprimes_debut = match_debut.group(1) if match_debut else ""
    nettoye = re.sub(r'^[^a-zA-ZÀ-ÿ]+', '', nettoye)
    
    # Caractères supprimés à la fin
    match_fin = re.search(r'([^a-zA-ZÀ-ÿ]+)$', nettoye)
    supprimes_fin = match_fin.group(1) if match_fin else ""
    nettoye = re.sub(r'[^a-zA-ZÀ-ÿ]+$', '', nettoye)
    
    return nettoye.strip(), supprimes_debut, supprimes_fin

print(f"\nAnalyse de {len(df)} lignes...")
modifications = []
compteur_modifs = 0

for index, row in df.iterrows():
    original = row[COLONNE_NOMENCLATURE]
    if pd.isna(original):
        continue
    
    original_str = str(original)
    nettoye, supprimes_debut, supprimes_fin = nettoyer_bords_detail(original_str)
    
    if nettoye != original_str:
        compteur_modifs += 1
        modifications.append({
            'ligne': index + 2,
            'id': row.get(COLONNE_ID, 'N/A'),
            'original': original_str,
            'nettoye': nettoye,
            'supprime_debut': supprimes_debut,
            'supprime_fin': supprimes_fin
        })
        df.at[index, COLONNE_NOMENCLATURE] = nettoye

# --- SAUVEGARDE DYNAMIQUE ---
dossier_source = os.path.dirname(chemin_fichier)
chemin_sauvegarde = os.path.join(dossier_source, 'CNPS_Nomenclatures_Nettoyees.xlsx')
rapport_path = os.path.join(dossier_source, 'Rapport_Nettoyage_Bords.csv')

df.to_excel(chemin_sauvegarde, index=False)

if modifications:
    df_modifications = pd.DataFrame(modifications)
    # Utilisation de l'encodage utf-8-sig pour assurer la compatibilité Excel/Linux
    df_modifications.to_csv(rapport_path, index=False, sep=';', encoding='utf-8-sig')

print(f"\n✅ Terminé : {compteur_modifs} lignes nettoyées.")
print(f"💾 Fichier : {chemin_sauvegarde}")
print(f"💾 Rapport : {rapport_path}")