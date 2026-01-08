import pandas as pd
import re
import os
import sys

# --- CONFIGURATION AUTOMATIQUE DES CHEMINS ---
# Détecte le dossier racine du projet (project_torch_classifier)
# os.path.abspath(__file__) donne le chemin du script actuel
# os.path.dirname remonte d'un niveau
current_script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_script_dir)

# Chemin relatif vers le fichier Excel (Utilise os.path.join pour les / et \)
chemin_fichier = os.path.join(base_dir, "torchTestClassifiers", "data", "entrainer", "CNPS_A_Nettoyer.xlsx")

# --- VÉRIFICATION DU FICHIER ---
if not os.path.exists(chemin_fichier):
    print(f"ERREUR : Le fichier n'existe pas à l'emplacement : {chemin_fichier}")
    print("Vérifiez que le dossier 'torchTestClassifiers' est bien présent à la racine.")
    sys.exit(1) # Quitter avec une erreur pour stopper GitHub Actions si besoin

# Charger le fichier
print(f"Chargement du fichier : {chemin_fichier}")
df = pd.read_excel(chemin_fichier)

# ... (Le reste de ta logique de fonction est parfaite et ne change pas) ...

def est_purement_numerique(valeur):
    if pd.isna(valeur):
        return False
    valeur_str = str(valeur).strip()
    if not valeur_str:
        return False
    if re.fullmatch(r'^\d+([.,]\d+)?$', valeur_str):
        return True
    if valeur_str.replace(' ', '').isdigit():
        return True
    return False

print(f"\nAnalyse de {len(df)} lignes...")
compteur_nc = 0
modifications = []

# Pour éviter le warning de type dans Pandas
df['code'] = df['code'].astype(object)

for index, row in df.iterrows():
    nomenclature = row['nomenclature']
    if est_purement_numerique(nomenclature):
        ancien_code = row['code']
        id_val = row['id']
        
        df.at[index, 'code'] = "NC"
        compteur_nc += 1
        
        modifications.append({
            'ligne': index + 2,
            'id': id_val,
            'ancienne_nomenclature': nomenclature,
            'ancien_code': ancien_code,
            'nouveau_code': "NC"
        })

# --- SAUVEGARDE DYNAMIQUE ---
# On récupère le dossier où se trouve le fichier source pour enregistrer à côté
dossier_source = os.path.dirname(chemin_fichier)
chemin_sauvegarde = os.path.join(dossier_source, 'CNPS_Code_NC.xlsx')
chemin_modifications = os.path.join(dossier_source, 'Modifications_NC.csv')

df.to_excel(chemin_sauvegarde, index=False)
if modifications:
    df_modifications = pd.DataFrame(modifications)
    df_modifications.to_csv(chemin_modifications, index=False, sep=';', encoding='utf-8')

print(f"\n✅ Succès ! {compteur_nc} lignes traitées.")
print(f"💾 Sortie : {chemin_sauvegarde}")