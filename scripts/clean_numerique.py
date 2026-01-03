import pandas as pd
import re
import os

# Chemin complet vers votre fichier Excel
chemin_fichier = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\CNPS_A_Nettoyer.xlsx"

# Vérifier si le fichier existe
if not os.path.exists(chemin_fichier):
    print(f"ERREUR : Le fichier n'existe pas à l'emplacement : {chemin_fichier}")
    print("Vérifiez le chemin et réessayez.")
    exit()

# Charger le fichier
print(f"Chargement du fichier : {chemin_fichier}")
df = pd.read_excel(chemin_fichier)

# Afficher les premières lignes pour vérifier
print(f"Fichier chargé : {len(df)} lignes, {len(df.columns)} colonnes")
print("Colonnes disponibles :", df.columns.tolist())

# Colonnes importantes
COLONNE_NOMENCLATURE = 'nomenclature'
COLONNE_CODE = 'code'
COLONNE_ID = 'id'

# Vérifier que les colonnes existent
colonnes_requises = [COLONNE_NOMENCLATURE, COLONNE_CODE, COLONNE_ID]
for col in colonnes_requises:
    if col not in df.columns:
        print(f"ERREUR : La colonne '{col}' n'existe pas dans le fichier.")
        print(f"Colonnes disponibles : {df.columns.tolist()}")
        exit()

# Fonction pour vérifier si une valeur est purement numérique
def est_purement_numerique(valeur):
    if pd.isna(valeur):
        return False
    
    # Convertir en chaîne
    valeur_str = str(valeur).strip()
    
    # Si vide
    if not valeur_str:
        return False
    
    # Vérifier si c'est un nombre (entier ou décimal)
    # Pattern pour: chiffres uniquement, avec ou sans décimales, avec ou sans espaces
    if re.fullmatch(r'^\d+([.,]\d+)?$', valeur_str):
        return True
    
    # Vérifier si c'est uniquement des chiffres, même avec espaces autour
    if valeur_str.replace(' ', '').isdigit():
        return True
    
    return False

print(f"\nAnalyse de {len(df)} lignes...")
compteur_nc = 0

# Liste pour suivre les modifications
modifications = []

for index, row in df.iterrows():
    nomenclature = row[COLONNE_NOMENCLATURE]
    
    # Vérifier si la nomenclature est purement numérique
    if est_purement_numerique(nomenclature):
        ancien_code = row[COLONNE_CODE]
        id_val = row[COLONNE_ID]
        
        # Mettre "NC" dans la colonne code
        df.at[index, COLONNE_CODE] = "NC"
        compteur_nc += 1
        
        # Enregistrer la modification
        modifications.append({
            'ligne': index + 2,  # +2 car Excel commence à 1 et header à ligne 1
            'id': id_val,
            'ancienne_nomenclature': nomenclature,
            'ancien_code': ancien_code,
            'nouveau_code': "NC"
        })
        
        print(f"Ligne {index+2}: ID {id_val} - '{nomenclature}' → code: 'NC' (ancien: {ancien_code})")

print(f"\n{compteur_nc} lignes marquées comme 'NC'.")

# Afficher un résumé des modifications
if modifications:
    print("\n=== RÉSUMÉ DES MODIFICATIONS ===")
    print(f"Total de lignes marquées 'NC': {len(modifications)}")
    
    # Créer un DataFrame pour les modifications
    df_modifications = pd.DataFrame(modifications)
    print("\nExemples de modifications :")
    print(df_modifications.head(10).to_string(index=False))
    
    # Statistiques sur les anciens codes remplacés
    print("\n--- Statistiques des anciens codes remplacés ---")
    anciens_codes = df_modifications['ancien_code'].value_counts()
    print(anciens_codes.head(10))

# Sauvegarder le résultat
dossier_source = os.path.dirname(chemin_fichier)
chemin_sauvegarde = os.path.join(dossier_source, 'CNPS_Code_NC.xlsx')

df.to_excel(chemin_sauvegarde, index=False)
print(f"\nFichier sauvegardé sous : {chemin_sauvegarde}")

# Créer également un fichier CSV des modifications
chemin_modifications = os.path.join(dossier_source, 'Modifications_NC.csv')
df_modifications.to_csv(chemin_modifications, index=False, sep=';', encoding='utf-8')
print(f"Liste des modifications sauvegardée sous : {chemin_modifications}")

# Afficher un résumé final
print("\n=== RÉSUMÉ FINAL ===")
print(f"Fichier source : {chemin_fichier}")
print(f"Lignes totales : {len(df)}")
print(f"Lignes marquées 'NC' : {compteur_nc}")
print(f"Pourcentage : {compteur_nc/len(df)*100:.2f}%")
print(f"Fichier de sortie : {chemin_sauvegarde}")