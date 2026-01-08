import pandas as pd
import re
import os

# 1. On trouve où se situe le script actuel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. On construit le chemin à partir de là
# Chemin complet vers votre fichier Excel
chemin_fichier = os.path.join(BASE_DIR, "data", "entrainer", "CNPS_Code_NC.xlsx")

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

# Fonction pour détecter si une chaîne ne contient QUE des caractères spéciaux
def est_uniquement_caracteres_speciaux(valeur):
    if pd.isna(valeur):
        return False
    
    valeur_str = str(valeur).strip()
    
    # Si vide
    if not valeur_str:
        return False
    
    # Si ce sont uniquement des espaces
    if all(c.isspace() for c in valeur_str):
        return False
    
    # Vérifier s'il y a au moins un caractère alphanumérique (lettre ou chiffre)
    # Pattern: cherche n'importe quelle lettre (maj/min, avec accents) ou chiffre
    if re.search(r'[a-zA-Z0-9À-ÿ]', valeur_str):
        return False  # Contient au moins un caractère alphanumérique
    
    # Si on arrive ici, c'est qu'il n'y a QUE des caractères spéciaux
    # Mais on veut s'assurer qu'il y a au moins un caractère non-espace
    caracteres_non_espaces = [c for c in valeur_str if not c.isspace()]
    if not caracteres_non_espaces:
        return False  # Que des espaces
    
    return True

print(f"\nAnalyse de {len(df)} lignes...")
compteur_nc = 0

# Liste pour suivre les modifications
modifications = []
exemples_trouves = []

print("\nRecherche des nomenclatures avec uniquement des caractères spéciaux...")

for index, row in df.iterrows():
    nomenclature = row[COLONNE_NOMENCLATURE]
    
    # Vérifier si la nomenclature ne contient que des caractères spéciaux
    if est_uniquement_caracteres_speciaux(nomenclature):
        ancien_code = row[COLONNE_CODE]
        id_val = row[COLONNE_ID]
        
        # Enregistrer un exemple
        if len(exemples_trouves) < 15:
            exemples_trouves.append({
                'nomenclature': nomenclature,
                'ancien_code': ancien_code,
                'id': id_val
            })
        
        # Mettre "NC" dans la colonne code (remplace l'ancien code)
        df.at[index, COLONNE_CODE] = "NC"
        compteur_nc += 1
        
        # Enregistrer la modification
        modifications.append({
            'ligne': index + 2,
            'id': id_val,
            'nomenclature': nomenclature,
            'ancien_code': ancien_code,
            'action': f"Code '{ancien_code}' remplacé par 'NC'"
        })
        
        # Afficher les cas avec des caractères spéciaux évidents
        if re.search(r'[!@#$%^&*()_+\-=\[\]{};:"\\|,.<>\/?]+', str(nomenclature)):
            print(f"✓ Ligne {index+2}: ID {id_val} - '{nomenclature}' → Ancien code: '{ancien_code}' → Nouveau: 'NC'")

print(f"\n{compteur_nc} lignes modifiées (remplacement par 'NC').")

# Afficher des exemples détaillés
if exemples_trouves:
    print("\n=== EXEMPLES DÉTAILLÉS DES CARACTÈRES SPÉCIAUX DÉTECTÉS ===")
    for i, exemple in enumerate(exemples_trouves, 1):
        nom = exemple['nomenclature']
        # Afficher la représentation Python pour voir les caractères cachés
        print(f"{i}. '{nom}'")
        print(f"   Repr Python: {repr(nom)}")
        print(f"   Longueur: {len(str(nom))} caractères")
        print(f"   ID: {exemple['id']}, Ancien code: {exemple['ancien_code']}")
        print()

# Afficher un résumé des modifications
if modifications:
    print("\n=== RÉSUMÉ DES MODIFICATIONS ===")
    
    # Statistiques par type d'ancien code
    codes_remplaces = {}
    for mod in modifications:
        ancien = mod['ancien_code']
        if pd.notna(ancien):
            codes_remplaces[ancien] = codes_remplaces.get(ancien, 0) + 1
    
    print(f"\nCodes remplacés par 'NC' (Top 10):")
    for code, count in sorted(codes_remplaces.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  Code {code}: {count} fois")
    
    # Catégoriser les types de caractères spéciaux
    categories = {
        'Ponctuation': 0,
        'Symboles': 0,
        'Tirets/soulignés': 0,
        'Autres': 0
    }
    
    for mod in modifications:
        nom = str(mod['nomenclature'])
        if re.fullmatch(r'^[.,!?;:]+$', nom):
            categories['Ponctuation'] += 1
        elif re.fullmatch(r'^[@#$%^&*()+=\[\]{}|<>/~`]+$', nom):
            categories['Symboles'] += 1
        elif re.fullmatch(r'^[-_]+$', nom):
            categories['Tirets/soulignés'] += 1
        else:
            categories['Autres'] += 1
    
    print("\nCatégories de caractères spéciaux trouvés:")
    for categorie, count in categories.items():
        if count > 0:
            print(f"  {categorie}: {count}")

# Sauvegarder le résultat
dossier_source = os.path.dirname(chemin_fichier)
chemin_sauvegarde = os.path.join(dossier_source, 'CNPS_Special_Chars_NC.xlsx')

df.to_excel(chemin_sauvegarde, index=False)
print(f"\nFichier sauvegardé sous : {chemin_sauvegarde}")

# Créer un rapport détaillé
if modifications:
    rapport_path = os.path.join(dossier_source, 'Rapport_Caracteres_Speciaux.txt')
    with open(rapport_path, 'w', encoding='utf-8') as f:
        f.write("=== RAPPORT: REMPLACEMENT DES CODES POUR CARACTÈRES SPÉCIAUX ===\n\n")
        f.write(f"Fichier source: {chemin_fichier}\n")
        f.write(f"Date d'analyse: {pd.Timestamp.now()}\n")
        f.write(f"Total lignes analysées: {len(df)}\n")
        f.write(f"Lignes modifiées: {len(modifications)}\n")
        f.write(f"Pourcentage: {len(modifications)/len(df)*100:.2f}%\n\n")
        
        f.write("=== EXEMPLES DE MODIFICATIONS ===\n")
        for mod in modifications[:50]:  # Limiter à 50 exemples
            f.write(f"Ligne {mod['ligne']}: ID {mod['id']}\n")
            f.write(f"  Nomenclature: '{mod['nomenclature']}'\n")
            f.write(f"  Ancien code: {mod['ancien_code']} → Nouveau: 'NC'\n")
            f.write(f"  {mod['action']}\n\n")
    
    print(f"Rapport détaillé sauvegardé sous : {rapport_path}")

# Afficher un résumé final
print("\n=== RÉSUMÉ FINAL ===")
print(f"Fichier source : {chemin_fichier}")
print(f"Lignes totales : {len(df)}")
print(f"Lignes avec uniquement caractères spéciaux : {compteur_nc}")
print(f"Pourcentage : {compteur_nc/len(df)*100:.2f}%")
print(f"Fichier de sortie : {chemin_sauvegarde}")

# Vérification: afficher quelques lignes avant/après
print("\n=== VÉRIFICATION (5 premières modifications) ===")
for i, mod in enumerate(modifications[:5], 1):
    print(f"{i}. Ligne {mod['ligne']}")
    print(f"   Avant: Nomenclature='{mod['nomenclature']}', Code={mod['ancien_code']}")
    print(f"   Après: Code='NC'")
    print()