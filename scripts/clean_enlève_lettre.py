import pandas as pd
import os

# Chemin complet vers votre fichier Excel
chemin_fichier = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\CNPS_Nomenclatures_Nettoyees.xlsx"

# Vérifier si le fichier existe
if not os.path.exists(chemin_fichier):
    print(f"ERREUR : Le fichier n'existe pas à l'emplacement : {chemin_fichier}")
    exit()

# Charger le fichier
print(f"Chargement du fichier : {chemin_fichier}")
df = pd.read_excel(chemin_fichier)

# Colonnes importantes
COLONNE_NOMENCLATURE = 'nomenclature'
COLONNE_CODE = 'code'
COLONNE_ID = 'id'

# Vérifier que les colonnes existent
if COLONNE_NOMENCLATURE not in df.columns:
    print(f"ERREUR : La colonne '{COLONNE_NOMENCLATURE}' n'existe pas.")
    exit()

print(f"\nAnalyse de {len(df)} lignes...")
print("Recherche des nomenclatures qui sont UNIQUEMENT une lettre...\n")

# Liste pour suivre les modifications
modifications = []
compteur_nc = 0
lettres_trouvees = []

# Analyser chaque ligne
for index, row in df.iterrows():
    nomenclature = row[COLONNE_NOMENCLATURE]
    
    if pd.isna(nomenclature):
        continue
    
    nomenclature_str = str(nomenclature).strip()
    
    # Vérifier si c'est EXACTEMENT une lettre (et rien d'autre)
    # Conditions:
    # 1. Longueur = 1
    # 2. C'est une lettre (pas un chiffre, pas un symbole)
    if len(nomenclature_str) == 1 and nomenclature_str.isalpha():
        # Récupérer les valeurs
        code_original = row.get(COLONNE_CODE, '')
        id_val = row.get(COLONNE_ID, 'N/A')
        
        # Enregistrer la lettre trouvée
        if nomenclature_str not in lettres_trouvees:
            lettres_trouvees.append(nomenclature_str)
        
        # Enregistrer l'ancien code
        ancien_code = str(code_original) if pd.notna(code_original) else 'vide'
        
        # Mettre "NC" dans la colonne code
        df.at[index, COLONNE_CODE] = "NC"
        compteur_nc += 1
        
        # Enregistrer la modification
        modifications.append({
            'ligne': index + 2,  # +2 pour correspondre à la numérotation Excel
            'id': id_val,
            'nomenclature': nomenclature_str,
            'ancien_code': ancien_code,
            'nouveau_code': "NC"
        })
        
        # Afficher immédiatement
        print(f"Ligne {index+2}: Lettre '{nomenclature_str}' → code: 'NC' (ancien: {ancien_code})")

print(f"\n{compteur_nc} codes remplacés par 'NC'.")

# Afficher les lettres uniques trouvées
if lettres_trouvees:
    print(f"\n=== LETTRES UNIQUES TROUVÉES ===")
    print(f"Nombre de lettres différentes : {len(lettres_trouvees)}")
    print(f"Lettres : {', '.join(sorted(lettres_trouvees))}")
    
    # Statistiques par lettre
    print(f"\n=== STATISTIQUES PAR LETTRE ===")
    compteur_par_lettre = {}
    for mod in modifications:
        lettre = mod['nomenclature']
        compteur_par_lettre[lettre] = compteur_par_lettre.get(lettre, 0) + 1
    
    for lettre, count in sorted(compteur_par_lettre.items()):
        print(f"  '{lettre}' : {count} occurrence(s)")

# Vérifier ce qui n'est PAS détecté (pour comprendre)
print(f"\n=== VÉRIFICATION DE CE QUI N'EST PAS DÉTECTÉ ===")
exemples_non_detectes = []
for index, row in df.iterrows():
    nomenclature = row[COLONNE_NOMENCLATURE]
    
    if pd.isna(nomenclature):
        continue
    
    nomenclature_str = str(nomenclature).strip()
    
    # Exemples à vérifier (mais pas à modifier)
    if len(nomenclature_str) > 1 and nomenclature_str.replace('.', '').isalpha():
        # C'est des lettres mais plus d'une lettre
        if len(exemples_non_detectes) < 10:  # Limiter à 10 exemples
            exemples_non_detectes.append(nomenclature_str)

if exemples_non_detectes:
    print("Exemples NON modifiés (plus d'une lettre ou avec points) :")
    for exemple in exemples_non_detectes:
        print(f"  '{exemple}'")

# Sauvegarder le résultat
dossier_source = os.path.dirname(chemin_fichier)
chemin_sauvegarde = os.path.join(dossier_source, 'CNPS_Lettres_Uniques_NC.xlsx')

df.to_excel(chemin_sauvegarde, index=False)
print(f"\nFichier sauvegardé sous : {chemin_sauvegarde}")

# Sauvegarder un rapport des modifications
if modifications:
    rapport_path = os.path.join(dossier_source, 'Rapport_Lettres_Uniques_NC.csv')
    
    # Créer un DataFrame pour le rapport
    rapport_df = pd.DataFrame(modifications)
    rapport_df.to_csv(rapport_path, index=False, sep=';', encoding='utf-8')
    
    print(f"Rapport des modifications : {rapport_path}")
    
    # Afficher un résumé du rapport
    print(f"\n=== RAPPORT DES MODIFICATIONS ===")
    print(rapport_df[['ligne', 'nomenclature', 'ancien_code']].to_string(index=False))

# Afficher un résumé final
print("\n" + "="*50)
print("RÉSUMÉ FINAL")
print("="*50)
print(f"Fichier source        : {chemin_fichier}")
print(f"Lignes totales        : {len(df)}")
print(f"Lettres uniques       : {compteur_nc}")
print(f"Pourcentage modifié   : {compteur_nc/len(df)*100:.2f}%")
print(f"Fichier de sortie     : {chemin_sauvegarde}")

# Exemples de ce qui sera et ne sera pas modifié
print("\n" + "="*50)
print("EXEMPLES DE DÉTECTION")
print("="*50)
exemples_test = [
    ("A", True),      # ✓ Une seule lettre
    ("B", True),      # ✓ Une seule lettre  
    ("X", True),      # ✓ Une seule lettre
    ("1", False),     # ✗ Un chiffre
    ("@", False),     # ✗ Un symbole
    (" ", False),     # ✗ Un espace
    ("AB", False),    # ✗ Deux lettres
    ("A.B", False),   # ✗ Lettre avec point
    ("AAA", False),   # ✗ Trois lettres
    ("A ", False),    # ✗ Lettre avec espace (sera stripped)
]

print("Ce qui sera modifié :")
for texte, sera_modifie in exemples_test:
    if sera_modifie:
        print(f"  ✓ '{texte}' → 'NC'")
print("\nCe qui ne sera PAS modifié :")
for texte, sera_modifie in exemples_test:
    if not sera_modifie:
        print(f"  ✗ '{texte}' → inchangé")