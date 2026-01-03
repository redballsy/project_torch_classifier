import pandas as pd
import os
import re

# Chemin complet vers votre fichier Excel
chemin_fichier = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\CNPS_Lettres_Uniques_NC.xlsx"

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

# Définir les patterns regex pour détecter les styles spécifiques
patterns_a_detecter = [
    # Pattern 1: Deux lettres (AB, CD, XY, etc.)
    r'^[A-Z]{2}$',
    
    # Pattern 2: Lettre.Point.Lettre (A.B, X.Y, etc.)
    r'^[A-Z]\.[A-Z]$',
    
    # Pattern 3: Trois lettres identiques (AAA, BBB, CCC, etc.)
    r'^([A-Z])\1{2}$',
    
    # Pattern 4: Lettre.Point.Lettre.Point.Lettre (A.B.C, X.Y.Z, etc.)
    r'^[A-Z]\.[A-Z]\.[A-Z]$',
    
    # Pattern 5: Quatre lettres identiques (AAAA, BBBB, etc.)
    r'^([A-Z])\1{3}$',
    
    # Pattern 6: Deux lettres identiques (AA, BB, CC, etc.)
    r'^([A-Z])\1$',
    
    # Pattern 7: Trois lettres (pas forcément identiques) - ABC, XYZ, etc.
    r'^[A-Z]{3}$',
]

print(f"\nAnalyse de {len(df)} lignes...")
print("Recherche des nomenclatures avec patterns de lettres...\n")

# Liste pour suivre les modifications
modifications = []
compteur_nc = 0

# Analyser chaque ligne
for index, row in df.iterrows():
    nomenclature = row[COLONNE_NOMENCLATURE]
    
    if pd.isna(nomenclature):
        continue
    
    nomenclature_str = str(nomenclature).strip()
    
    # Vérifier si c'est en majuscules (pour nos patterns)
    if nomenclature_str.isupper():
        pattern_trouve = None
        
        # Tester chaque pattern
        for i, pattern in enumerate(patterns_a_detecter, 1):
            if re.fullmatch(pattern, nomenclature_str):
                pattern_trouve = i
                break
        
        # Si un pattern a été trouvé
        if pattern_trouve is not None:
            # Récupérer les valeurs
            code_original = row.get(COLONNE_CODE, '')
            id_val = row.get(COLONNE_ID, 'N/A')
            
            # Enregistrer l'ancien code
            ancien_code = str(code_original) if pd.notna(code_original) else 'vide'
            
            # Mettre "NC" dans la colonne code
            df.at[index, COLONNE_CODE] = "NC"
            compteur_nc += 1
            
            # Déterminer le type de pattern
            types_patterns = {
                1: "deux_lettres",
                2: "lettre_point_lettre", 
                3: "trois_lettres_identiques",
                4: "lettre_point_lettre_point_lettre",
                5: "quatre_lettres_identiques",
                6: "deux_lettres_identiques",
                7: "trois_lettres"
            }
            
            type_pattern = types_patterns.get(pattern_trouve, f"pattern_{pattern_trouve}")
            
            # Enregistrer la modification
            modifications.append({
                'ligne': index + 2,
                'id': id_val,
                'nomenclature': nomenclature_str,
                'pattern': pattern_trouve,
                'type': type_pattern,
                'ancien_code': ancien_code,
                'nouveau_code': "NC"
            })
            
            # Afficher
            print(f"Ligne {index+2}: '{nomenclature_str}' ({type_pattern}) → code: 'NC'")

print(f"\n{compteur_nc} codes remplacés par 'NC'.")

# Statistiques détaillées
if modifications:
    print(f"\n=== STATISTIQUES DÉTAILLÉES ===")
    
    # Par type de pattern
    print("\nDistribution par type de pattern :")
    types_comptes = {}
    for mod in modifications:
        type_pattern = mod['type']
        types_comptes[type_pattern] = types_comptes.get(type_pattern, 0) + 1
    
    for type_pattern, count in sorted(types_comptes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {type_pattern} : {count}")
    
    # Les nomenclatures les plus courantes
    print("\nNomenclatures les plus courantes :")
    noms_comptes = {}
    for mod in modifications:
        nom = mod['nomenclature']
        noms_comptes[nom] = noms_comptes.get(nom, 0) + 1
    
    for nom, count in sorted(noms_comptes.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  '{nom}' : {count}")

# Test de la détection
print(f"\n=== TEST DE DÉTECTION DES PATTERNS ===")
test_cases = [
    ("AB", True, "deux_lettres"),
    ("A.B", True, "lettre_point_lettre"),
    ("AAA", True, "trois_lettres_identiques"),
    ("A.B.C", True, "lettre_point_lettre_point_lettre"),
    ("AAAA", True, "quatre_lettres_identiques"),
    ("AA", True, "deux_lettres_identiques"),
    ("ABC", True, "trois_lettres"),
    ("XYZ", True, "trois_lettres"),
    ("A", False, "une lettre seulement"),
    ("ABCD", False, "quatre lettres différentes"),
    ("A.B", False, "avec minuscule"),
    ("ab", False, "minuscules"),
    ("A.B.C.D", False, "trop de points"),
    ("1.2.3", False, "chiffres"),
    ("@#$", False, "symboles"),
]

print("Exemples qui seront modifiés :")
for texte, sera_modifie, description in test_cases:
    if sera_modifie:
        print(f"  ✓ '{texte}' → 'NC' ({description})")

print("\nExemples qui ne seront PAS modifiés :")
for texte, sera_modifie, description in test_cases:
    if not sera_modifie:
        print(f"  ✗ '{texte}' → inchangé ({description})")

# Vérifier les limites
print(f"\n=== VÉRIFICATION DES LIMITES ===")
print("Conditions pour être détecté :")
print("  1. Doit être en MAJUSCULES")
print("  2. Doit correspondre à un pattern spécifique")
print("  3. Pas de chiffres ou symboles (sauf points séparateurs)")

# Sauvegarder le résultat
dossier_source = os.path.dirname(chemin_fichier)
chemin_sauvegarde = os.path.join(dossier_source, 'CNPS_Patterns_Lettres_NC.xlsx')

df.to_excel(chemin_sauvegarde, index=False)
print(f"\nFichier sauvegardé sous : {chemin_sauvegarde}")

# Sauvegarder un rapport détaillé
if modifications:
    rapport_path = os.path.join(dossier_source, 'Rapport_Patterns_Lettres.csv')
    rapport_df = pd.DataFrame(modifications)
    rapport_df.to_csv(rapport_path, index=False, sep=';', encoding='utf-8')
    print(f"Rapport détaillé : {rapport_path}")
    
    # Afficher un aperçu du rapport
    print(f"\n=== APERÇU DU RAPPORT (premières 10 lignes) ===")
    print(rapport_df[['ligne', 'nomenclature', 'type', 'ancien_code']].head(10).to_string(index=False))

# Résumé final
print("\n" + "="*50)
print("RÉSUMÉ FINAL")
print("="*50)
print(f"Fichier analysé       : {os.path.basename(chemin_fichier)}")
print(f"Lignes totales        : {len(df)}")
print(f"Patterns détectés     : {compteur_nc}")
print(f"Pourcentage modifié   : {compteur_nc/len(df)*100:.2f}%")
print(f"Fichier de sortie     : {os.path.basename(chemin_sauvegarde)}")
print(f"Patterns recherchés   : {len(patterns_a_detecter)}")