import pandas as pd
import os
import re

# Chemin complet vers votre fichier Excel
chemin_fichier = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\CNPS_Patterns_Lettres_NC.xlsx"

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

# Définir les patterns regex pour détecter les styles en MINUSCULES
patterns_a_detecter_minuscules = [
    # Pattern 1: Deux lettres minuscules (ab, cd, xy, etc.)
    r'^[a-z]{2}$',
    
    # Pattern 2: Lettre.Point.Lettre minuscules (a.b, x.y, etc.)
    r'^[a-z]\.[a-z]$',
    
    # Pattern 3: Trois lettres identiques minuscules (aaa, bbb, ccc, etc.)
    r'^([a-z])\1{2}$',
    
    # Pattern 4: Lettre.Point.Lettre.Point.Lettre minuscules (a.b.c, x.y.z, etc.)
    r'^[a-z]\.[a-z]\.[a-z]$',
    
    # Pattern 5: Quatre lettres identiques minuscules (aaaa, bbbb, etc.)
    r'^([a-z])\1{3}$',
    
    # Pattern 6: Deux lettres identiques minuscules (aa, bb, cc, etc.)
    r'^([a-z])\1$',
    
    # Pattern 7: Trois lettres minuscules (pas forcément identiques) - abc, xyz, etc.
    r'^[a-z]{3}$',
]

print(f"\nAnalyse de {len(df)} lignes...")
print("Recherche des nomenclatures en MINUSCULES avec patterns de lettres...\n")

# Liste pour suivre les modifications
modifications = []
compteur_nc = 0

# Analyser chaque ligne
for index, row in df.iterrows():
    nomenclature = row[COLONNE_NOMENCLATURE]
    
    if pd.isna(nomenclature):
        continue
    
    nomenclature_str = str(nomenclature).strip()
    
    # Vérifier si c'est en minuscules (pour nos patterns)
    # On vérifie si c'est composé uniquement de lettres minuscules et/ou points
    if re.fullmatch(r'^[a-z\.]+$', nomenclature_str):
        pattern_trouve = None
        
        # Tester chaque pattern
        for i, pattern in enumerate(patterns_a_detecter_minuscules, 1):
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
                1: "deux_lettres_minuscules",
                2: "lettre_point_lettre_minuscules", 
                3: "trois_lettres_identiques_minuscules",
                4: "lettre_point_lettre_point_lettre_minuscules",
                5: "quatre_lettres_identiques_minuscules",
                6: "deux_lettres_identiques_minuscules",
                7: "trois_lettres_minuscules"
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

print(f"\n{compteur_nc} codes remplacés par 'NC' (minuscules).")

# Détecter aussi les cas MIXTES (majuscules ET minuscules)
print(f"\n=== VÉRIFICATION DES CAS MIXTES ===")
compteur_mixte = 0

for index, row in df.iterrows():
    nomenclature = row[COLONNE_NOMENCLATURE]
    
    if pd.isna(nomenclature):
        continue
    
    nomenclature_str = str(nomenclature).strip()
    
    # Patterns pour cas mixtes (lettres seulement, avec ou sans points)
    patterns_mixtes = [
        # Deux lettres (majuscules ou minuscules)
        r'^[A-Za-z]{2}$',
        
        # Lettre.Point.Lettre
        r'^[A-Za-z]\.[A-Za-z]$',
        
        # Trois lettres identiques
        r'^([A-Za-z])\1{2}$',
        
        # Lettre.Point.Lettre.Point.Lettre
        r'^[A-Za-z]\.[A-Za-z]\.[A-Za-z]$',
        
        # Deux lettres identiques
        r'^([A-Za-z])\1$',
    ]
    
    for pattern in patterns_mixtes:
        if re.fullmatch(pattern, nomenclature_str):
            compteur_mixte += 1
            print(f"  Ligne {index+2}: '{nomenclature_str}' → Pattern mixte détecté")
            break

print(f"Total patterns mixtes détectés: {compteur_mixte}")

# Statistiques détaillées pour minuscules
if modifications:
    print(f"\n=== STATISTIQUES DÉTAILLÉES (MINUSCULES) ===")
    
    # Par type de pattern
    print("\nDistribution par type de pattern :")
    types_comptes = {}
    for mod in modifications:
        type_pattern = mod['type']
        types_comptes[type_pattern] = types_comptes.get(type_pattern, 0) + 1
    
    for type_pattern, count in sorted(types_comptes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {type_pattern} : {count}")
    
    # Les nomenclatures les plus courantes en minuscules
    print("\nNomenclatures en minuscules les plus courantes :")
    noms_comptes = {}
    for mod in modifications:
        nom = mod['nomenclature']
        noms_comptes[nom] = noms_comptes.get(nom, 0) + 1
    
    for nom, count in sorted(noms_comptes.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  '{nom}' : {count}")

# Test de la détection
print(f"\n=== TEST DE DÉTECTION DES PATTERNS MINUSCULES ===")
test_cases = [
    ("ab", True, "deux lettres minuscules"),
    ("a.b", True, "lettre.point.lettre minuscules"),
    ("aaa", True, "trois lettres identiques minuscules"),
    ("a.b.c", True, "lettre.point.lettre.point.lettre minuscules"),
    ("aaaa", True, "quatre lettres identiques minuscules"),
    ("aa", True, "deux lettres identiques minuscules"),
    ("abc", True, "trois lettres minuscules"),
    ("xyz", True, "trois lettres minuscules"),
    ("a", False, "une lettre seulement"),
    ("abcd", False, "quatre lettres différentes"),
    ("A.B", False, "majuscules"),
    ("AB", False, "majuscules"),
    ("a.B.c", False, "mixte majuscule/minuscule"),
    ("1.2.3", False, "chiffres"),
    ("@#$", False, "symboles"),
    ("test", False, "mot complet"),
    ("manager", False, "mot complet"),
]

print("Exemples en minuscules qui seront modifiés :")
for texte, sera_modifie, description in test_cases:
    if sera_modifie:
        print(f"  ✓ '{texte}' → 'NC' ({description})")

print("\nExemples qui ne seront PAS modifiés :")
for texte, sera_modifie, description in test_cases:
    if not sera_modifie:
        print(f"  ✗ '{texte}' → inchangé ({description})")

# Vérifier les limites
print(f"\n=== CONDITIONS POUR LA DÉTECTION ===")
print("Conditions pour être détecté (minuscules) :")
print("  1. Doit être en MINUSCULES uniquement")
print("  2. Doit correspondre à un pattern spécifique")
print("  3. Pas de majuscules mélangées")
print("  4. Pas de chiffres ou symboles (sauf points séparateurs)")
print("  5. Pas de mots complets (comme 'test', 'agent', etc.)")

# Sauvegarder le résultat
dossier_source = os.path.dirname(chemin_fichier)
chemin_sauvegarde = os.path.join(dossier_source, 'CNPS_Patterns_Minuscules_NC.xlsx')

df.to_excel(chemin_sauvegarde, index=False)
print(f"\nFichier sauvegardé sous : {chemin_sauvegarde}")

# Sauvegarder un rapport détaillé
if modifications:
    rapport_path = os.path.join(dossier_source, 'Rapport_Patterns_Minuscules.csv')
    rapport_df = pd.DataFrame(modifications)
    rapport_df.to_csv(rapport_path, index=False, sep=';', encoding='utf-8')
    print(f"Rapport détaillé : {rapport_path}")
    
    # Afficher un aperçu du rapport
    print(f"\n=== APERÇU DU RAPPORT (premières 10 lignes) ===")
    print(rapport_df[['ligne', 'nomenclature', 'type', 'ancien_code']].head(10).to_string(index=False))

# Résumé final
print("\n" + "="*50)
print("RÉSUMÉ FINAL - MINUSCULES")
print("="*50)
print(f"Fichier analysé       : {os.path.basename(chemin_fichier)}")
print(f"Lignes totales        : {len(df)}")
print(f"Patterns minuscules   : {compteur_nc}")
print(f"Patterns mixtes       : {compteur_mixte}")
print(f"Total détections      : {compteur_nc + compteur_mixte}")
print(f"Fichier de sortie     : {os.path.basename(chemin_sauvegarde)}")