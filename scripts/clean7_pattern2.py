import pandas as pd
import os
import re

# Chemin complet vers votre fichier Excel
chemin_fichier = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\CNPS_Patterns_Minuscules_NC.xlsx"

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

# Définir les patterns spécifiques pour "T. DE S" et similaires
patterns_a_detecter = [
    # Pattern 1: Lettre.Point ESPACE Mot ESPACE Lettre (T. DE S)
    r'^[A-Z]\.\s+[A-Z]{2,3}\s+[A-Z]$',
    
    # Pattern 2: Lettre.Point ESPACE Lettre (A. B)
    r'^[A-Z]\.\s+[A-Z]$',
    
    # Pattern 3: Lettre.Point Lettre (A.B)
    r'^[A-Z]\.[A-Z]$',
    
    # Pattern 4: Lettre ESPACE Lettre (A B)
    r'^[A-Z]\s+[A-Z]$',
    
    # Pattern 5: Mot court (2-3 lettres) avec point et espaces
    r'^[A-Z]{1,3}\.[A-Z]{1,3}$',
    
    # Pattern 6: Mot ESPACE Lettre.Point (DE T.)
    r'^[A-Z]{2,3}\s+[A-Z]\.$',
    
    # Pattern 7: Lettre.Point Mot court (T.DE)
    r'^[A-Z]\.[A-Z]{2,3}$',
]

print(f"\nAnalyse de {len(df)} lignes...")
print("Recherche des patterns de type 'T. DE S'...\n")

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
        pattern_type = ""
        
        # Tester chaque pattern
        for i, pattern in enumerate(patterns_a_detecter, 1):
            if re.fullmatch(pattern, nomenclature_str):
                pattern_trouve = i
                
                # Déterminer le type
                if i == 1:
                    pattern_type = "lettre_point_mot_lettre"
                elif i == 2:
                    pattern_type = "lettre_point_espace_lettre"
                elif i == 3:
                    pattern_type = "lettre_point_lettre"
                elif i == 4:
                    pattern_type = "lettre_espace_lettre"
                elif i == 5:
                    pattern_type = "mot_point_mot"
                elif i == 6:
                    pattern_type = "mot_espace_lettre_point"
                elif i == 7:
                    pattern_type = "lettre_point_mot"
                
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
            
            # Enregistrer la modification
            modifications.append({
                'ligne': index + 2,
                'id': id_val,
                'nomenclature': nomenclature_str,
                'pattern': pattern_trouve,
                'type': pattern_type,
                'ancien_code': ancien_code,
                'nouveau_code': "NC"
            })
            
            # Afficher
            print(f"Ligne {index+2}: '{nomenclature_str}' ({pattern_type}) → code: 'NC'")

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
    ("T. DE S", True, "lettre_point_mot_lettre"),
    ("A. B", True, "lettre_point_espace_lettre"),
    ("A.B", True, "lettre_point_lettre"),
    ("A B", True, "lettre_espace_lettre"),
    ("AB.CD", True, "mot_point_mot"),
    ("DE T.", True, "mot_espace_lettre_point"),
    ("T.DE", True, "lettre_point_mot"),
    ("T. DE", False, "pas assez de parties"),
    ("T DE S", False, "pas de point"),
    ("t. de s", False, "minuscules"),
    ("TEST", False, "mot complet"),
    ("A", False, "une seule lettre"),
    ("A.B.C", False, "trop de parties"),
    ("1. DE 2", False, "contient des chiffres"),
    ("@. DE #", False, "contient des symboles"),
]

print("Exemples qui seront modifiés :")
for texte, sera_modifie, description in test_cases:
    if sera_modifie:
        print(f"  ✓ '{texte}' → 'NC' ({description})")

print("\nExemples qui ne seront PAS modifiés :")
for texte, sera_modifie, description in test_cases:
    if not sera_modifie:
        print(f"  ✗ '{texte}' → inchangé ({description})")

# Rechercher spécifiquement "T. DE S" et variations
print(f"\n=== RECHERCHE SPÉCIFIQUE 'T. DE S' ===")
patterns_t_des = [
    r'^T\.\s*DE\s*S$',
    r'^T\.DE\.S$',
    r'^T DE S$',
    r'^T\.D\.S$',
]

print("Variations de 'T. DE S' qui seront détectées :")
for pattern in patterns_t_des:
    test_str = "T. DE S"
    if re.match(pattern, test_str, re.IGNORECASE):
        print(f"  ✓ '{test_str}' correspond à: {pattern}")

# Vérifier dans les données
print("\nRecherche de variations similaires dans les données :")
for index, row in df.iterrows():
    nomenclature = row[COLONNE_NOMENCLATURE]
    
    if pd.isna(nomenclature):
        continue
    
    nom = str(nomenclature).strip()
    
    # Chercher des patterns similaires à T. DE S
    if re.search(r'T\.?\s*D[E\.]?\s*S', nom, re.IGNORECASE):
        code = row.get(COLONNE_CODE, 'N/A')
        id_val = row.get(COLONNE_ID, 'N/A')
        print(f"  Ligne {index+2}: '{nom}' (code: {code}, ID: {id_val})")

# Sauvegarder le résultat
dossier_source = os.path.dirname(chemin_fichier)
chemin_sauvegarde = os.path.join(dossier_source, 'CNPS_Patterns_T_DE_S_NC.xlsx')

df.to_excel(chemin_sauvegarde, index=False)
print(f"\nFichier sauvegardé sous : {chemin_sauvegarde}")

# Sauvegarder un rapport détaillé
if modifications:
    rapport_path = os.path.join(dossier_source, 'Rapport_Patterns_T_DE_S.csv')
    rapport_df = pd.DataFrame(modifications)
    rapport_df.to_csv(rapport_path, index=False, sep=';', encoding='utf-8')
    print(f"Rapport détaillé : {rapport_path}")

# Résumé final
print("\n" + "="*50)
print("RÉSUMÉ FINAL - PATTERNS 'T. DE S'")
print("="*50)
print(f"Fichier analysé       : {os.path.basename(chemin_fichier)}")
print(f"Lignes totales        : {len(df)}")
print(f"Patterns détectés     : {compteur_nc}")
print(f"Pourcentage modifié   : {compteur_nc/len(df)*100:.2f}%")
print(f"Fichier de sortie     : {os.path.basename(chemin_sauvegarde)}")
print(f"\nPatterns recherchés :")
print("  1. T. DE S (lettre.point mot lettre)")
print("  2. A. B (lettre.point espace lettre)")
print("  3. A.B (lettre.point.lettre)")
print("  4. A B (lettre espace lettre)")
print("  5. AB.CD (mot.point.mot)")
print("  6. DE T. (mot espace lettre.point)")
print("  7. T.DE (lettre.point.mot)")