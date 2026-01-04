import pandas as pd
import re
import os

# Chemin complet vers votre fichier Excel
chemin_fichier = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\CNPS_Special_Chars_NC.xlsx"

# Vérifier si le fichier existe
if not os.path.exists(chemin_fichier):
    print(f"ERREUR : Le fichier n'existe pas à l'emposition : {chemin_fichier}")
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

# Fonction pour nettoyer les bords de la chaîne
def nettoyer_bords(texte):
    if pd.isna(texte):
        return texte
    
    texte_str = str(texte)
    
    # Supprimer les caractères non-lettres au DÉBUT
    # Pattern: un ou plusieurs caractères qui ne sont pas des lettres (maj/min ou avec accents)
    # au début de la chaîne
    texte_str = re.sub(r'^[^a-zA-ZÀ-ÿ]+', '', texte_str)
    
    # Supprimer les caractères non-lettres à la FIN
    # Pattern: un ou plusieurs caractères qui ne sont pas des lettres à la fin
    texte_str = re.sub(r'[^a-zA-ZÀ-ÿ]+$', '', texte_str)
    
    # Garder les espaces intérieurs et les apostrophes intérieures
    return texte_str.strip()

# Fonction alternative plus détaillée pour le débogage
def nettoyer_bords_detail(texte):
    if pd.isna(texte):
        return texte, "NA", "NA"
    
    original = str(texte)
    nettoye = original
    
    # Caractères supprimés au début
    match_debut = re.match(r'^([^a-zA-ZÀ-ÿ]+)', nettoye)
    supprimes_debut = match_debut.group(1) if match_debut else ""
    nettoye = re.sub(r'^[^a-zA-ZÀ-ÿ]+', '', nettoye)
    
    # Caractères supprimés à la fin
    match_fin = re.search(r'([^a-zA-ZÀ-ÿ]+)$', nettoye)
    supprimes_fin = match_fin.group(1) if match_fin else ""
    nettoye = re.sub(r'[^a-zA-ZÀ-ÿ]+$', '', nettoye)
    
    nettoye_final = nettoye.strip()
    
    return nettoye_final, supprimes_debut, supprimes_fin

print(f"\nAnalyse de {len(df)} lignes...")
print("Nettoyage des caractères spéciaux/chiffres au début et à la fin des nomenclatures...\n")

# Liste pour suivre les modifications
modifications = []
compteur_modifs = 0

# Analyser et nettoyer chaque ligne
for index, row in df.iterrows():
    original = row[COLONNE_NOMENCLATURE]
    
    if pd.isna(original):
        continue
    
    original_str = str(original)
    
    # Nettoyer et obtenir les détails
    nettoye, supprimes_debut, supprimes_fin = nettoyer_bords_detail(original_str)
    
    # Vérifier si des modifications ont été faites
    if nettoye != original_str:
        compteur_modifs += 1
        
        # Enregistrer la modification
        modifications.append({
            'ligne': index + 2,
            'id': row.get(COLONNE_ID, 'N/A'),
            'code': row.get(COLONNE_CODE, 'N/A'),
            'original': original_str,
            'nettoye': nettoye,
            'supprime_debut': supprimes_debut,
            'supprime_fin': supprimes_fin
        })
        
        # Mettre à jour le DataFrame
        df.at[index, COLONNE_NOMENCLATURE] = nettoye
        
        # Afficher les cas intéressants
        if supprimes_debut or supprimes_fin:
            print(f"Ligne {index+2}: '{original_str}' → '{nettoye}'")
            if supprimes_debut:
                print(f"  Supprimé au début: '{supprimes_debut}'")
            if supprimes_fin:
                print(f"  Supprimé à la fin: '{supprimes_fin}'")
            print()

print(f"\n{compteur_modifs} nomenclatures nettoyées.")

# Afficher des statistiques
if modifications:
    print("\n=== STATISTIQUES DES MODIFICATIONS ===")
    
    # Catégoriser les types de caractères supprimés
    types_supprimes = {
        'chiffres': 0,
        'ponctuation': 0,
        'symboles': 0,
        'espaces': 0,
        'melanges': 0
    }
    
    for mod in modifications:
        # Analyser le début
        if mod['supprime_debut']:
            dept = mod['supprime_debut']
            if dept.isdigit():
                types_supprimes['chiffres'] += 1
            elif re.match(r'^[.,!?;:]+$', dept):
                types_supprimes['ponctuation'] += 1
            elif re.match(r'^[@#$%^&*()]+$', dept):
                types_supprimes['symboles'] += 1
            elif dept.isspace():
                types_supprimes['espaces'] += 1
            else:
                types_supprimes['melanges'] += 1
        
        # Analyser la fin
        if mod['supprime_fin']:
            fin = mod['supprime_fin']
            if fin.isdigit():
                types_supprimes['chiffres'] += 1
            elif re.match(r'^[.,!?;:]+$', fin):
                types_supprimes['ponctuation'] += 1
            elif re.match(r'^[@#$%^&*()]+$', fin):
                types_supprimes['symboles'] += 1
            elif fin.isspace():
                types_supprimes['espaces'] += 1
            else:
                types_supprimes['melanges'] += 1
    
    print("\nTypes de caractères supprimés:")
    for type_char, count in types_supprimes.items():
        if count > 0:
            print(f"  {type_char}: {count} fois")
    
    # Exemples les plus courants
    print("\n=== EXEMPLES LES PLUS COURANTS ===")
    
    # Regrouper par type de nettoyage
    exemples_par_type = {}
    for mod in modifications[:20]:  # Prendre les 20 premiers pour l'exemple
        type_clean = ""
        if mod['supprime_debut'] and mod['supprime_fin']:
            type_clean = "debut_et_fin"
        elif mod['supprime_debut']:
            type_clean = "debut_seulement"
        elif mod['supprime_fin']:
            type_clean = "fin_seulement"
        
        if type_clean not in exemples_par_type:
            exemples_par_type[type_clean] = []
        exemples_par_type[type_clean].append(mod)
    
    for type_clean, exemples in exemples_par_type.items():
        print(f"\n{len(exemples)} exemples de nettoyage {type_clean}:")
        for ex in exemples[:5]:  # Limiter à 5 par type
            print(f"  '{ex['original']}' → '{ex['nettoye']}'")

# Sauvegarder le résultat
dossier_source = os.path.dirname(chemin_fichier)
chemin_sauvegarde = os.path.join(dossier_source, 'CNPS_Nomenclatures_Nettoyees.xlsx')

df.to_excel(chemin_sauvegarde, index=False)
print(f"\nFichier sauvegardé sous : {chemin_sauvegarde}")

# Sauvegarder un rapport des modifications
if modifications:
    rapport_path = os.path.join(dossier_source, 'Rapport_Nettoyage_Bords.csv')
    df_modifications = pd.DataFrame(modifications)
    df_modifications.to_csv(rapport_path, index=False, sep=';', encoding='utf-8')
    print(f"Rapport des modifications : {rapport_path}")

# Afficher des exemples spécifiques
print("\n=== EXEMPLES DÉTAILLÉS ===")
exemples_test = [
    ".assistant",
    "8 assistant 888", 
    "!!!Manager!!!",
    "123Ingénieur456",
    "@@@Technicien@@@",
    "  Agent  ",  # espaces
    "1.2.3.Comptable",
    "***Directeur***",
    "Test-123-",  # tiret à la fin
    "##CHEF##"
]

print("Exemples de nettoyage (test):")
for ex in exemples_test:
    nettoye, debut, fin = nettoyer_bords_detail(ex)
    print(f"  '{ex}' → '{nettoye}'")
    if debut:
        print(f"    Début supprimé: '{debut}'")
    if fin:
        print(f"    Fin supprimée: '{fin}'")

print("\n=== RÉSUMÉ FINAL ===")
print(f"Lignes totales: {len(df)}")
print(f"Nomenclatures nettoyées: {compteur_modifs}")
print(f"Pourcentage modifié: {compteur_modifs/len(df)*100:.1f}%")