import pandas as pd
import os

# Chemin du fichier original
file_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\corpus_complet.xlsx"

# Chemin du nouveau fichier
new_file_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\corpus_complet_minuscule.xlsx"

print(f"📂 Chargement du fichier : {file_path}")

# Charger le fichier Excel
try:
    df = pd.read_excel(file_path)
    print(f"✅ Fichier chargé avec succès")
    print(f"   Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"   Colonnes : {', '.join(df.columns.tolist())}")
except FileNotFoundError:
    print(f"❌ Erreur : Fichier non trouvé à l'emplacement spécifié")
    print(f"   Vérifiez le chemin : {file_path}")
    exit()
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")
    exit()

# Afficher un aperçu avant transformation
print("\n📋 Aperçu avant transformation :")
print(df[['code', 'nomenclature', 'variante']].head(3))
print("...")

# Vérifier que les colonnes existent
required_cols = ['nomenclature', 'variante']
for col in required_cols:
    if col not in df.columns:
        print(f"❌ Erreur : Colonne '{col}' non trouvée dans le fichier")
        print(f"   Colonnes disponibles : {df.columns.tolist()}")
        exit()

print("\n🔧 Transformation en minuscules...")

# Sauvegarder les valeurs originales pour vérification
df['nomenclature_original'] = df['nomenclature']
df['variante_original'] = df['variante']

# Appliquer la transformation en minuscules
df['nomenclature'] = df['nomenclature'].str.lower()
df['variante'] = df['variante'].str.lower()

print("✅ Transformation appliquée")

# Afficher un aperçu après transformation
print("\n📋 Aperçu après transformation :")
print(df[['code', 'nomenclature', 'variante']].head(3))
print("...")

# Afficher quelques exemples de transformation
print("\n🎯 Exemples de transformation :")
print("-" * 70)

for i in range(min(3, len(df))):
    code = df.loc[i, 'code']
    nom_orig = df.loc[i, 'nomenclature_original']
    nom_new = df.loc[i, 'nomenclature']
    var_orig = df.loc[i, 'variante_original']
    var_new = df.loc[i, 'variante']
    
    print(f"Code {code}:")
    print(f"  Nomenclature : '{nom_orig}' → '{nom_new}'")
    print(f"  Variante     : '{var_orig}' → '{var_new}'")
    print()

# Sauvegarder le nouveau fichier
print(f"💾 Sauvegarde du nouveau fichier : {new_file_path}")

try:
    # Exporter vers le nouveau fichier
    df.to_excel(new_file_path, index=False)
    
    # Vérifier que le fichier a été créé
    if os.path.exists(new_file_path):
        file_size = os.path.getsize(new_file_path) / 1024 / 1024  # en MB
        print(f"✅ Fichier sauvegardé avec succès")
        print(f"   Taille : {file_size:.2f} MB")
        print(f"   Lignes : {len(df)}")
        print(f"   Colonnes : {len(df.columns)}")
    else:
        print(f"❌ Erreur : Le fichier n'a pas été créé")
        
except Exception as e:
    print(f"❌ Erreur lors de la sauvegarde : {e}")
    exit()

print("\n" + "="*70)
print("📊 RÉSUMÉ DE LA TRANSFORMATION")
print("="*70)
print(f"Fichier original conservé : {file_path}")
print(f"Nouveau fichier créé     : {new_file_path}")
print(f"Colonnes transformées    : nomenclature, variante")
print(f"Colonne non modifiée     : code")
print(f"Fichier original préservé ✓")
print("="*70)

# Vérification finale
print("\n🔍 Vérification rapide :")
print(f"1. Fichier original existe : {'✅' if os.path.exists(file_path) else '❌'}")
print(f"2. Nouveau fichier existe  : {'✅' if os.path.exists(new_file_path) else '❌'}")
print(f"3. Taille différente       : {'✅' if os.path.exists(file_path) and os.path.exists(new_file_path) and os.path.getsize(file_path) != os.path.getsize(new_file_path) else '⚠️'}")