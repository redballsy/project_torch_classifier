import pandas as pd
import unicodedata
import os

def remove_accents(input_str):
    if not isinstance(input_str, str):
        return input_str
    # Normalisation NFKD pour séparer les caractères de leurs accents
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# Configuration des chemins
input_file = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\CITP_08.xlsx"
output_file = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\CITP_081.xlsx"

if os.path.exists(input_file):
    print(f"📂 Lecture de : {input_file}")
    df = pd.read_excel(input_file)

    print("🔧 Traitement de la colonne 'nomenclature'...")
    
    # Conversion en string, mise en minuscule et suppression des accents
    df['nomenclature'] = df['nomenclature'].astype(str).str.lower()
    df['nomenclature'] = df['nomenclature'].apply(remove_accents)
    
    # Nettoyage des espaces blancs superflus
    df['nomenclature'] = df['nomenclature'].str.strip()

    # Sauvegarde
    try:
        df.to_excel(output_file, index=False)
        print(f"✅ Fichier sauvegardé avec succès : {output_file}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {e}")

    print("\n📊 Aperçu des 5 premières lignes :")
    print(df[['nomenclature']].head())
else:
    print("❌ Le fichier source n'a pas été trouvé. Vérifiez le chemin.")