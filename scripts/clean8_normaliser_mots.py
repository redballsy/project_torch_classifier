import pandas as pd
import unicodedata
import re

def nettoyer_texte(texte):
    if not isinstance(texte, str):
        return ""
    
    # 1. Mise en minuscule
    texte = texte.lower()
    
    # 2. Suppression des accents (Normalisation Unicode)
    # Exemple: "Ingénieur" -> "Ingenieur"
    texte = unicodedata.normalize('NFD', texte)
    texte = "".join([c for c in texte if unicodedata.category(c) != 'Mn'])
    
    # 3. Suppression de la ponctuation et des caractères spéciaux
    # On ne garde que les lettres et les espaces
    texte = re.sub(r'[^a-z\s]', ' ', texte)
    
    # 4. Suppression des espaces doubles
    texte = " ".join(texte.split())
    
    return texte

# --- CONFIGURATION DES CHEMINS ---
PATH_ENTREE = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\entrainer2.xlsx"
PATH_SORTIE = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\entrainer2_propre.xlsx"

print("⏳ Lecture du fichier...")
df = pd.read_excel(PATH_ENTREE)

if 'nomenclature' in df.columns:
    print("🧹 Nettoyage de la colonne 'nomenclature'...")
    # Applique la fonction de nettoyage
    df['nomenclature'] = df['nomenclature'].apply(nettoyer_texte)
    
    # Optionnel : Supprimer les lignes qui se retrouveraient vides après nettoyage
    df = df[df['nomenclature'] != ""]
    
    print(f"💾 Sauvegarde du fichier nettoyé vers : {PATH_SORTIE}")
    df.to_excel(PATH_SORTIE, index=False)
    print("✅ Terminé !")
else:
    print("❌ Erreur : La colonne 'nomenclature' est introuvable dans le fichier.")