import pandas as pd
import re
from spellchecker import SpellChecker
from tqdm import tqdm

# --- CONFIGURATION DES CHEMINS ---
FILE_INPUT = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\CNPS_A_Nettoyer.xlsx"
FILE_OUTPUT = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\CNPS_Nettoyer2_Final.xlsx"
COLONNE_A_CORRIGER = 'nomenclature' # Vérifiez si le nom exact est bien celui-là

# --- INITIALISATION ---
spell_fr = SpellChecker(language='fr')
spell_en = SpellChecker(language='en')
tqdm.pandas() # Active le support tqdm pour pandas

def correct_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return text
    
    # Séparation en mots tout en gardant une trace de la casse
    words = str(text).split()
    corrected_words = []
    
    for word in words:
        # Nettoyage pour vérification (enlève ponctuation collée)
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        
        # On ne corrige que si le mot est composé de lettres et n'est pas dans les dictionnaires
        if clean_word.isalpha() and clean_word not in spell_fr and clean_word not in spell_en:
            # Tentative de correction
            correction = spell_fr.correction(clean_word) or spell_en.correction(clean_word)
            corrected_words.append(correction if correction else word)
        else:
            corrected_words.append(word)
            
    return " ".join(corrected_words)

def main():
    print(f"Chargement du fichier : {FILE_INPUT}...")
    try:
        df = pd.read_excel(FILE_INPUT)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return

    # --- ÉTAPE 1 : Nettoyage des lignes inutiles ---
    print("Étape 1 : Suppression des lignes sans texte alphanumérique...")
    initial_count = len(df)
    # On garde les lignes qui contiennent au moins une lettre ou un chiffre
    df = df[df[COLONNE_A_CORRIGER].astype(str).str.contains(r'[a-zA-Z0-9]', na=False)].copy()
    print(f"Lignes supprimées : {initial_count - len(df)}")

    # --- ÉTAPE 2 : Correction orthographique ---
    print("Étape 2 : Correction orthographique (Français/Anglais)...")
    # On utilise progress_apply pour voir la barre de progression
    df[COLONNE_A_CORRIGER] = df[COLONNE_A_CORRIGER].progress_apply(correct_text)

    # --- ÉTAPE 3 : Exportation ---
    print(f"Étape 3 : Sauvegarde vers {FILE_OUTPUT}...")
    df.to_excel(FILE_OUTPUT, index=False)
    print("Opération terminée avec succès !")

if __name__ == "__main__":
    main()