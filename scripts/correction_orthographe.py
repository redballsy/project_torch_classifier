import pandas as pd
from spellchecker import SpellChecker
import os
from pathlib import Path

# Initialiser le correcteur orthographique français
spell_fr = SpellChecker(language='fr')

# Fonction pour corriger l'orthographe
def correct_spelling_fr(text):
    if isinstance(text, str):
        words = text.split()
        corrected_words = []
        for word in words:
            try:
                # Obtenir la correction du mot
                corrected = spell_fr.correction(word)
                # Garder la correction si elle existe, sinon garder le mot original
                corrected_words.append(corrected if corrected is not None else word)
            except:
                # En cas d'erreur, garder le mot original
                corrected_words.append(word)
        return ' '.join(corrected_words)
    return text

# Définir le chemin du fichier avec le path complet
file_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\entrainer2_propre.xlsx"

# Vérifier si le fichier existe
if not os.path.exists(file_path):
    print(f"❌ Erreur : Le fichier '{file_path}' n'existe pas.")
    print("Vérifiez le chemin et assurez-vous que le fichier existe.")
else:
    try:
        # Charger le fichier Excel
        print(f"📂 Chargement du fichier : {file_path}")
        df = pd.read_excel(file_path)
        
        # Vérifier si la colonne "nomenclature" existe
        if 'nomenclature' not in df.columns:
            print("❌ Erreur : La colonne 'nomenclature' n'existe pas dans le fichier.")
            print(f"Colonnes disponibles : {list(df.columns)}")
        else:
            print(f"✅ Fichier chargé avec succès. {len(df)} lignes trouvées.")
            print("\n📊 Aperçu des données originales :")
            print(df.head())
            
            # Appliquer la correction sur la colonne "nomenclature"
            print("\n🔧 Correction de l'orthographe en cours...")
            print("Cette opération peut prendre quelques instants...")
            df["nomenclature_corrigee"] = df["nomenclature"].apply(correct_spelling_fr)
            
            # Créer le chemin de sortie dans le même dossier
            input_dir = os.path.dirname(file_path)
            input_filename = os.path.basename(file_path)
            output_filename = input_filename.replace('.xlsx', '_corrige.xlsx')
            output_path = os.path.join(input_dir, output_filename)
            
            # Sauvegarder dans un nouveau fichier Excel
            df.to_excel(output_path, index=False)
            
            print(f"\n✅ Correction terminée !")
            print(f"📁 Fichier sauvegardé sous : {output_path}")
            print(f"📏 Taille : {os.path.getsize(output_path) / 1024:.1f} Ko")
            
            print("\n📝 Aperçu des corrections :")
            print(df[["nomenclature", "nomenclature_corrigee"]].head(20))
            
            # Statistiques des corrections
            mask = df["nomenclature"] != df["nomenclature_corrigee"]
            corrected_count = mask.sum()
            print(f"\n📊 Statistiques des corrections :")
            print(f"- Total lignes : {len(df)}")
            print(f"- Lignes corrigées : {corrected_count}")
            print(f"- Pourcentage corrigé : {corrected_count/len(df)*100:.1f}%")
            
            # Afficher quelques exemples de corrections
            if corrected_count > 0:
                print("\n🔍 Exemples de corrections :")
                corrected_samples = df[mask].head(10)
                for idx, row in corrected_samples.iterrows():
                    print(f"  Avant : {row['nomenclature']}")
                    print(f"  Après : {row['nomenclature_corrigee']}")
                    print("  ---")
            
            # Optionnel : Sauvegarder aussi un CSV pour référence
            csv_path = output_path.replace('.xlsx', '.csv')
            df.to_csv(csv_path, index=False, sep=';', encoding='utf-8')
            print(f"\n📄 Version CSV également sauvegardée : {csv_path}")
            
    except PermissionError:
        print(f"❌ Erreur : Permission refusée pour accéder au fichier.")
        print("Assurez-vous que le fichier n'est pas ouvert dans un autre programme.")
    except Exception as e:
        print(f"❌ Erreur lors du traitement : {type(e).__name__}")
        print(f"Message d'erreur : {str(e)}")