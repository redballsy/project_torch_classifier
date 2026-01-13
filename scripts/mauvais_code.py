import pandas as pd

# Chemins des fichiers
file_to_predict = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\topredict.xlsx"
file_reference = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\CITP_08.xlsx"
output_file = "codes_manquants.xlsx"

def trouver_codes_uniques():
    try:
        # 1. Chargement des fichiers
        # On suppose que la colonne contenant les codes s'appelle 'code'
        df_topredict = pd.read_excel(file_to_predict)
        df_reference = pd.read_excel(file_reference)

        # Identification de la colonne (ajustez 'code' par le vrai nom de la colonne)
        col_name = 'code' 

        # 2. Extraction des listes de codes
        # On transforme la référence en "set" pour une recherche beaucoup plus rapide
        codes_reference = set(df_reference[col_name].unique())

        # 3. Filtrage : on garde les lignes de 'topredict' dont le code n'est PAS dans la référence
        # L'ordre original est conservé car on filtre directement le DataFrame initial
        manquants = df_topredict[~df_topredict[col_name].isin(codes_reference)]

        # 4. Exportation vers un nouveau fichier Excel
        manquants.to_excel(output_file, index=False)
        
        print(f"Succès ! {len(manquants)} codes trouvés. Fichier enregistré sous : {output_file}")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    trouver_codes_uniques()