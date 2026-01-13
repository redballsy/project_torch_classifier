import pandas as pd

# Chemins des fichiers
file_to_predict = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\topredict.xlsx"
file_reference = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\CITP_08.xlsx"
# On enregistre le résultat dans un nouveau fichier par sécurité
output_cleaned = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\topredict_nettoye.xlsx"

def nettoyer_fichier():
    try:
        # 1. Chargement des données
        df_topredict = pd.read_excel(file_to_predict)
        df_reference = pd.read_excel(file_reference)

        # Remplacez 'code' par le nom exact de votre colonne
        col_name = 'code'

        # 2. On crée une liste des codes valides (référence)
        codes_valides = set(df_reference[col_name].unique())

        # 3. On filtre : on garde uniquement si le code EXISTE dans codes_valides
        df_filtre = df_topredict[df_topredict[col_name].isin(codes_valides)]

        # 4. Sauvegarde
        df_filtre.to_excel(output_cleaned, index=False)

        # Calcul pour information
        nb_supprimes = len(df_topredict) - len(df_filtre)
        print(f"Nettoyage terminé !")
        print(f"Lignes supprimées (car absentes de CITP_08) : {nb_supprimes}")
        print(f"Fichier enregistré ici : {output_cleaned}")

    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    nettoyer_fichier()