import pandas as pd
import re
from docx import Document
import os

# Chemins des fichiers
excel_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\variante.xlsx"
word_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\ctips.docx"
output_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\variante_complete.xlsx"

# Vérifier que les fichiers existent
if not os.path.exists(excel_path):
    print(f"ERREUR : Fichier Excel introuvable : {excel_path}")
    exit()
if not os.path.exists(word_path):
    print(f"ERREUR : Fichier Word introuvable : {word_path}")
    exit()

print("Lecture du fichier Excel...")
# Lire le fichier Excel
df_excel = pd.read_excel(excel_path)
print(f"Fichier Excel lu : {len(df_excel)} lignes")

print("Lecture du fichier Word...")
# Lire le fichier Word
try:
    doc = Document(word_path)
    text_content = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_content.append(paragraph.text.strip())
    print(f"Fichier Word lu : {len(text_content)} paragraphes non vides")
except Exception as e:
    print(f"ERREUR lors de la lecture du fichier Word : {e}")
    exit()

# Parser le contenu du Word
code_nomenclature_dict = {}
current_code = None
current_nomenclature = None
variantes_list = []

print("Parsing du contenu Word...")
for line in text_content:
    # Chercher un code au début de la ligne (4 chiffres)
    code_match = re.match(r'^(\d{4})\s+(.+)$', line)
    
    if code_match:
        # Si on avait déjà un code en cours, sauvegarder ses données
        if current_code and current_nomenclature and variantes_list:
            code_nomenclature_dict[current_code] = {
                'nomenclature': current_nomenclature,
                'variantes': variantes_list
            }
        
        # Nouveau code
        current_code = code_match.group(1)
        current_nomenclature = code_match.group(2)
        variantes_list = []
    
    # Lignes avec variantes (ne commencent pas par un code)
    elif current_code and line and not re.match(r'^\d{4}', line):
        # Nettoyer la ligne et séparer les variantes par virgule
        line_clean = line.replace('\n', ' ').replace('  ', ' ')
        # Supprimer les points à la fin
        line_clean = line_clean.rstrip('.')
        variantes = [v.strip() for v in line_clean.split(',') if v.strip()]
        variantes_list.extend(variantes)

# Ajouter le dernier code traité
if current_code and current_nomenclature and variantes_list:
    code_nomenclature_dict[current_code] = {
        'nomenclature': current_nomenclature,
        'variantes': variantes_list
    }

print(f"Codes trouvés dans le Word : {len(code_nomenclature_dict)}")

# Créer un nouveau DataFrame avec les données fusionnées
new_rows = []

print("Fusion des données Excel et Word...")
for _, row in df_excel.iterrows():
    code = str(row['code']).strip()
    
    if code in code_nomenclature_dict:
        nomenclature = code_nomenclature_dict[code]['nomenclature']
        variantes = code_nomenclature_dict[code]['variantes']
        
        for variante in variantes:
            new_rows.append({
                'code': code,
                'nomenclature': nomenclature,
                'variante': variante
            })
    else:
        # Si le code n'est pas trouvé dans le Word, garder la ligne telle quelle
        new_rows.append({
            'code': code,
            'nomenclature': row['nomenclature'],
            'variante': ''
        })

# Créer le nouveau DataFrame
df_new = pd.DataFrame(new_rows)

# Sauvegarder dans un nouveau fichier Excel
print(f"Sauvegarde dans {output_path}...")
df_new.to_excel(output_path, index=False)

print("\n" + "="*50)
print("TRAITEMENT TERMINÉ !")
print("="*50)
print(f"Fichier généré : {output_path}")
print(f"Nombre total de lignes : {len(df_new):,}")
print(f"Nombre de codes traités : {len(code_nomenclature_dict)}")
print(f"Codes trouvés dans le Word mais pas dans l'Excel :")
missing_in_excel = set(code_nomenclature_dict.keys()) - set(df_excel['code'].astype(str).str.strip())
for code in sorted(missing_in_excel):
    print(f"  - {code}")
print(f"Codes dans l'Excel mais sans variantes dans le Word :")
missing_variantes = set(df_excel['code'].astype(str).str.strip()) - set(code_nomenclature_dict.keys())
for code in sorted(missing_variantes):
    print(f"  - {code}")
print("="*50)

# Afficher un échantillon des données
print("\nÉchantillon des données générées :")
print(df_new.head(15).to_string())