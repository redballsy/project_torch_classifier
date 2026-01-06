import pandas as pd
import re

# Use 'r' to fix the Unicode error
file_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\entrainer2_propre.xlsx"

# 1. Load the file
df = pd.read_excel(file_path)

# 2. Define the Regex Pattern
# This looks for: 1-3 letters, followed by more groups of 1-3 letters (a, a b c, aaa, sdo aa)
nc_pattern = r'^([a-zA-Z]{1,3})(\s[a-zA-Z]{1,3})*$'

# 3. Identify matches and update the 'code' column
# We use .astype(str) to avoid errors with empty cells
mask = df['nomenclature'].astype(str).str.strip().str.match(nc_pattern)
df.loc[mask, 'code'] = 'NC'

# 4. Save to a new file (so you don't overwrite your original by mistake)
output_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\entrainer2_updated.xlsx"
df.to_excel(output_path, index=False)

print(f"Success! {mask.sum()} rows were updated to 'NC'.")