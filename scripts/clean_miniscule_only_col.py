import pandas as pd
import os

def add_lowercase_column():
    """
    Read the Excel file, add a new column with lowercase text,
    and save the result as topredict.xlsx in the same location.
    """
    try:
        # Define the input file path
        input_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\nomenclatureminiscule.xlsx"
        
        # Define the output file path (same directory with new name)
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, "topredict.xlsx")
        
        print(f"Reading file from: {input_path}")
        
        # Check if file exists
        if not os.path.exists(input_path):
            print(f"Error: File not found at {input_path}")
            return None
        
        # Read the Excel file
        df = pd.read_excel(input_path)
        
        print(f"Original dataframe shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")
        
        # Assuming the column with nomenclature is the first column
        # Let's find the correct column name
        if len(df.columns) > 0:
            text_column = df.columns[0]  # Get the first column name
            print(f"Processing column: '{text_column}'")
            
            # Create a new column with lowercase text
            new_column_name = f"{text_column}_lowercase"
            df[new_column_name] = df[text_column].apply(
                lambda x: str(x).lower() if pd.notna(x) else ""
            )
            
            print(f"Added new column: '{new_column_name}'")
            print(f"New dataframe shape: {df.shape}")
            
            # Save to new file
            df.to_excel(output_path, index=False)
            
            print(f"\nProcessing complete!")
            print(f"Input file: {input_path}")
            print(f"Output file: {output_path}")
            
            # Show sample of the data
            print("\nSample of first 5 rows with new lowercase column:")
            for i in range(min(5, len(df))):
                original = str(df[text_column].iloc[i]) if pd.notna(df[text_column].iloc[i]) else ""
                lowercase = str(df[new_column_name].iloc[i]) if pd.notna(df[new_column_name].iloc[i]) else ""
                print(f"Row {i}:")
                print(f"  Original: '{original}'")
                print(f"  Lowercase: '{lowercase}'")
                print()
            
            return df
        else:
            print("Error: No columns found in the Excel file")
            return None
    
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None

# Alternative function that keeps the same column structure but converts existing column to lowercase
def convert_to_lowercase():
    """
    Alternative: Convert the existing column to lowercase instead of adding a new column.
    """
    try:
        input_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\nomenclatureminiscule.xlsx"
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, "topredict.xlsx")
        
        print(f"Reading from: {input_path}")
        
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            return None
        
        df = pd.read_excel(input_path)
        
        # Get the column name (should be 'nomenclature' based on your data)
        if 'nomenclature' in df.columns:
            text_column = 'nomenclature'
        else:
            text_column = df.columns[0]
        
        print(f"Column to process: '{text_column}'")
        print(f"Rows to process: {len(df)}")
        
        # Convert the existing column to lowercase
        df[text_column] = df[text_column].apply(
            lambda x: str(x).lower() if pd.notna(x) else str(x)
        )
        
        # Save the result
        df.to_excel(output_path, index=False)
        
        print(f"\nSuccess! File saved to: {output_path}")
        print("\nSample of converted data:")
        print(df.head(10))
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Simple version - just add lowercase column
def simple_add_lowercase():
    """
    Simple version: Add a new column with lowercase text.
    """
    input_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\nomenclatureminiscule.xlsx"
    output_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\topredict.xlsx"
    
    print(f"Processing file: {input_path}")
    
    # Read the Excel file
    df = pd.read_excel(input_path)
    
    # Get the column name (first column)
    original_col = df.columns[0]
    print(f"Original column name: '{original_col}'")
    
    # Create a new column with lowercase text
    df['nomenclature_lowercase'] = df[original_col].apply(
        lambda x: str(x).lower() if pd.notna(x) else ''
    )
    
    # Save to new file
    df.to_excel(output_path, index=False)
    
    print(f"File saved to: {output_path}")
    print(f"Number of rows processed: {len(df)}")
    
    return df

# Run the processing
if __name__ == "__main__":
    print("Adding lowercase column to nomenclature data...")
    
    # Choose which function to use:
    # Option 1: Add a new column with lowercase text
    result = simple_add_lowercase()
    
    # Option 2: Convert existing column to lowercase
    # result = convert_to_lowercase()
    
    # Option 3: Detailed version with new column
    # result = add_lowercase_column()
    
    if result is not None:
        print("\n✓ Success! File has been processed.")
        print("\nSummary:")
        print(f"  - Input file: nomenclatureminiscule.xlsx")
        print(f"  - Output file: topredict.xlsx")
        print(f"  - Rows processed: {len(result)}")
        print(f"  - Columns in output: {list(result.columns)}")
    else:
        print("\n✗ Failed to process the file.")