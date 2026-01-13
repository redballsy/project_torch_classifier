import pandas as pd
import os

def process_codes():
    """
    Read the Excel file from the specified path, remove the last 2 digits 
    from 6-digit codes, and save the result as codeuniqnew.xlsx in the same location.
    """
    try:
        # Get the current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Script directory: {script_dir}")
        
        # Define the input file path RELATIVE TO THE SCRIPT
        # Assuming your script is in the project root or scripts folder
        # Adjust the path based on your actual structure
        
        # Option 1: If script is in project root
        input_path = os.path.join(script_dir, "torchTestClassifiers", "data", "entrainer", "codeuniq2.xlsx")
        
        # Option 2: If script is in scripts folder (like your case)
        if "scripts" in script_dir:
            # Go up one level from scripts folder to project root
            project_root = os.path.dirname(script_dir)
            input_path = os.path.join(project_root, "torchTestClassifiers", "data", "entrainer", "codeuniq2.xlsx")
        
        print(f"Looking for input file at: {input_path}")
        
        # Check if the input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file not found at {input_path}")
            print("\nTrying alternative paths...")
            
            # Try absolute path you mentioned in command prompt
            alt_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\codeuniq2.xlsx"
            print(f"Trying: {alt_path}")
            
            if os.path.exists(alt_path):
                input_path = alt_path
                print(f"Found file at alternative path!")
            else:
                print("File not found. Please check the file path.")
                print("Current working directory:", os.getcwd())
                print("\nFiles in current directory:", os.listdir(os.getcwd()))
                return None
        
        print(f"Reading file from: {input_path}")
        
        # Read the Excel file
        df = pd.read_excel(input_path)
        
        # Get the first column name
        code_col = df.columns[0]
        print(f"Column name: '{code_col}'")
        print(f"Number of rows: {len(df)}")
        
        # Process each code - remove last 2 digits if it's a 6-digit number
        df[code_col] = df[code_col].apply(
            lambda x: str(x)[:-2] if pd.notna(x) and len(str(x).strip()) == 6 and str(x).strip().isdigit() else str(x).strip()
        )
        
        # Define the output file path (same directory as input)
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, "codeuniqnew.xlsx")
        
        # Save to new file
        df.to_excel(output_path, index=False)
        
        print("\nProcessing complete!")
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        
        # Show sample of processed data
        print("\nSample of first 5 processed codes:")
        original_df = pd.read_excel(input_path)
        for i in range(min(5, len(df))):
            original = str(original_df[code_col].iloc[i]).strip() if pd.notna(original_df[code_col].iloc[i]) else ''
            processed = str(df[code_col].iloc[i]).strip() if pd.notna(df[code_col].iloc[i]) else ''
            print(f"  Row {i}: '{original}' -> '{processed}'")
        
        return df
    
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None

# Or use this simpler version with hardcoded absolute path:
def process_codes_simple():
    """Simpler version with absolute path"""
    try:
        # Use the absolute path you mentioned
        input_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\codeuniq2.xlsx"
        output_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\codeuniqnew.xlsx"
        
        print(f"Reading from: {input_path}")
        
        # Check if file exists
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            return None
        
        # Read the Excel file
        df = pd.read_excel(input_path)
        
        # Get the column name
        code_col = df.columns[0]
        
        # Process the codes
        df[code_col] = df[code_col].apply(
            lambda x: str(x)[:-2] if pd.notna(x) and len(str(x).strip()) == 6 and str(x).strip().isdigit() else str(x).strip()
        )
        
        # Save the result
        df.to_excel(output_path, index=False)
        
        print(f"\nSuccess! Saved to: {output_path}")
        print(f"Processed {len(df)} rows")
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("Starting code processing...")
    
    # Try the simple version first
    result = process_codes_simple()
    
    if result is None:
        print("\nTrying alternative method...")
        result = process_codes()
    
    if result is not None:
        print("\n✓ Success! File processed successfully.")
    else:
        print("\n✗ Failed to process the file.")