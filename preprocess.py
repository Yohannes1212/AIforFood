import pandas as pd
import numpy as np
import os
from glob import glob

def load_s1p_file(file_path):
    """
    Load an s1p file and return frequency, gain, and phase data
    """
    try:
        # Skip the header lines (if any) and load the data
        data = pd.read_csv(file_path, delimiter=r'\s+', header=None, skiprows=0)
        frequency = data.iloc[:, 0]
        gain = data.iloc[:, 1]
        phase = data.iloc[:, 2]
        return frequency, gain, phase
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None, None, None

def extract_metadata(filename):
    """
    Extract product type and storage condition from filename
    Example: A_1_1.s1p -> Product Type: A (bread), Storage: 1 (open)
    """
    parts = os.path.basename(filename).split('_')
    product_type = 'Bread' if parts[0] == 'A' else 'Cookies'
    storage_condition = {
        '1': 'Open',
        '2': 'Wrapped',
        '3': 'Humid'
    }.get(parts[1], 'Unknown')
    
    return product_type, storage_condition

def create_dataset(data_folder):
    """
    Create a dataset from all s1p files in the folder
    """
    # Get all s1p files
    s1p_files = glob(os.path.join(data_folder, '*.s1p'))
    
    if not s1p_files:
        raise ValueError(f"No .s1p files found in {data_folder}")

    # Lists to store data
    all_features = []
    product_types = []
    storage_conditions = []

    # Process each file
    for file_path in s1p_files:
        # Load data
        frequency, gain, phase = load_s1p_file(file_path)
        
        if gain is None or phase is None:
            continue
            
        # Concatenate gain and phase as features
        features = np.concatenate([gain, phase])
        
        # Extract metadata
        product_type, storage = extract_metadata(file_path)
        
        # Append to lists
        all_features.append(features)
        product_types.append(product_type)
        storage_conditions.append(storage)

    # Convert to numpy arrays
    X = np.array(all_features)
    
    # Create feature names
    feature_names = ([f'gain_{i}' for i in range(len(gain))] + 
                    [f'phase_{i}' for i in range(len(phase))])
    
    # Create DataFrame with features
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add labels
    df['Product_Type'] = product_types
    df['Storage_Condition'] = storage_conditions
    
    return df

def main():
    # Set the path to your data folder
    data_folder = './Food/Bakery'  # Adjust this path as needed
    
    try:
        # Create the dataset
        print("Processing files...")
        df = create_dataset(data_folder)
        
        # Save the processed dataset
        output_file = 'processed_bakery_data.csv'
        df.to_csv(output_file, index=False)
        print(f"\nDataset created successfully!")
        print(f"Shape: {df.shape}")
        print(f"\nSample of the data:")
        print(df.head())
        
        # Print some basic information
        print("\nProduct Type distribution:")
        print(df['Product_Type'].value_counts())
        print("\nStorage Condition distribution:")
        print(df['Storage_Condition'].value_counts())
        
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")

if __name__ == "__main__":
    main()