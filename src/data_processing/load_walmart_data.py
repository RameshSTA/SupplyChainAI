"""
Module for loading, merging, and performing initial cleaning of Walmart sales data.

This script defines a function to read 'train.csv', 'features.csv', and
'stores.csv', merge them into a single pandas DataFrame, and perform
basic data type conversions and sorting. It includes path determination logic
to locate the raw data files relative to the project structure.
"""
import pandas as pd
import os

def _determine_data_paths() -> tuple[str, str]:
    """
    Determines the project root and the path to the raw data directory.

    It first attempts to locate the project root by navigating up from the
    current file's location, assuming the script is within a structured
    project (e.g., 'src/data_processing/'). If `__file__` is not defined
    (e.g., in an interactive session), it falls back to using the current
    working directory and attempts to find the 'data/raw' subdirectory,
    trying one level up if necessary.

    Returns:
        tuple[str, str]: A tuple containing:
            - project_root (str): The determined absolute path to the project root.
            - data_raw_path (str): The determined absolute path to the 'data/raw' directory.
    """
    try:
        # Assumes this script might be in 'src/data_processing/'
        # So, project root is three levels up from this file's directory.
        current_file_path = os.path.abspath(__file__)
        # src/data_processing -> src -> project_root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    except NameError:  # __file__ is not defined
        project_root = os.getcwd()
        # Fallback heuristic: if 'data/raw' isn't in CWD, try one level up.
        # This helps if CWD is 'src', 'notebooks', or 'app' inside the project root.
        if not os.path.exists(os.path.join(project_root, 'data', 'raw')):
            parent_dir = os.path.abspath(os.path.join(project_root, '..'))
            if os.path.exists(os.path.join(parent_dir, 'data', 'raw')):
                project_root = parent_dir
            else:
                # If still not found, issue a warning but proceed with current project_root guess.
                # The load_and_merge_walmart_data function will later check specific file paths.
                print(
                    f"Warning: Could not reliably auto-locate 'data/raw' relative to CWD ({project_root}). "
                    "Ensure the 'data_path' argument for loading functions is correct or "
                    "that the script is run from a recognized project location."
                )
    data_raw_path = os.path.join(project_root, 'data', 'raw')
    return project_root, data_raw_path

# Determine paths at module level for default use.
# These can be overridden by passing 'data_path' to the loading function.
_PROJECT_ROOT, _DEFAULT_DATA_RAW_PATH = _determine_data_paths()


def load_and_merge_walmart_data(data_path: str = _DEFAULT_DATA_RAW_PATH) -> pd.DataFrame | None:
    """
    Loads Walmart sales forecasting datasets and merges them.

    Reads 'train.csv', 'features.csv', and 'stores.csv' from the specified
    `data_path`, merges them based on common keys, converts the 'Date'
    column to datetime objects, and sorts the resulting DataFrame.

    Args:
        data_path (str, optional): Path to the directory containing the raw
            CSV files ('train.csv', 'features.csv', 'stores.csv').
            Defaults to a path determined relative to this script's location
            (typically 'PROJECT_ROOT/data/raw/').

    Returns:
        pandas.DataFrame | None: A merged and initially cleaned DataFrame containing
            sales, store, and feature data. Returns None if any required file
            is not found or if a significant error occurs during loading or merging.
    """
    print(f"Attempting to load data from: {data_path}")
    try:
        train_file = os.path.join(data_path, 'train.csv')
        features_file = os.path.join(data_path, 'features.csv')
        stores_file = os.path.join(data_path, 'stores.csv')

        # Validate existence of all required files before attempting to load
        required_files = {'train.csv': train_file, 'features.csv': features_file, 'stores.csv': stores_file}
        for name, path_to_file in required_files.items():
            if not os.path.exists(path_to_file):
                print(f"ERROR: Required data file '{name}' not found at '{path_to_file}'.")
                return None

        df_train = pd.read_csv(train_file)
        df_features = pd.read_csv(features_file)
        df_stores = pd.read_csv(stores_file)

    except FileNotFoundError: # Should be caught by the checks above, but as a safeguard.
        print(f"Error: One or more CSV files not found in '{data_path}'.") # Redundant due to prior check
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading files: {e}")
        return None

    # Merge DataFrames
    try:
        df_merged = pd.merge(df_train, df_stores, on='Store', how='left')
        df_merged = pd.merge(df_merged, df_features, on=['Store', 'Date', 'IsHoliday'], how='left')
    except Exception as e:
        print(f"Error during DataFrame merging: {e}")
        return None

    # Initial Data Cleaning & Transformation
    try:
        df_merged['Date'] = pd.to_datetime(df_merged['Date'])
        df_merged.sort_values(by=['Store', 'Dept', 'Date'], inplace=True)
        df_merged.reset_index(drop=True, inplace=True)
    except Exception as e:
        print(f"Error during initial data cleaning (Date conversion or sorting): {e}")
        return None

    print(f"Successfully loaded and merged data. Final DataFrame shape: {df_merged.shape}")
    return df_merged

if __name__ == '__main__':
    print("Running 'load_walmart_data.py' as a standalone script for testing...")
    print(f"Using determined PROJECT_ROOT: {_PROJECT_ROOT}")
    print(f"Using default DATA_RAW_PATH: {_DEFAULT_DATA_RAW_PATH}")

    # Test with default path first
    merged_df = load_and_merge_walmart_data()

    if merged_df is not None:
        print("\n--- Test Succeeded: Basic Info of Merged DataFrame ---")
        merged_df.info(verbose=True, show_counts=True) # More detailed info
        print("\n--- Test: Missing values summary (sum of nulls per column): ---")
        missing_counts = merged_df.isnull().sum()
        print(missing_counts[missing_counts > 0].sort_values(ascending=False)) # Show only cols with missing
        print(f"\n--- Test: First 5 rows of merged data: ---")
        print(merged_df.head())
    else:
        print("\nTest Failed: Data loading and merging did not return a DataFrame.")
        print("Please check the file paths and ensure CSV files are not corrupted.")

    print("\n--- End of 'load_walmart_data.py' test run ---")