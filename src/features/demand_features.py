"""
Feature engineering script for Walmart sales demand forecasting.

This script loads cleaned Walmart sales data, engineers various time-series
features relevant for demand forecasting (e.g., date/time components, lags,
rolling window statistics, cyclical features), and saves the augmented
DataFrame.
"""
import pandas as pd
import numpy as np
import os

# --- Configuration: Define base paths ---
# This script determines paths relative to its assumed location within the project.
try:
    # Assumes this script is in 'PROJECT_ROOT/src/features/'
    # Thus, PROJECT_ROOT is three levels up from this file's directory.
    _CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CURRENT_SCRIPT_PATH)))
except NameError:  # __file__ is not defined (e.g., in an interactive environment)
    # Fallback: Assume current working directory is project root.
    # This might need adjustment if running interactively from a different depth.
    PROJECT_ROOT = os.getcwd()
    print(
        f"Warning: `__file__` attribute not defined. Using current working directory "
        f"as PROJECT_ROOT: '{PROJECT_ROOT}'. Ensure this is correct for data paths."
    )

PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
CLEANED_DATA_FILENAME = 'walmart_data_cleaned.parquet'
FEATURED_DATA_FILENAME = 'walmart_data_featured.parquet'

def load_cleaned_data(file_path: str) -> pd.DataFrame | None:
    """
    Loads cleaned Walmart sales data from a Parquet or CSV file.

    It first attempts to load the specified file. If it's a Parquet file
    and not found, it attempts to load a CSV file with the same base name
    as a fallback.

    Args:
        file_path (str): The full path to the cleaned data file.
                         Typically ends with '.parquet' or '.csv'.

    Returns:
        pd.DataFrame | None: Loaded DataFrame, or None if loading fails or
                             the file is not found after checking fallbacks.
    """
    print(f"--- Attempting to load cleaned data from: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: Cleaned data file not found at '{file_path}'.")
        # Fallback for Parquet: try loading CSV if Parquet specified but not found
        if file_path.lower().endswith('.parquet'):
            csv_fallback_path = file_path[:-len('.parquet')] + '.csv'
            if os.path.exists(csv_fallback_path):
                print(f"Primary file '{file_path}' not found. Attempting to load CSV fallback: '{csv_fallback_path}'")
                file_path = csv_fallback_path
            else:
                print(f"Neither Parquet ('{file_path}') nor CSV fallback ('{csv_fallback_path}') found.")
                return None
        else: # If not a parquet file initially and not found
            return None

    try:
        if file_path.lower().endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, parse_dates=['Date']) # Ensure 'Date' is parsed if CSV
        else:
            print(f"Error: Unsupported file format for '{file_path}'. Please use .parquet or .csv.")
            return None
        print(f"Successfully loaded cleaned data from '{file_path}'. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading cleaned data from '{file_path}': {e}")
        return None

def engineer_demand_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Engineers new features for demand forecasting from the cleaned Walmart DataFrame.

    Features created include:
    - Date/Time components (Year, Month, WeekOfYear, DayOfYear, DayOfWeek, IsWeekend).
    - Lagged sales features (for various past weeks).
    - Rolling window statistics (mean, std, min, max of sales over past periods).
    - Cyclical features for month and week of year using sin/cos transformations.
    - Ensures 'IsHoliday' is integer type.

    Args:
        df (pd.DataFrame): The cleaned Walmart DataFrame. Must contain 'Store',
                           'Dept', 'Date', and 'Weekly_Sales' columns.
                           The 'Date' column is expected to be of datetime type
                           or convertible to it.

    Returns:
        pd.DataFrame | None: The DataFrame augmented with new features,
                             or None if the input DataFrame is invalid.
    """
    if df is None:
        print("Error: Input DataFrame is None. Cannot engineer features.")
        return None
    required_cols = ['Store', 'Dept', 'Date', 'Weekly_Sales']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Input DataFrame is missing required columns: {missing}. Cannot engineer features.")
        return None

    print("\n--- Starting Feature Engineering for Demand Forecasting ---")
    df_featured = df.copy()

    # Ensure 'Date' column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_featured['Date']):
        try:
            df_featured['Date'] = pd.to_datetime(df_featured['Date'])
            print("Converted 'Date' column to datetime objects.")
        except Exception as e_date:
            print(f"Error converting 'Date' column to datetime: {e_date}. Aborting feature engineering.")
            return None

    # Sort data - CRUCIAL for correct calculation of lags and rolling windows
    df_featured.sort_values(by=['Store', 'Dept', 'Date'], inplace=True)
    df_featured.reset_index(drop=True, inplace=True) # Good practice after sort

    # 1. Date/Time Features
    print("Creating Date/Time features (Year, Month, WeekOfYear, DayOfYear, DayOfWeek, IsWeekend)...")
    df_featured['Year'] = df_featured['Date'].dt.year
    df_featured['Month'] = df_featured['Date'].dt.month
    df_featured['WeekOfYear'] = df_featured['Date'].dt.isocalendar().week.astype(int)
    df_featured['DayOfYear'] = df_featured['Date'].dt.dayofyear
    df_featured['DayOfWeek'] = df_featured['Date'].dt.dayofweek  # Monday=0, Sunday=6
    # 'IsWeekend' might be more relevant for daily data, but included for completeness
    df_featured['IsWeekend'] = df_featured['DayOfWeek'].isin([5, 6]).astype(int)

    # 2. Lag Features for Weekly_Sales
    # These represent sales from previous weeks for the same store-department.
    print("Creating Lag features for Weekly_Sales (lags: 1, 2, 3, 4, 8, 12, 26, 52 weeks)...")
    lags = [1, 2, 3, 4, 8, 12, 26, 52]
    for lag in lags:
        df_featured[f'Sales_Lag_{lag}'] = df_featured.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)

    # 3. Rolling Window Statistics for Weekly_Sales
    # These capture trends and volatility using data *prior* to the current week (using shift(1)).
    print("Creating Rolling Window features for Weekly_Sales (means, std, min, max for windows: 4, 8, 12, 26 weeks)...")
    windows = [4, 8, 12, 26]
    grouped_sales = df_featured.groupby(['Store', 'Dept'])['Weekly_Sales']
    for window in windows:
        # Calculate rolling stats on sales data shifted by 1 to avoid data leakage from current week's sales
        shifted_sales = grouped_sales.transform(lambda x: x.shift(1))
        df_featured[f'Sales_Roll_Mean_{window}'] = shifted_sales.rolling(window=window, min_periods=1).mean()
        df_featured[f'Sales_Roll_Std_{window}'] = shifted_sales.rolling(window=window, min_periods=1).std()
        df_featured[f'Sales_Roll_Min_{window}'] = shifted_sales.rolling(window=window, min_periods=1).min()
        df_featured[f'Sales_Roll_Max_{window}'] = shifted_sales.rolling(window=window, min_periods=1).max()

    # 4. Cyclical Features for Seasonality (Month and WeekOfYear)
    # These represent cyclical patterns in a way that's continuous for models.
    print("Creating Cyclical features for Month and WeekOfYear (sin/cos transformations)...")
    df_featured['Month_sin'] = np.sin(2 * np.pi * df_featured['Month'] / 12.0)
    df_featured['Month_cos'] = np.cos(2 * np.pi * df_featured['Month'] / 12.0)
    df_featured['WeekOfYear_sin'] = np.sin(2 * np.pi * df_featured['WeekOfYear'] / 52.0) # Approx 52 weeks
    df_featured['WeekOfYear_cos'] = np.cos(2 * np.pi * df_featured['WeekOfYear'] / 52.0)

    # 5. IsHoliday feature (ensure it's integer type if present)
    if 'IsHoliday' in df_featured.columns:
        df_featured['IsHoliday'] = df_featured['IsHoliday'].astype(int)
    else:
        print("Warning: 'IsHoliday' column not found. This feature will be missing.")

    # --- Information about NaNs from new features ---
    # Lag and rolling window features naturally create NaNs at the start of each time series.
    # These NaNs typically need to be handled (e.g., imputation, or some models handle them)
    # before training certain types of models.
    new_lag_roll_features = [col for col in df_featured.columns if 'Sales_Lag_' in col or 'Sales_Roll_' in col]
    num_new_features = len(df_featured.columns) - len(df.columns)
    print(f"\nNumber of new features engineered: {num_new_features}")
    print("Note: NaNs introduced by lag/rolling features will require handling prior to some model training stages.")
    if new_lag_roll_features: # Only print if such features were created
        nan_summary = df_featured[new_lag_roll_features].isnull().sum()
        print("Top 10 features with most NaNs (typically lags/rolling windows):")
        print(nan_summary[nan_summary > 0].sort_values(ascending=False).head(10))

    print("--- Feature Engineering Complete ---")
    return df_featured

def save_featured_data(df: pd.DataFrame, file_path: str):
    """
    Saves the DataFrame with engineered features to a Parquet or CSV file.

    The directory for the file path will be created if it does not exist.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The full path (including filename and extension)
                         where the file should be saved. Supports '.parquet'
                         and '.csv' extensions.
    """
    if df is None:
        print("Error: Input DataFrame is None. Nothing to save.")
        return
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Error: Input is not a non-empty DataFrame. Nothing to save.")
        return

    print(f"\n--- Saving DataFrame with engineered features to: {file_path} ---")
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if file_path.lower().endswith('.parquet'):
            df.to_parquet(file_path, index=False)
        elif file_path.lower().endswith('.csv'):
            df.to_csv(file_path, index=False)
        else:
            print(f"Error: Unsupported file format for saving: '{file_path}'. Please use .parquet or .csv.")
            return
        print(f"Successfully saved featured data to '{file_path}'")
    except Exception as e:
        print(f"Error saving featured data to '{file_path}': {e}")

# --- Main execution block for this script ---
if __name__ == '__main__':
    print("--- Running Feature Engineering Script (demand_features.py) ---")
    print(f"Using PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Processed data expected at: {PROCESSED_DATA_PATH}")

    # 1. Load cleaned data
    cleaned_data_full_path = os.path.join(PROCESSED_DATA_PATH, CLEANED_DATA_FILENAME)
    df_cleaned_main = load_cleaned_data(cleaned_data_full_path)

    if df_cleaned_main is not None:
        # 2. Engineer features
        df_featured_main = engineer_demand_features(df_cleaned_main)

        if df_featured_main is not None:
            # 3. Save the DataFrame with features
            featured_data_full_path = os.path.join(PROCESSED_DATA_PATH, FEATURED_DATA_FILENAME)
            save_featured_data(df_featured_main, featured_data_full_path)

            print("\n--- Feature Engineering Script Finished Successfully ---")
            print(f"Featured data saved at: {featured_data_full_path}")
            print("\n--- First 5 rows of featured data: ---")
            print(df_featured_main.head())
            print("\n--- Info of featured data: ---")
            df_featured_main.info(verbose=True, show_counts=True)
        else:
            print("Feature engineering process failed. Output data not saved.")
    else:
        print("Loading cleaned data failed. Cannot proceed with feature engineering.")

    print("\n--- End of demand_features.py run ---")