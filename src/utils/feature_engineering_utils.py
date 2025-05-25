"""
Utility functions for common feature engineering tasks in demand forecasting.

This module provides functions to generate:
- Date-based features (year, month, week, day, cyclical features).
- Lagged features for a target variable, respecting groups.
- Rolling window statistical features (mean, std, min, max) for a target
  variable, respecting groups and preventing data leakage.
- A comprehensive orchestrator function to apply these transformations.
"""
import pandas as pd
import numpy as np

def generate_date_features(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """
    Generates various date and cyclical time-based features from a date column.

    Features created: 'Year', 'Month', 'WeekOfYear', 'DayOfYear', 'DayOfWeek',
    'Month_sin', 'Month_cos', 'WeekOfYear_sin', 'WeekOfYear_cos'.
    Commented out: 'IsWeekend', cyclical DayOfWeek features (can be enabled if needed).

    Args:
        df: Input DataFrame.
        date_col: Name of the column containing datetime objects.
                  If not datetime, it will be converted.

    Returns:
        DataFrame with added date-based features.
    """
    if date_col not in df.columns:
        print(f"Warning: Date column '{date_col}' not found. Skipping date feature generation.")
        return df

    df_copy = df.copy()
    # Ensure the date column is of datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        try:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        except Exception as e:
            print(f"Error converting column '{date_col}' to datetime: {e}. Skipping date feature generation.")
            return df

    df_copy['Year'] = df_copy[date_col].dt.year
    df_copy['Month'] = df_copy[date_col].dt.month
    df_copy['WeekOfYear'] = df_copy[date_col].dt.isocalendar().week.astype(int)
    df_copy['DayOfYear'] = df_copy[date_col].dt.dayofyear
    df_copy['DayOfWeek'] = df_copy[date_col].dt.dayofweek  # Monday=0, Sunday=6
    # df_copy['IsWeekend'] = df_copy['DayOfWeek'].isin([5, 6]).astype(int) # Enable if daily data or relevant

    # Cyclical features for month and week of year
    df_copy['Month_sin'] = np.sin(2 * np.pi * df_copy['Month'] / 12.0)
    df_copy['Month_cos'] = np.cos(2 * np.pi * df_copy['Month'] / 12.0)
    df_copy['WeekOfYear_sin'] = np.sin(2 * np.pi * df_copy['WeekOfYear'] / 52.0) # Using 52 for weekly data
    df_copy['WeekOfYear_cos'] = np.cos(2 * np.pi * df_copy['WeekOfYear'] / 52.0)
    # Optional: Cyclical features for day of week
    # df_copy['DayOfWeek_sin'] = np.sin(2 * np.pi * df_copy['DayOfWeek'] / 7.0)
    # df_copy['DayOfWeek_cos'] = np.cos(2 * np.pi * df_copy['DayOfWeek'] / 7.0)

    print(f"Generated date-based features from column '{date_col}'.")
    return df_copy

def generate_lag_features(df: pd.DataFrame, group_cols: list[str], target_col: str, lags: list[int]) -> pd.DataFrame:
    """
    Generates lag features for a target column, grouped by specified columns.

    The input DataFrame `df` **must be sorted** by the `group_cols` and
    the relevant date/time column before calling this function to ensure
    correct lag calculation within each group.

    Args:
        df: Input DataFrame, pre-sorted by `group_cols` and date.
        group_cols: List of column names to group by (e.g., ['Store', 'Dept']).
        target_col: Name of the target column for which to create lags.
        lags: List of integers representing the lag periods (e.g., [1, 7, 14]).

    Returns:
        DataFrame with added lag features (e.g., '{target_col}_Lag_{lag}').
    """
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found. Skipping lag feature generation.")
        return df
    if not all(col in df.columns for col in group_cols):
        missing_groups = [col for col in group_cols if col not in df.columns]
        print(f"Warning: Group columns {missing_groups} not found. Skipping lag feature generation.")
        return df

    df_copy = df.copy()
    print(f"Generating lag features for '{target_col}' with lags: {lags}, grouped by: {group_cols}...")
    for lag in lags:
        df_copy[f'{target_col}_Lag_{lag}'] = df_copy.groupby(group_cols, observed=True)[target_col].shift(lag)
    return df_copy

def generate_rolling_window_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    windows: list[int],
    stat_fns: list[str] | None = None
) -> pd.DataFrame:
    """
    Generates rolling window statistical features for a target column,
    grouped by specified columns. Uses `shift(1)` on the target within
    each group before calculating rolling statistics to prevent data leakage
    from the current observation.

    The input DataFrame `df` **must be sorted** by the `group_cols` and
    the relevant date/time column before calling this function.

    Args:
        df: Input DataFrame, pre-sorted by `group_cols` and date.
        group_cols: List of column names to group by (e.g., ['Store', 'Dept']).
        target_col: Name of the target column for which to calculate rolling stats.
        windows: List of integers representing the window sizes for rolling stats.
        stat_fns (optional): List of strings specifying statistics to compute.
                             Supported: 'mean', 'std', 'min', 'max'.
                             Defaults to ['mean', 'std'].

    Returns:
        DataFrame with added rolling window features
        (e.g., '{target_col}_Roll_Mean_{window}').
    """
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found. Skipping rolling window feature generation.")
        return df
    if not all(col in df.columns for col in group_cols):
        missing_groups = [col for col in group_cols if col not in df.columns]
        print(f"Warning: Group columns {missing_groups} not found. Skipping rolling window feature generation.")
        return df

    if stat_fns is None:
        stat_fns = ['mean', 'std'] # Default statistics

    df_copy = df.copy()
    print(f"Generating rolling window features for '{target_col}' (windows: {windows}, stats: {stat_fns}), grouped by: {group_cols}...")

    for window in windows:
        if 'mean' in stat_fns:
            df_copy[f'{target_col}_Roll_Mean_{window}'] = df_copy.groupby(group_cols, observed=True)[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
        if 'std' in stat_fns:
            df_copy[f'{target_col}_Roll_Std_{window}'] = df_copy.groupby(group_cols, observed=True)[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )
        if 'min' in stat_fns:
            df_copy[f'{target_col}_Roll_Min_{window}'] = df_copy.groupby(group_cols, observed=True)[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
            )
        if 'max' in stat_fns:
            df_copy[f'{target_col}_Roll_Max_{window}'] = df_copy.groupby(group_cols, observed=True)[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
            )
    return df_copy


def engineer_all_demand_features(
    df_input: pd.DataFrame,
    date_col: str = 'Date',
    target_col: str = 'Weekly_Sales',
    group_cols_for_ts: list[str] | None = None
) -> pd.DataFrame:
    """
    Applies a comprehensive suite of feature engineering steps tailored for
    demand forecasting tasks.

    This orchestrator function calls individual feature generation utilities for
    date features, lag features, and rolling window features.

    Important:
        If `group_cols_for_ts` is provided (e.g., ['Store', 'Dept']), the input
        DataFrame `df_input` **must be pre-sorted** by these grouping columns
        and the `date_col` to ensure correct calculation of time-dependent
        features like lags and rolling statistics.

    Args:
        df_input: Input DataFrame. Must contain `date_col` and `target_col`.
        date_col: Name of the date column. Defaults to 'Date'.
        target_col: Name of the target sales column. Defaults to 'Weekly_Sales'.
        group_cols_for_ts (optional): List of column names to group by when
            generating lag and rolling window features (e.g., ['Store', 'Dept']).
            If None, a warning is issued if default 'Store'/'Dept' are not found,
            as lags/rolls might be incorrect for multi-series data without grouping.
            For single time series DataFrames, this can be None or will not affect results.

    Returns:
        pd.DataFrame: DataFrame augmented with all engineered features. Returns the
                      original DataFrame if input is empty or essential columns are missing.
    """
    if df_input is None or df_input.empty:
        print("Input DataFrame is empty or None. Skipping all feature engineering.")
        return df_input

    df = df_input.copy()

    # 1. Generate Date Features
    print("\nStep 1: Generating Date Features...")
    df = generate_date_features(df, date_col=date_col)

    # Determine grouping columns for time-series features (lags, rolling windows)
    actual_group_cols = []
    if group_cols_for_ts: # If user explicitly provides them
        actual_group_cols = [col for col in group_cols_for_ts if col in df.columns]
        if len(actual_group_cols) != len(group_cols_for_ts):
            missing = [col for col in group_cols_for_ts if col not in actual_group_cols]
            print(f"Warning: Specified group_cols {missing} not found in DataFrame. "
                  "Lag/rolling features might be affected or skipped for these groups.")
    elif 'Store' in df.columns and 'Dept' in df.columns: # Default if nothing provided
        actual_group_cols = ['Store', 'Dept']
        print(f"Using default group_cols: {actual_group_cols} for lag/rolling features.")
    else:
        print("Warning: `group_cols_for_ts` not provided, and default 'Store'/'Dept' columns not found. "
              "Lag and rolling window features will be calculated across the entire dataset, "
              "which may be incorrect if it contains multiple independent time series.")
        # If no grouping, lag/rolling features operate on the whole DataFrame. This is only
        # correct if df_input is already a single time series.

    # 2. Generate Lag Features
    # These should typically match features used during model training.
    # The input DataFrame `df` must be sorted by date within each group defined by `actual_group_cols`.
    if actual_group_cols: # Proceed only if sensible grouping columns are identified
        print("\nStep 2: Generating Lag Features...")
        sales_lags_config = [1, 2, 3, 4, 8, 12, 26, 52] # Example lags
        df = generate_lag_features(df, group_cols=actual_group_cols, target_col=target_col, lags=sales_lags_config)
    elif not actual_group_cols and len(df) > 1: # No groups, but more than one row, print warning for global lags
        print("Info: No grouping columns for lag features; lags will be global. Ensure this is intended for your data structure.")
        sales_lags_config = [1, 2, 3, 4] # Shorter lags might be more relevant for non-grouped data
        # For a single series passed without group_cols, groupby([dummy_col]) could be used, or pass a dummy list.
        # However, generate_lag_features currently expects group_cols.
        # For now, we skip if no valid group_cols and df has multiple series implicitly.
        # A better approach would be to require group_cols, or handle single series explicitly.

    # 3. Generate Rolling Window Features
    # Similar to lags, DataFrame must be sorted by date within each group.
    if actual_group_cols: # Proceed only if sensible grouping columns are identified
        print("\nStep 3: Generating Rolling Window Features...")
        sales_windows_config = [4, 8, 12, 26] # Example window sizes
        sales_stat_fns_config = ['mean', 'std', 'min', 'max'] # Statistics to calculate
        df = generate_rolling_window_features(
            df, group_cols=actual_group_cols, target_col=target_col,
            windows=sales_windows_config, stat_fns=sales_stat_fns_config
        )
    elif not actual_group_cols and len(df) > 1:
        print("Info: No grouping columns for rolling window features; stats will be global. Ensure this is intended.")
        # Similar considerations as for lags if actual_group_cols is empty.

    # Other features (e.g., 'IsHoliday', 'Type_Encoded') are assumed to be either
    # already present in df_input or handled by other preprocessing steps.

    print("\nComprehensive feature engineering process complete.")
    return df


if __name__ == '__main__': # pragma: no cover
    print("--- Testing feature_engineering_utils.py ---")

    # Create a more representative dummy DataFrame for Walmart-like data
    dates = pd.to_datetime(['2010-02-05', '2010-02-12', '2010-02-19', '2010-02-26', '2010-03-05',
                            '2010-03-12', '2010-03-19', '2010-03-26', '2010-04-02', '2010-04-09'])
    data_store1_dept1 = {
        'Date': dates,
        'Store': 1,
        'Dept': 1,
        'Weekly_Sales': np.array([24924.50, 46039.49, 41595.55, 19403.54, 21827.90,
                                  21043.39, 22136.64, 26229.21, 57258.43, 42960.91]),
        'IsHoliday': [False, True, False, False, False, False, False, False, False, False]
    }
    data_store1_dept2 = {
        'Date': dates,
        'Store': 1,
        'Dept': 2,
        'Weekly_Sales': np.array([50605.27, 45039.40, 40595.50, 18403.50, 20827.90,
                                  20043.30, 21136.60, 25229.20, 56258.40, 41960.90]),
        'IsHoliday': [False, True, False, False, False, False, False, False, False, False]
    }
    sample_df_s1d1 = pd.DataFrame(data_store1_dept1)
    sample_df_s1d2 = pd.DataFrame(data_store1_dept2)
    sample_df_combined = pd.concat([sample_df_s1d1, sample_df_s1d2], ignore_index=True)

    # IMPORTANT: Sort before feature engineering involving lags or rolling windows
    sample_df_combined.sort_values(by=['Store', 'Dept', 'Date'], inplace=True)
    sample_df_combined.reset_index(drop=True, inplace=True)

    print("\nOriginal Combined Sample DataFrame (Sorted):")
    print(sample_df_combined)

    # Test comprehensive feature engineering
    # Explicitly pass group_cols_for_ts for data known to have groups
    df_with_all_features = engineer_all_demand_features(
        sample_df_combined,
        group_cols_for_ts=['Store', 'Dept']
    )

    print("\nDataFrame with all engineered features:")
    pd.set_option('display.max_columns', None) # Ensure all columns are shown
    pd.set_option('display.width', 200)      # Adjust console display width
    print(df_with_all_features.head(20)) # Print more rows to see effects across groups
    print("\n--- Info of DataFrame with all features: ---")
    df_with_all_features.info(verbose=True, show_counts=True)

    print("\n--- Checking NaNs introduced by lags/rolling windows (expected at start of each group): ---")
    lag_roll_columns = [col for col in df_with_all_features.columns if 'Lag_' in col or 'Roll_' in col]
    if lag_roll_columns:
        print(df_with_all_features[lag_roll_columns].isnull().sum())
    else:
        print("No lag or rolling window features were generated (check logs).")

    print("\n--- Test: Single series without explicit group_cols (should use defaults if available or warn) ---")
    single_series_df = sample_df_s1d1.copy()
    # When group_cols_for_ts is None, it tries to find 'Store', 'Dept'
    df_single_featured = engineer_all_demand_features(single_series_df)
    print(df_single_featured.head())