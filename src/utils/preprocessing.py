"""
Utility functions for data preprocessing in machine learning pipelines.

This module provides functions for:
1.  Preprocessing features for regression models, including handling missing
    values, label encoding categorical features, and aligning features
    with a predefined list (e.g., for prediction using a trained model).
2.  Creating sequential data (X, y pairs) suitable for time series
    forecasting models like LSTMs.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_regression_features(
    df: pd.DataFrame,
    target_col: str | None = 'Weekly_Sales',
    feature_list_from_training: list[str] | None = None,
    is_training_phase: bool = True
) -> tuple[pd.DataFrame | None, pd.Series | None, pd.Series | None, list[str] | None]:
    """
    Prepares a DataFrame for regression modeling.

    Handles missing value imputation (median for numeric, mode for categorical),
    label encodes 'Type', and conditionally 'Store' and 'Dept' if they are
    non-numeric. Aligns features to `feature_list_from_training` if provided
    (typically for prediction/evaluation phases), or derives features during
    the training phase.

    Args:
        df: Input DataFrame. Must contain 'Date', 'Store', 'Dept', and features.
            If `is_training_phase` is True, `target_col` must also be present.
        target_col: Name of the target variable column. If `is_training_phase`
            is False, this can be None (no 'y' will be returned).
            Defaults to 'Weekly_Sales'.
        feature_list_from_training (optional): A specific list of feature names
            that the output `X_processed` should contain, in that order.
            - If provided (prediction/evaluation): `X_processed` will have these
              features. Missing features from this list will be added as columns
              of zeros.
            - If None (training): Features are derived from `df` (excluding target
              and 'Date'), and `final_feature_columns_in_X` will be the sorted
              list of these derived features.
        is_training_phase: Boolean indicating if the function is used for training
            (True) or prediction/evaluation (False). Defaults to True.

    Returns:
        A tuple (X_processed, y, dates, final_feature_columns_in_X):
        - X_processed: DataFrame of processed features.
        - y: Series of the target variable (None if not in training phase or target_col is None).
        - dates: Series of 'Date' values, aligned with X_processed and y.
        - final_feature_columns_in_X: List of column names in X_processed, in their order.
        Returns (None, None, None, None) if a critical error occurs (e.g., empty input, missing essential columns).
    """
    print(f"--- Preprocessing data for regression (Training Phase: {is_training_phase}) ---")
    if df is None or df.empty:
        print("ERROR: Input DataFrame is None or empty. Preprocessing cannot proceed.")
        return None, None, None, None

    required_base_cols = ['Date', 'Store', 'Dept'] # Essential for structure and identity
    for col in required_base_cols:
        if col not in df.columns:
            print(f"ERROR: Required base column '{col}' not found in input DataFrame. Preprocessing aborted.")
            return None, None, None, None

    df_processed = df.copy()
    y_series = None
    dates_series = None # Initialize

    # Ensure 'Date' column is datetime and extract it
    if 'Date' in df_processed.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_processed['Date']):
            try:
                df_processed['Date'] = pd.to_datetime(df_processed['Date'])
            except Exception as e_date: # pragma: no cover
                print(f"ERROR: Could not convert 'Date' column to datetime: {e_date}. Preprocessing aborted.")
                return None, None, None, None
        dates_series = df_processed['Date'].copy() # Store dates before any potential row drops
    else: # Should have been caught by required_base_cols check
        print("ERROR: 'Date' column is missing, which is required. Preprocessing aborted.")
        return None, None, None, None


    if is_training_phase:
        if target_col is None or target_col not in df_processed.columns:
            print(f"ERROR: Target column '{target_col}' must be specified and present in DataFrame for training phase.")
            return None, None, None, None
        # Drop rows where target is NaN for training, as they are unusable for supervised learning
        df_processed.dropna(subset=[target_col], inplace=True)
        if df_processed.empty: # pragma: no cover
            print(f"ERROR: DataFrame became empty after dropping NaNs in target column '{target_col}'. Cannot proceed.")
            return None, None, None, None
        y_series = df_processed[target_col].copy()
        dates_series = df_processed['Date'].copy() # Re-align dates with df_processed after potential row drops
        print(f"Target column '{target_col}' extracted. Shape: {y_series.shape}")

    # Determine features to process
    if feature_list_from_training:
        # Use the provided list, but only process columns actually present in df_processed initially
        # Missing ones will be handled later by adding zero columns if needed.
        features_to_iterate = [f for f in feature_list_from_training if f in df_processed.columns]
        missing_in_df_initially = [f for f in feature_list_from_training if f not in df_processed.columns]
        if missing_in_df_initially:
            print(f"Info: Features from training list initially missing in input df: {missing_in_df_initially}. "
                  "These will be added as zero columns if not created during processing (e.g. Type_Encoded).")
    else: # Training phase or when feature_list_from_training is not provided
        exclude_from_features = [target_col] if target_col and target_col in df_processed.columns else []
        exclude_from_features.append('Date') # 'Date' is handled separately
        features_to_iterate = [col for col in df_processed.columns if col not in exclude_from_features]

    X_intermediate_df = pd.DataFrame(index=df_processed.index) # Preserve index from df_processed (which aligns with y_series and dates_series)
    final_feature_names_collected = []

    # Impute NaNs and perform encoding
    for col in features_to_iterate:
        if col not in df_processed.columns: continue # Should not happen if features_to_iterate is from df_processed.columns

        # Impute missing values
        if df_processed[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                fill_value = df_processed[col].median()
                X_intermediate_df[col] = df_processed[col].fillna(fill_value)
                # print(f"Imputed NaNs in numeric column '{col}' with median: {fill_value}")
            else: # Categorical or object type
                mode_values = df_processed[col].mode()
                fill_value = mode_values[0] if not mode_values.empty else "Unknown"
                X_intermediate_df[col] = df_processed[col].fillna(fill_value)
                # print(f"Imputed NaNs in categorical column '{col}' with mode: {fill_value}")
        else:
            X_intermediate_df[col] = df_processed[col]

        # Specific encodings
        if col == 'Type' and 'Type' in X_intermediate_df.columns: # Ensure 'Type' exists after potential NaN fill
            if 'Type_Encoded' not in final_feature_names_collected:
                le = LabelEncoder()
                try:
                    X_intermediate_df['Type_Encoded'] = le.fit_transform(X_intermediate_df['Type'].astype(str))
                    final_feature_names_collected.append('Type_Encoded')
                except Exception as e_le_type: # pragma: no cover
                    print(f"Warning: Could not label encode 'Type': {e_le_type}. Keeping original 'Type' if no 'Type_Encoded'.")
                    if 'Type' not in final_feature_names_collected: final_feature_names_collected.append('Type')
            # If Type_Encoded was successfully created, original 'Type' might be skipped later if not explicitly in feature_list_from_training
            if 'Type_Encoded' in final_feature_names_collected and col == 'Type':
                continue # Avoid adding original 'Type' if 'Type_Encoded' is the goal and not using a fixed feature list

        # Optional: Encode 'Store' and 'Dept' if they are non-numeric (usually not the case for Walmart data)
        # For this dataset, Store and Dept are typically numeric IDs already.
        # This block is a safeguard if they were, for example, string type.
        if col in ['Store', 'Dept'] and not pd.api.types.is_numeric_dtype(X_intermediate_df[col]): # pragma: no cover
            encoded_col = f"{col}_Encoded"
            if encoded_col not in final_feature_names_collected:
                le_sd = LabelEncoder()
                try:
                    X_intermediate_df[encoded_col] = le_sd.fit_transform(X_intermediate_df[col].astype(str))
                    final_feature_names_collected.append(encoded_col)
                except Exception as e_le_sd_col:
                    print(f"Warning: Could not label encode '{col}': {e_le_sd_col}. Keeping original.")
                    if col not in final_feature_names_collected: final_feature_names_collected.append(col)
            if encoded_col in final_feature_names_collected and col in features_to_iterate:
                continue

        # Add the original or processed column name to the list if not handled by encoding logic above
        if col not in final_feature_names_collected and col not in ['Type', 'Store', 'Dept']: # Avoid adding originals if encoded versions are preferred
             final_feature_names_collected.append(col)
        elif col in ['Type', 'Store', 'Dept'] and f"{col}_Encoded" not in final_feature_names_collected and col not in final_feature_names_collected:
            final_feature_names_collected.append(col) # Add original if not encoded and not already added


    # Finalize feature set for X_processed
    if feature_list_from_training:
        final_feature_columns_for_X = []
        for f_train_name in feature_list_from_training:
            # Handle cases where original name was in training list but encoded version exists now
            if f_train_name == 'Type' and 'Type_Encoded' in X_intermediate_df.columns:
                final_feature_columns_for_X.append('Type_Encoded')
            elif f_train_name == 'Store' and 'Store_Encoded' in X_intermediate_df.columns: # pragma: no cover
                final_feature_columns_for_X.append('Store_Encoded')
            elif f_train_name == 'Dept' and 'Dept_Encoded' in X_intermediate_df.columns: # pragma: no cover
                final_feature_columns_for_X.append('Dept_Encoded')
            elif f_train_name in X_intermediate_df.columns:
                final_feature_columns_for_X.append(f_train_name)
            else: # Feature was in training list but is entirely missing from current data
                print(f"Warning: Feature '{f_train_name}' from training list is not found in current processed data. "
                      "It will be created as a column of zeros for compatibility.")
                X_intermediate_df[f_train_name] = 0 # Add as zero column
                final_feature_columns_for_X.append(f_train_name)
        # Ensure unique columns in the specified order
        X_processed_df = X_intermediate_df[final_feature_columns_for_X].copy()
    else: # Training phase, derive feature list from processed columns
        # Remove original 'Type', 'Store', 'Dept' if their encoded versions were created and are in final_feature_names_collected
        if 'Type_Encoded' in final_feature_names_collected and 'Type' in final_feature_names_collected:
            final_feature_names_collected.remove('Type')
        if 'Store_Encoded' in final_feature_names_collected and 'Store' in final_feature_names_collected: # pragma: no cover
            final_feature_names_collected.remove('Store')
        if 'Dept_Encoded' in final_feature_names_collected and 'Dept' in final_feature_names_collected: # pragma: no cover
            final_feature_names_collected.remove('Dept')

        # Use unique, sorted list of features found/created
        final_feature_columns_for_X = sorted(list(set(final_feature_names_collected)))
        X_processed_df = X_intermediate_df[final_feature_columns_for_X].copy()

    print(f"Shape of processed features X: {X_processed_df.shape}")
    if y_series is not None:
        print(f"Shape of target y: {y_series.shape}")
        if not X_processed_df.index.equals(y_series.index): # pragma: no cover
            # This can happen if df_processed was indexed differently than X_intermediate_df
            # (though X_intermediate_df was initialized with df_processed.index)
            # Or if lengths mismatch due to an error.
            print("Warning: Index of X_processed and y_series do not match. "
                  f"X len: {len(X_processed_df)}, y len: {len(y_series)}. This may cause issues in model training.")
            if len(X_processed_df) == len(y_series): # Attempt re-alignment if lengths are same
                X_processed_df.index = y_series.index
                dates_series = dates_series.loc[y_series.index] # Align dates as well
                print("Attempted to re-align index of X_processed and dates_series with y_series.")


    if X_processed_df.empty and is_training_phase: # pragma: no cover
        print("ERROR: Processed features DataFrame (X_processed_df) is empty. Cannot proceed with training.")
        return None, None, None, None

    return X_processed_df, y_series, dates_series, final_feature_columns_for_X


def create_sequences(
    feature_data: np.ndarray,
    target_data: np.ndarray,
    sequence_length: int,
    future_steps: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences of feature data (X) and corresponding target values (y)
    for training time series forecasting models like LSTMs.

    Args:
        feature_data: NumPy array of input features. Expected shape (n_samples, n_features).
                      If 1D, it's reshaped to (n_samples, 1).
        target_data: NumPy array of the target variable. Expected shape (n_samples,)
                     or (n_samples, 1). If 1D, it's reshaped.
        sequence_length: The number of time steps in each input sequence (X_sequence).
        future_steps: How many steps into the future the target value (y_target) is,
                      relative to the *end* of the input sequence.
                      Default is 1, meaning y_target is the value immediately
                      following the input sequence.

    Returns:
        A tuple containing two NumPy arrays:
        - X_sequences: Input sequences. Shape (num_sequences, sequence_length, n_features).
        - y_targets: Target values. Shape (num_sequences,).
        Returns empty arrays if not enough data is available to form any sequences.

    Raises:
        ValueError: If `feature_data` and `target_data` do not have the same
                    number of samples (i.e., same length along the first axis).
    """
    X_seq_list, y_target_list = [], []

    # Ensure inputs are 2D NumPy arrays for consistent processing
    if feature_data.ndim == 1:
        feature_data = feature_data.reshape(-1, 1)
    if target_data.ndim == 1:
        target_data = target_data.reshape(-1, 1)

    if feature_data.shape[0] != target_data.shape[0]:
        raise ValueError(
            f"feature_data (length {feature_data.shape[0]}) and target_data (length {target_data.shape[0]}) "
            "must have the same number of samples (rows)."
        )

    num_samples = feature_data.shape[0]
    num_features_in_x = feature_data.shape[1]

    # Calculate the number of sequences that can be created
    # The last possible start index `i` for an X_sequence is such that
    # `i + sequence_length` (end of X_sequence) plus `future_steps - 1` (to reach target)
    # is within the bounds of `target_data`.
    # So, `i + sequence_length + future_steps - 1 < num_samples`.
    # Thus, `i < num_samples - sequence_length - future_steps + 1`.
    # The loop will go up to `num_samples - sequence_length - future_steps`.
    # Total iterations: `num_samples - sequence_length - future_steps + 1`.
    if num_samples < sequence_length + future_steps:
        print(
            f"Warning: Not enough data (samples: {num_samples}) to create any sequences "
            f"with sequence_length {sequence_length} and future_steps {future_steps}."
        )
        return np.empty((0, sequence_length, num_features_in_x)), np.empty((0,))

    for i in range(num_samples - sequence_length - future_steps + 1):
        X_seq_list.append(feature_data[i : (i + sequence_length)])
        # Target is `future_steps` from the end of the X sequence.
        # If X ends at index `i + sequence_length - 1`, target is at `i + sequence_length - 1 + future_steps`.
        # Given 0-based indexing, this is `target_data[i + sequence_length + future_steps - 1]`.
        y_target_list.append(target_data[i + sequence_length + future_steps - 1, 0]) # Assuming target is scalar

    # Convert lists to NumPy arrays
    if not X_seq_list: # Should be caught by the length check above, but as a safeguard
        return np.empty((0, sequence_length, num_features_in_x)), np.empty((0,))

    return np.array(X_seq_list), np.array(y_target_list)