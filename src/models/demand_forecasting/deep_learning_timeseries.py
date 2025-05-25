"""
Trains and evaluates Long Short-Term Memory (LSTM) networks for demand forecasting.

This script includes functionalities for:
- Setting up project paths and importing necessary deep learning libraries.
- Logging experiment details (hyperparameters, metrics, paths to saved models/scalers).
- Loading pre-featured sales data.
- Calculating evaluation metrics (MAE, RMSE, MAPE).
- Creating input/output sequences suitable for LSTM training.
- Defining, training, evaluating, and saving LSTM models and associated scalers.
- A main pipeline function to orchestrate the LSTM modeling process for selected store-department pairs.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
import joblib # For saving scalers

# Deep Learning specific imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Project Path Setup ---
_PROJECT_ROOT_DL = None # Module-level variable

def _get_project_root_dl() -> str:
    """Determines project root assuming this script is in a nested structure."""
    try:
        # Assumes file is at: PROJECT_ROOT/src/models/demand_forecasting/deep_learning.py
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
        return project_root
    except NameError:  # __file__ is not defined
        cwd = os.getcwd()
        if os.path.exists(os.path.join(cwd, 'src')): return cwd
        elif os.path.basename(cwd) == 'src': return os.path.dirname(cwd)
        print(
            f"Warning: `__file__` not defined. Using CWD '{cwd}' as PROJECT_ROOT. "
            "Ensure this is correct for data and model paths."
        )
        return cwd

_PROJECT_ROOT_DL = _get_project_root_dl()

# --- Configuration Paths ---
PROCESSED_DATA_PATH = os.path.join(_PROJECT_ROOT_DL, 'data', 'processed')
FEATURED_DATA_FILENAME = 'walmart_data_featured.parquet'
REPORTS_PATH = os.path.join(_PROJECT_ROOT_DL, 'reports')
EXPERIMENT_LOGS_PATH = os.path.join(REPORTS_PATH, 'experiment_logs')
MODEL_STORE_PATH = os.path.join(_PROJECT_ROOT_DL, 'models_store', 'demand_forecasting')
SCALER_STORE_PATH = os.path.join(MODEL_STORE_PATH, 'scalers') # For LSTM scalers

# Ensure necessary directories exist
os.makedirs(EXPERIMENT_LOGS_PATH, exist_ok=True)
os.makedirs(MODEL_STORE_PATH, exist_ok=True)
os.makedirs(SCALER_STORE_PATH, exist_ok=True)

# --- Experiment Logging ---
_DL_EXPERIMENT_LOG_FILE = os.path.join(EXPERIMENT_LOGS_PATH, 'deep_learning_experiments.csv')
_dl_experiment_records = [] # Module-level list

def log_dl_experiment(
    store_id: int,
    dept_id: int,
    model_name: str,
    params: dict, # Should include model_path, scaler_paths, sequence_length, features_used, etc.
    metrics: dict | None,
    train_period_str: str,
    test_period_str: str,
    # features_used_for_lstm is now expected to be part of 'params' from train_evaluate_lstm
    run_timestamp: str | None = None
):
    """
    Logs the details of a single deep learning model training experiment.

    Args:
        store_id: Identifier for the store.
        dept_id: Identifier for the department.
        model_name: Name of the model (e.g., "LSTM_S1D1").
        params: Dictionary of parameters used for the model. This should include
                hyperparameters, paths to saved artifacts (model, scalers),
                sequence length, and the list of features used.
        metrics: Dictionary of evaluation metrics (e.g., {'MAE': val, 'RMSE': val}).
        train_period_str: String representation of the training period.
        test_period_str: String representation of the test period.
        run_timestamp: Timestamp of the run; defaults to current time if None.
    """
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # The `params` dict from train_evaluate_lstm should already contain
    # model_path, scaler paths, sequence_length, features_used_for_lstm_input,
    # train_period_in_params, and test_period_in_params.
    # For logging, we just ensure it's a string.
    features_list_from_params = params.get('features_used_for_lstm_input', [])

    record = {
        'Timestamp': run_timestamp,
        'Store': store_id,
        'Dept': dept_id,
        'Model': model_name,
        'Parameters': str(params), # String representation of the full parameters dict
        'Train_Period': train_period_str,
        'Test_Period': test_period_str,
        'Features_Count': len(features_list_from_params),
        'Features_List_Used': str(features_list_from_params)
    }
    # Add metrics, ensuring keys exist and handling None for metrics
    for metric_key in ['MAE', 'RMSE', 'MAPE']:
        record[metric_key] = metrics.get(metric_key, np.nan) if metrics else np.nan

    _dl_experiment_records.append(record)
    print(f"Logged DL experiment for {model_name} (Store: {store_id}, Dept: {dept_id})")

def save_dl_experiment_log():
    """Saves all accumulated deep learning experiment records to a CSV file."""
    if not _dl_experiment_records:
        print("No new deep learning experiments were logged to save.")
        return

    log_df = pd.DataFrame(_dl_experiment_records)
    try:
        header_needed = not (
            os.path.exists(_DL_EXPERIMENT_LOG_FILE) and
            os.path.getsize(_DL_EXPERIMENT_LOG_FILE) > 0
        )
        log_df.to_csv(_DL_EXPERIMENT_LOG_FILE, mode='a', header=header_needed, index=False)
        print(f"Deep learning experiment log saved/appended to '{_DL_EXPERIMENT_LOG_FILE}'")
        _dl_experiment_records.clear() # Clear records after successful save
    except Exception as e:
        print(f"Error saving deep learning experiment log: {e}")

# --- Data Loading ---
def _load_featured_data_dl(file_path: str = os.path.join(PROCESSED_DATA_PATH, FEATURED_DATA_FILENAME)) -> pd.DataFrame | None:
    """
    Loads the preprocessed and feature-engineered Walmart dataset for DL modeling.
    (Functionally similar to _load_featured_data_ts, kept separate for potential DL-specific preprocessing later)
    """
    print(f"--- Loading featured data for Deep Learning from: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: Featured data file not found at '{file_path}'")
        if file_path.lower().endswith('.parquet'):
            csv_fallback_path = file_path[:-len('.parquet')] + '.csv'
            if os.path.exists(csv_fallback_path):
                print(f"Parquet not found, attempting to load CSV: '{csv_fallback_path}'")
                file_path = csv_fallback_path
            else: return None
        else: return None
    try:
        if file_path.lower().endswith('.parquet'): df = pd.read_parquet(file_path)
        elif file_path.lower().endswith('.csv'): df = pd.read_csv(file_path, parse_dates=['Date'])
        else: print(f"Error: Unsupported file format for '{file_path}'."); return None
        print(f"Successfully loaded featured data from '{file_path}'. Shape: {df.shape}")
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e: print(f"Error loading featured data from '{file_path}': {e}"); return None

# --- Metrics Calculation ---
def _calculate_dl_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates MAE, RMSE, and MAPE for deep learning model evaluation.
    Assumes y_true and y_pred are NumPy arrays of original scale values.
    """
    if y_true.size == 0 or y_pred.size == 0 or len(y_true) != len(y_pred):
        print("Warning: y_true or y_pred is empty or lengths mismatch in _calculate_dl_metrics. Returning NaN metrics.")
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    y_true_mape, y_pred_mape = y_true.copy(), y_pred.copy()
    zero_mask = (y_true_mape == 0)
    y_true_mape[zero_mask] = np.nan
    y_pred_mape[zero_mask] = np.nan

    if np.all(np.isnan(y_true_mape)):
        mape = np.nan
    else:
        mape = np.nanmean(np.abs((y_true_mape - y_pred_mape) / y_true_mape)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# --- Sequence Creation ---
def _create_sequences_for_lstm(
    feature_data: np.ndarray, # Should contain all features, including the target if it's used as a feature
    target_data: np.ndarray,  # Specifically the target column (can be one of the columns in feature_data)
    sequence_length: int,
    future_steps: int = 1 # Number of steps ahead to predict (default is 1 for next step)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences of feature data (X) and corresponding target values (y)
    for training time series forecasting models like LSTMs.

    Args:
        feature_data: NumPy array of input features. Shape (n_samples, n_features).
        target_data: NumPy array of the target variable. Shape (n_samples,) or (n_samples, 1).
        sequence_length: The number of time steps in each input sequence.
        future_steps: How many steps into the future the target value is, relative
                      to the end of the input sequence. Default is 1 (predict next step).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - X_sequences: NumPy array of input sequences. Shape (n_sequences, sequence_length, n_features).
            - y_targets: NumPy array of target values. Shape (n_sequences,).
            Returns empty arrays if not enough data to form sequences.
    """
    X_sequences, y_targets = [], []
    if feature_data.ndim == 1: # Ensure feature_data is 2D
        feature_data = feature_data.reshape(-1, 1)
    if target_data.ndim == 1: # Ensure target_data is 2D for consistent indexing
        target_data = target_data.reshape(-1, 1)

    if len(feature_data) != len(target_data):
        raise ValueError("feature_data and target_data must have the same number of samples.")

    num_total_samples = len(feature_data)
    num_features = feature_data.shape[1]

    # Check if there's enough data to create at least one sequence
    if num_total_samples < sequence_length + future_steps:
        print(f"Warning: Not enough data (samples: {num_total_samples}) to create sequences "
              f"with sequence_length {sequence_length} and future_steps {future_steps}.")
        return np.empty((0, sequence_length, num_features)), np.empty((0,))

    for i in range(num_total_samples - sequence_length - future_steps + 1):
        X_sequences.append(feature_data[i:(i + sequence_length)])
        # Target is future_steps ahead from the end of the current input sequence
        y_targets.append(target_data[i + sequence_length + future_steps - 1, 0]) # Assumes target is single value

    if not X_sequences: # Should be caught by the earlier length check, but as a safeguard
        return np.empty((0, sequence_length, num_features)), np.empty((0,))

    return np.array(X_sequences), np.array(y_targets)

# --- LSTM Model Definition ---
def _define_lstm_model(
    input_shape: tuple, # (sequence_length, n_features)
    lstm_units: int = 50,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Defines a simple LSTM model architecture for time series forecasting.

    Args:
        input_shape: Tuple specifying the shape of input sequences (sequence_length, n_features).
        lstm_units: Number of units in the LSTM layer. Defaults to 50.
        dropout_rate: Dropout rate for regularization. Defaults to 0.2.
        learning_rate: Learning rate for the Adam optimizer. Defaults to 0.001.

    Returns:
        tf.keras.Model: A compiled Keras LSTM model.
    """
    print(f"Defining LSTM model architecture:")
    print(f"  Input Shape: {input_shape}")
    print(f"  LSTM Units: {lstm_units}, Dropout Rate: {dropout_rate}")
    print(f"  Learning Rate: {learning_rate}")

    model = Sequential([
        Input(shape=input_shape, name="input_layer"),
        LSTM(lstm_units, return_sequences=False, name="lstm_layer"), # False as it's the last LSTM layer before Dense
        Dropout(dropout_rate, name="dropout_layer"),
        Dense(lstm_units // 2, activation='relu', name="dense_hidden_layer"), # Optional hidden dense layer
        Dense(1, name="output_layer") # Output layer for single-step regression forecast
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error') # Common loss for regression
    model.summary(print_fn=lambda x: print(x)) # Print summary to console
    return model

# --- LSTM Training and Evaluation ---
def train_evaluate_lstm(
    df_series_for_lstm: pd.DataFrame, # DataFrame for a single store-dept
    target_col_name: str,
    feature_cols_to_scale: list, # All columns (features + target) to be scaled together by feature_scaler
    sequence_length: int,
    model_hyperparams: dict, # Includes lstm_units, dropout_rate, learning_rate, epochs, batch_size, patience
    test_size_ratio: float = 0.2,
    store_id: int = 0,
    dept_id: int = 0
) -> tuple[dict, dict | None, tf.keras.Model | None, MinMaxScaler | None, MinMaxScaler | None, str, str]:
    """
    Preprocesses data, defines, trains, and evaluates an LSTM model for a specific time series.

    This function handles:
    1.  Train-test splitting of the input DataFrame.
    2.  Scaling of features:
        - `feature_scaler`: Fits on all `feature_cols_to_scale` from the training set and transforms
          both train and test sets. This scaler is saved.
        - `target_scaler`: Fits *only* on the unscaled target column from the training set.
          This scaler is saved and used to inverse-transform predictions to their original scale.
    3.  Creation of input sequences (X) and target sequences (y) for LSTM.
    4.  Definition and compilation of the LSTM model.
    5.  Model training with early stopping and learning rate reduction callbacks.
    6.  Prediction on the test set and inverse scaling of predictions.
    7.  Calculation of evaluation metrics (MAE, RMSE, MAPE).
    8.  Saving the trained Keras model.
    9.  Returning a dictionary of logged parameters (including artifact paths), metrics,
        the model object, scalers, and train/test period strings.

    Args:
        df_series_for_lstm: DataFrame for a single store-department, sorted by date.
                            Must contain 'Date' and all columns listed in `feature_cols_to_scale`.
        target_col_name: Name of the target variable column (e.g., 'Weekly_Sales').
        feature_cols_to_scale: List of all column names (including the target column)
                               that will be scaled together by the `feature_scaler`.
        sequence_length: Length of input sequences for the LSTM.
        model_hyperparams: Dictionary of hyperparameters for the LSTM model and training
                           (e.g., 'lstm_units', 'dropout_rate', 'epochs', 'batch_size').
        test_size_ratio: Proportion of data to use for the test set. Defaults to 0.2.
        store_id: Store ID for logging and artifact naming.
        dept_id: Department ID for logging and artifact naming.

    Returns:
        tuple: (logged_params_dict, metrics_dict, trained_model, feature_scaler_obj, target_scaler_obj,
                train_period_string, test_period_string)
               Returns (hyperparams, None, None, None, None, "Error", "Error") on critical failure.
    """
    model_name_for_log = f"LSTM_S{store_id}D{dept_id}"
    print(f"\n--- Training LSTM for Store {store_id}, Dept {dept_id} using features: {feature_cols_to_scale} ---")

    if target_col_name not in feature_cols_to_scale: # Critical check
        error_msg = f"Target column '{target_col_name}' must be included in 'feature_cols_to_scale' for correct processing."
        print(f"Error: {error_msg}")
        model_hyperparams['error'] = error_msg
        return model_hyperparams, None, None, None, None, "Error: Target not in features", "Error: Target not in features"

    # Prepare data for scaling (all specified feature columns)
    data_for_scaling = df_series_for_lstm[feature_cols_to_scale].values

    # Train-test split based on ratio
    num_samples = len(data_for_scaling)
    num_test_samples = int(num_samples * test_size_ratio)
    num_train_samples = num_samples - num_test_samples

    if num_train_samples < sequence_length * 2 or num_test_samples < 1: # Need enough data for sequences and testing
        error_msg = (f"Not enough data for train/test split and sequence creation. "
                     f"Train samples: {num_train_samples}, Test samples: {num_test_samples}, Seq_len: {sequence_length}")
        print(error_msg)
        model_hyperparams['error'] = error_msg
        return model_hyperparams, None, None, None, None, "Error: Data split", "Error: Data split"

    train_unscaled_all_ft = data_for_scaling[:num_train_samples]
    test_unscaled_all_ft = data_for_scaling[num_train_samples:]

    train_dates = df_series_for_lstm['Date'].iloc[:num_train_samples]
    test_dates = df_series_for_lstm['Date'].iloc[num_train_samples:]
    train_period_str = f"{train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')}"
    test_period_str = f"{test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')}"

    # Scaler for all features (including target, as it's part of feature_cols_to_scale)
    feature_scaler_instance = MinMaxScaler(feature_range=(0, 1))
    train_scaled_all_ft = feature_scaler_instance.fit_transform(train_unscaled_all_ft)
    test_scaled_all_ft = feature_scaler_instance.transform(test_unscaled_all_ft)

    # Separate scaler specifically for the target variable (for inverse transformation of predictions)
    target_column_index = feature_cols_to_scale.index(target_col_name)
    target_scaler_instance = MinMaxScaler(feature_range=(0, 1))
    # Fit target_scaler on the unscaled target column from the training data
    target_scaler_instance.fit(train_unscaled_all_ft[:, target_column_index].reshape(-1, 1))

    # Save scalers
    ts_now = datetime.now().strftime('%Y%m%d%H%M')
    fs_path = os.path.join(SCALER_STORE_PATH, f"feature_scaler_S{store_id}D{dept_id}_{ts_now}.joblib")
    ts_path = os.path.join(SCALER_STORE_PATH, f"target_scaler_S{store_id}D{dept_id}_{ts_now}.joblib")
    try:
        joblib.dump(feature_scaler_instance, fs_path)
        joblib.dump(target_scaler_instance, ts_path)
        print(f"Saved feature_scaler to: {fs_path}")
        print(f"Saved target_scaler to: {ts_path}")
    except Exception as e_scaler_save: # pragma: no cover
        print(f"Error saving scalers: {e_scaler_save}")
        fs_path, ts_path = f"Error: {e_scaler_save}", f"Error: {e_scaler_save}"


    # Prepare sequences: X uses all scaled features, y uses the scaled target column
    # The LSTM will predict the target column in its feature_scaler-transformed scale.
    X_train_seq, y_train_seq_target_scaled = _create_sequences_for_lstm(
        train_scaled_all_ft, # Features for X sequences
        train_scaled_all_ft[:, target_column_index], # Target column (already scaled by feature_scaler) for y sequences
        sequence_length
    )
    # For evaluation, we need y_test in original scale and also scaled for validation_data
    X_test_seq, y_test_seq_target_unscaled = _create_sequences_for_lstm(
        test_scaled_all_ft, # Features for X sequences
        test_unscaled_all_ft[:, target_column_index], # UNscaled target for final metric calculation
        sequence_length
    )
    _, y_test_seq_target_scaled_for_val = _create_sequences_for_lstm( # Scaled target for val_loss
        test_scaled_all_ft,
        test_scaled_all_ft[:, target_column_index], # Target column (scaled by feature_scaler)
        sequence_length
    )


    if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
        error_msg = (f"Not enough data to create sequences after train/test split. "
                     f"X_train_seq shape: {X_train_seq.shape}, X_test_seq shape: {X_test_seq.shape}")
        print(error_msg)
        model_hyperparams['error'] = error_msg
        return model_hyperparams, None, None, feature_scaler_instance, target_scaler_instance, train_period_str, test_period_str

    print(f"Sequence shapes: X_train: {X_train_seq.shape}, y_train_target_scaled: {y_train_seq_target_scaled.shape}")
    print(f"                 X_test: {X_test_seq.shape}, y_test_target_unscaled: {y_test_seq_target_unscaled.shape}")

    # Define and train LSTM model
    input_lstm_shape = (X_train_seq.shape[1], X_train_seq.shape[2]) # (sequence_length, n_features)
    lstm_model = _define_lstm_model(
        input_lstm_shape,
        lstm_units=model_hyperparams.get('lstm_units', 64),
        dropout_rate=model_hyperparams.get('dropout_rate', 0.2),
        learning_rate=model_hyperparams.get('learning_rate', 0.001)
    )
    # Callbacks
    early_stopping_cb = EarlyStopping(
        monitor='val_loss', patience=model_hyperparams.get('patience', 10),
        restore_best_weights=True, verbose=1
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=model_hyperparams.get('lr_patience', 5),
        min_lr=model_hyperparams.get('min_lr', 0.00001), verbose=1
    )

    print("Starting LSTM model training...")
    training_start_time = datetime.now()
    history = lstm_model.fit(
        X_train_seq, y_train_seq_target_scaled,
        epochs=model_hyperparams.get('epochs', 50),
        batch_size=model_hyperparams.get('batch_size', 32),
        validation_data=(X_test_seq, y_test_seq_target_scaled_for_val), # Use scaled y for val_loss
        callbacks=[early_stopping_cb, reduce_lr_cb],
        verbose=1 # Or 2 for one line per epoch, 0 for silent
    )
    print(f"LSTM training completed in: {datetime.now() - training_start_time}")

    # Evaluate model: predictions are scaled, need inverse transform using target_scaler
    y_pred_test_scaled = lstm_model.predict(X_test_seq)
    # y_pred_test_scaled is the LSTM's output. It's the target column scaled by feature_scaler.
    # To correctly inverse_transform with target_scaler, y_pred_test_scaled must be
    # in the same "form" that target_scaler was fit on (i.e., only the target column's values).
    # The current `target_scaler` was fit on the original unscaled target values.
    # The LSTM output (`y_pred_test_scaled`) is in the scale of `feature_scaler` applied to the target.
    # This setup implies `feature_scaler` applied to the target column results in the same
    # transformation as `target_scaler` applied to the target column. This holds for MinMaxScaler
    # if the target column's min/max relative to itself is the same as its min/max relative to
    # the feature set it was part of.
    # For simplicity and common practice, we assume this holds or is close enough.
    y_pred_test_original_scale = target_scaler_instance.inverse_transform(y_pred_test_scaled).flatten()


    # Trim arrays to the minimum length in case sequence creation resulted in slight mismatches for y_test
    min_eval_len = min(len(y_pred_test_original_scale), len(y_test_seq_target_unscaled))
    if min_eval_len == 0:
        print("Error: Zero length arrays for metric calculation after trimming. Check sequence creation output.")
        metrics_calculated = None
    else:
        y_pred_eval = y_pred_test_original_scale[:min_eval_len]
        y_true_eval = y_test_seq_target_unscaled[:min_eval_len]
        metrics_calculated = _calculate_dl_metrics(y_true_eval, y_pred_eval)
    print(f"{model_name_for_log} Test Metrics (original scale): {metrics_calculated}")

    # Save Keras model
    model_keras_filename = f"{model_name_for_log}_{datetime.now().strftime('%Y%m%d%H%M%S')}.keras"
    model_final_save_path = os.path.join(MODEL_STORE_PATH, model_keras_filename)
    try:
        lstm_model.save(model_final_save_path)
        print(f"Saved Keras LSTM model to: {model_final_save_path}")
    except Exception as e_model_save: # pragma: no cover
        model_final_save_path = f"Error saving Keras model: {e_model_save}"
        print(model_final_save_path)

    # Prepare parameters for logging
    params_for_logging = model_hyperparams.copy()
    params_for_logging.update({
        'sequence_length': sequence_length,
        'input_shape_logged': str(input_lstm_shape),
        'model_path': model_final_save_path,
        'feature_scaler_path': fs_path,
        'target_scaler_path': ts_path,
        'actual_epochs_trained': len(history.history.get('loss', [])),
        'features_used_for_lstm_input': feature_cols_to_scale, # List of columns that went into feature_scaler
        'train_period_in_params': train_period_str, # Store train/test period with params
        'test_period_in_params': test_period_str
    })

    return params_for_logging, metrics_calculated, lstm_model, feature_scaler_instance, target_scaler_instance, train_period_str, test_period_str

# --- Main Orchestration Function ---
def main_deep_learning_pipeline():
    """
    Main function to orchestrate the deep learning (LSTM) modeling pipeline.

    Loads featured data, defines features for LSTM, iterates through selected
    Store-Department combinations, trains and evaluates LSTM models,
    and logs the experiment results.
    """
    print("--- Starting Deep Learning (LSTM) Time Series Modeling Pipeline ---")
    # Set seeds for reproducibility
    np.random.seed(42)
    if tf: # Check if TensorFlow was imported successfully
        tf.random.set_seed(42)
        print(f"TensorFlow version: {tf.__version__}")
    else: # pragma: no cover
        print("TensorFlow not available. LSTM modeling cannot proceed.")
        return

    warnings.filterwarnings("ignore", category=UserWarning) # Suppress some common warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    df_featured_full = _load_featured_data_dl()
    if df_featured_full is None:
        print("Failed to load featured data. Exiting Deep Learning pipeline.")
        return

    # Example: Model for a specific Store-Department pair
    # In a full pipeline, you would iterate over many such pairs.
    store_dept_combinations_to_model = [(1, 1)] # Example: Store 1, Dept 1

    # Define the set of features to be used by the LSTM.
    # The target 'Weekly_Sales' MUST be included here for the current scaling strategy.
    base_lstm_feature_columns = [
        'Weekly_Sales', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Size',
        'Year', 'Month_sin', 'Month_cos', 'WeekOfYear_sin', 'WeekOfYear_cos',
        'Sales_Lag_1', 'Sales_Lag_4', 'Sales_Lag_12', 'Sales_Lag_52', # Key lags
        'Sales_Roll_Mean_4', 'Sales_Roll_Std_4', # Short-term rolling stats
        'Sales_Roll_Mean_12', 'Sales_Roll_Std_12' # Medium-term rolling stats
    ]
    # Dynamically add 'Type_Encoded' if available, or create it
    if 'Type_Encoded' in df_featured_full.columns:
        if 'Type_Encoded' not in base_lstm_feature_columns: # Avoid duplicates
             base_lstm_feature_columns.append('Type_Encoded')
    elif 'Type' in df_featured_full.columns: # If only 'Type' exists, encode it
        print("Label encoding 'Type' column as 'Type_Encoded' for LSTM features...")
        from sklearn.preprocessing import LabelEncoder # Local import is fine
        le_type_dl = LabelEncoder()
        df_featured_full['Type_Encoded'] = le_type_dl.fit_transform(df_featured_full['Type'].astype(str))
        if 'Type_Encoded' not in base_lstm_feature_columns:
            base_lstm_feature_columns.append('Type_Encoded')

    target_col = 'Weekly_Sales'

    for store_id_iter, dept_id_iter in store_dept_combinations_to_model:
        print(f"\n--- Processing LSTM for Store: {store_id_iter}, Dept: {dept_id_iter} ---")
        current_series_df = df_featured_full[
            (df_featured_full['Store'] == store_id_iter) & (df_featured_full['Dept'] == dept_id_iter)
        ].copy()

        # Ensure all selected base_lstm_feature_columns are present in this specific series_df
        # and that the target column is definitely among them.
        actual_lstm_features_for_series = [col for col in base_lstm_feature_columns if col in current_series_df.columns]
        if target_col not in actual_lstm_features_for_series:
            if target_col in current_series_df.columns: # Should generally be true if in base_lstm_feature_columns
                actual_lstm_features_for_series.insert(0, target_col) # Ensure target is present for scaling
                actual_lstm_features_for_series = sorted(list(set(actual_lstm_features_for_series))) # Deduplicate and sort
            else:
                print(f"Target column '{target_col}' is missing for Store {store_id_iter}-Dept {dept_id_iter}. Skipping LSTM for this series.")
                continue
        
        # Prepare data for this specific LSTM model (Date + selected features)
        data_for_lstm_model = current_series_df[['Date'] + actual_lstm_features_for_series].copy()
        # Drop rows where ANY of the selected LSTM features are NaN (esp. due to lags/rolling)
        data_for_lstm_model.dropna(subset=actual_lstm_features_for_series, inplace=True)
        data_for_lstm_model.reset_index(drop=True, inplace=True) # Crucial for consistent iloc indexing later

        # Check if enough data remains after NaN removal for sequence creation and splitting
        min_data_len_for_lstm = 60 # Arbitrary minimum, e.g., ~1 year weekly data + sequence_length
        if data_for_lstm_model.empty or len(data_for_lstm_model) < min_data_len_for_lstm:
            print(f"Not enough valid data rows ({len(data_for_lstm_model)}) after NaN drop for Store {store_id_iter}-Dept {dept_id_iter}. "
                  f"Required at least {min_data_len_for_lstm}. Skipping LSTM for this series.")
            continue

        current_run_timestamp_dl = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        seq_len = 12 # Example sequence length (e.g., 12 weeks)
        lstm_config_params = { # Hyperparameters for the LSTM model
            'lstm_units': 64, 'dropout_rate': 0.25, 'epochs': 50, # Example: reduce epochs for faster runs
            'batch_size': 32, 'learning_rate': 0.001,
            'patience': 10, 'lr_patience': 5, 'min_lr': 1e-6
        }

        # Train, evaluate, and get logged parameters and metrics
        logged_params_lstm, metrics_lstm, _, _, _, train_period_log_str, test_period_log_str = train_evaluate_lstm(
            df_series_for_lstm=data_for_lstm_model,
            target_col_name=target_col,
            feature_cols_to_scale=actual_lstm_features_for_series,
            sequence_length=seq_len,
            model_hyperparams=lstm_config_params,
            test_size_ratio=0.2, # Use 20% of the series for testing
            store_id=store_id_iter,
            dept_id=dept_id_iter
        )

        if logged_params_lstm: # Check if training produced parameters (even if metrics are None)
            log_dl_experiment(
                store_id=store_id_iter,
                dept_id=dept_id_iter,
                model_name=f'LSTM_S{store_id_iter}D{dept_id_iter}', # Consistent naming
                params=logged_params_lstm, # This dict contains model_path, scalers, sequence_length, etc.
                metrics=metrics_lstm,      # This can be None if evaluation failed
                train_period_str=train_period_log_str, # Passed back from train_evaluate_lstm
                test_period_str=test_period_log_str,   # Passed back from train_evaluate_lstm
                # features_used_for_lstm is now part of logged_params_lstm
                run_timestamp=current_run_timestamp_dl
            )

    save_dl_experiment_log()
    print("\n--- Deep Learning (LSTM) Time Series Modeling Pipeline Finished ---")

if __name__ == '__main__':
    main_deep_learning_pipeline()