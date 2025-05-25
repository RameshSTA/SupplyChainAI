"""
Trains and evaluates classical machine learning regression models for demand forecasting.

This script includes functions for:
- Setting up project paths and importing necessary modules.
- Logging experiment details (parameters, metrics, model paths).
- Loading pre-featured sales data.
- Calculating regression evaluation metrics (MAE, RMSE, MAPE).
- Performing time-based train-test splits.
- Training, evaluating, and saving Random Forest, XGBoost, and LightGBM regressors.
- A main pipeline function to orchestrate the model training and logging process.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
import joblib  # For saving/loading scikit-learn compatible models
import sys

# Machine Learning Regression Models
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Project Path Setup & Utility Import ---
_PROJECT_ROOT_REG = None
_PREPROCESSING_FUNC_LOADED = False

def _get_project_root_classical_reg() -> str:
    """Determines project root assuming this script is in a nested structure."""
    try:
        # Assumes file is at: PROJECT_ROOT/src/models/demand_forecasting/classical_regression.py
        # Thus, PROJECT_ROOT is four levels up from this file's directory.
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
        return project_root
    except NameError:  # __file__ is not defined (e.g., interactive environment)
        # Fallback: Assume CWD is project root or a common dev location like 'notebooks' or 'src'.
        # Heuristic: if 'src' exists in CWD, CWD is likely project root.
        # If CWD is 'src', try parent.
        cwd = os.getcwd()
        if os.path.exists(os.path.join(cwd, 'src')): # CWD is likely project root
            return cwd
        elif os.path.basename(cwd) == 'src': # CWD is src
            return os.path.dirname(cwd)
        # Default to CWD with a warning if unsure.
        print(
            f"Warning: `__file__` not defined. Using CWD '{cwd}' as PROJECT_ROOT. "
            "Ensure this is correct for data and utility paths."
        )
        return cwd

_PROJECT_ROOT_REG = _get_project_root_classical_reg()

# Attempt to import preprocess_regression_features
try:
    SRC_PATH_REG = os.path.join(_PROJECT_ROOT_REG, 'src')
    if SRC_PATH_REG not in sys.path:
        sys.path.insert(0, SRC_PATH_REG)
    from utils.preprocessing import preprocess_regression_features
    _PREPROCESSING_FUNC_LOADED = True
    print("Successfully imported 'preprocess_regression_features'.")
except ImportError as e:
    print(f"ERROR importing 'preprocess_regression_features': {e}. This script may not function correctly.")
    # Define a dummy function if import fails, to prevent immediate crashes later
    def preprocess_regression_features(*args, **kwargs):
        """Dummy function: Actual 'preprocess_regression_features' from utils failed to import."""
        print("CRITICAL ERROR: 'preprocess_regression_features' is NOT available.")
        return None, None, None, None

# --- Configuration Paths ---
PROCESSED_DATA_PATH = os.path.join(_PROJECT_ROOT_REG, 'data', 'processed')
FEATURED_DATA_FILENAME = 'walmart_data_featured.parquet'
REPORTS_PATH = os.path.join(_PROJECT_ROOT_REG, 'reports')
EXPERIMENT_LOGS_PATH = os.path.join(REPORTS_PATH, 'experiment_logs')
MODEL_STORE_PATH = os.path.join(_PROJECT_ROOT_REG, 'models_store', 'demand_forecasting')

# Ensure necessary directories exist
os.makedirs(EXPERIMENT_LOGS_PATH, exist_ok=True)
os.makedirs(MODEL_STORE_PATH, exist_ok=True)

# --- Experiment Logging ---
_REGRESSION_EXPERIMENT_LOG_FILE = os.path.join(EXPERIMENT_LOGS_PATH, 'classical_regression_experiments.csv')
_regression_experiment_records = [] # Module-level list to store records before saving

def log_regression_experiment(
    model_name: str,
    params_dict_to_log: dict,
    metrics: dict,
    train_period_str: str,
    test_period_str: str,
    features_used: list,
    store_id: any, # int or str
    dept_id: any,  # int or str
    run_timestamp: str | None = None
):
    """
    Logs the details of a single regression model training experiment.

    Args:
        model_name: Name of the model (e.g., "RandomForestRegressor").
        params_dict_to_log: Dictionary of parameters used for the model,
                              including model-specific hyperparameters and 'model_path'.
        metrics: Dictionary of evaluation metrics (e.g., {'MAE': val, 'RMSE': val}).
        train_period_str: String representation of the training period (e.g., "YYYY-MM-DD to YYYY-MM-DD").
        test_period_str: String representation of the test period.
        features_used: List of feature names used for training.
        store_id: Identifier for the store (e.g., 0 for global models).
        dept_id: Identifier for the department (e.g., 0 for global models).
        run_timestamp: Timestamp of the run; defaults to current time if None.
    """
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Consolidate all relevant info into the 'Parameters' field for logging robustness
    # while also keeping key fields separate for easier querying.
    final_params_for_str_logging = params_dict_to_log.copy() # Ensure original is not modified
    final_params_for_str_logging['train_period_logged_in_params'] = train_period_str
    final_params_for_str_logging['test_period_logged_in_params'] = test_period_str
    final_params_for_str_logging['features_used_logged_in_params'] = features_used # Redundant but explicit

    record = {
        'Timestamp': run_timestamp,
        'Store': store_id,
        'Dept': dept_id,
        'Model': model_name,
        'Parameters': str(final_params_for_str_logging), # String representation of the full parameters dict
        'Train_Period': train_period_str,               # Also as a separate, easily queryable column
        'Test_Period': test_period_str,                # Also as a separate, easily queryable column
        'Features_Count': len(features_used),
        'Features_List_Used': str(features_used)       # String representation of the feature list
    }
    record.update(metrics) # Add MAE, RMSE, MAPE etc.
    _regression_experiment_records.append(record)
    print(f"Logged experiment for {model_name} (Store: {store_id}, Dept: {dept_id})")

def save_regression_experiment_log():
    """Saves all accumulated regression experiment records to a CSV file."""
    if not _regression_experiment_records:
        print("No new regression experiments were logged to save.")
        return

    log_df = pd.DataFrame(_regression_experiment_records)
    try:
        # Append if log file already exists and is not empty, otherwise write with header
        if os.path.exists(_REGRESSION_EXPERIMENT_LOG_FILE) and os.path.getsize(_REGRESSION_EXPERIMENT_LOG_FILE) > 0:
            log_df.to_csv(_REGRESSION_EXPERIMENT_LOG_FILE, mode='a', header=False, index=False)
            print(f"Appended {len(_regression_experiment_records)} new experiments to '{_REGRESSION_EXPERIMENT_LOG_FILE}'")
        else:
            log_df.to_csv(_REGRESSION_EXPERIMENT_LOG_FILE, mode='w', header=True, index=False)
            print(f"Saved {len(_regression_experiment_records)} new experiments to '{_REGRESSION_EXPERIMENT_LOG_FILE}'")
        _regression_experiment_records.clear() # Clear records after saving
    except Exception as e:
        print(f"Error saving regression experiment log to '{_REGRESSION_EXPERIMENT_LOG_FILE}': {e}")

def _load_featured_data(file_path: str = os.path.join(PROCESSED_DATA_PATH, FEATURED_DATA_FILENAME)) -> pd.DataFrame | None:
    """
    Loads the preprocessed and feature-engineered Walmart dataset.

    Supports Parquet and CSV formats. Ensures 'Date' column is datetime.

    Args:
        file_path (str, optional): Full path to the featured data file.
            Defaults to the path constructed from `PROCESSED_DATA_PATH` and
            `FEATURED_DATA_FILENAME`.

    Returns:
        pd.DataFrame | None: Loaded DataFrame, or None if loading fails.
    """
    print(f"--- Loading featured data from: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: Featured data file not found at '{file_path}'")
        # Attempt CSV fallback if Parquet was specified
        if file_path.lower().endswith('.parquet'):
            csv_fallback_path = file_path[:-len('.parquet')] + '.csv'
            if os.path.exists(csv_fallback_path):
                print(f"Parquet file not found. Attempting to load CSV fallback: '{csv_fallback_path}'")
                file_path = csv_fallback_path # Update file_path to use CSV
            else:
                print(f"Neither Parquet ('{file_path}') nor CSV fallback ('{csv_fallback_path}') found.")
                return None
        else: # If not a parquet file initially and not found
            return None

    try:
        if file_path.lower().endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, parse_dates=['Date'])
        else:
            print(f"Error: Unsupported file format for '{file_path}'. Please use .parquet or .csv.")
            return None

        print(f"Successfully loaded featured data from '{file_path}'. Shape: {df.shape}")
        # Ensure Date column is datetime type
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Error loading featured data from '{file_path}': {e}")
        return None

def _calculate_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calculates MAE, RMSE, and MAPE for regression model evaluation.

    Handles potential division by zero in MAPE by replacing zeros in y_true
    with NaN before calculation, then using nanmean.

    Args:
        y_true: Series of actual target values.
        y_pred: Series of predicted target values.

    Returns:
        dict: A dictionary containing 'MAE', 'RMSE', and 'MAPE' scores.
              MAPE is returned as a percentage.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # For MAPE, handle cases where y_true is zero to avoid division by zero.
    y_true_mape, y_pred_mape = y_true.copy(), y_pred.copy()
    zero_mask = (y_true_mape == 0)
    y_true_mape[zero_mask] = np.nan # Replace 0s with NaN to exclude them from mean
    y_pred_mape[zero_mask] = np.nan # Corresponding predictions also become NaN for consistency

    if np.all(np.isnan(y_true_mape)): # If all true values were zero (or became NaN)
        mape = np.nan
    else:
        mape = np.nanmean(np.abs((y_true_mape - y_pred_mape) / y_true_mape)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def _time_based_split_regression(
    X: pd.DataFrame, y: pd.Series, dates: pd.Series, test_duration_weeks: int = 52
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series] | tuple[None]*6 :
    """
    Splits features (X), target (y), and dates into training and testing sets
    based on a specified time duration for the test set.

    Aligns indices of X, y, and dates before splitting if they do not match but have equal lengths.

    Args:
        X: DataFrame of features.
        y: Series of target variable.
        dates: Series of datetime objects corresponding to X and y.
        test_duration_weeks: Duration of the test set in weeks, counted
                             from the last available date. Defaults to 52.

    Returns:
        A tuple (X_train, X_test, y_train, y_test, train_dates, test_dates).
        Returns a tuple of six Nones if inputs are invalid or split results in empty sets.

    Raises:
        ValueError: If input data is None, empty, has mismatched lengths after attempting alignment,
                    or if the split results in empty train/test sets.
    """
    if X is None or y is None or dates is None or X.empty or y.empty or dates.empty:
        print("Error: Input X, y, or dates is None or empty for time-based split.")
        return (None,) * 6 # type: ignore

    # Attempt to align indices if they don't match but lengths are equal
    if not (X.index.equals(y.index) and X.index.equals(dates.index)):
        print("Warning: Indices of X, y, and dates do not match. Attempting to align by resetting index.")
        if len(X) == len(y) == len(dates):
            common_idx = pd.RangeIndex(start=0, stop=len(X), step=1)
            X = X.reset_index(drop=True).set_index(common_idx)
            y = y.reset_index(drop=True).set_index(common_idx)
            dates = dates.reset_index(drop=True).set_index(common_idx)
            print("Indices successfully reset and aligned for X, y, and dates.")
        else:
            print("Error: Lengths of X, y, and dates are different. Cannot align indices.")
            return (None,) * 6 # type: ignore

    if not pd.api.types.is_datetime64_any_dtype(dates):
        print("Error: 'dates' series must be of datetime type for time-based split.")
        return (None,) * 6 # type: ignore

    last_date = dates.max()
    split_date = last_date - pd.Timedelta(weeks=test_duration_weeks)

    train_mask = (dates <= split_date)
    test_mask = (dates > split_date)

    if not train_mask.any() or not test_mask.any():
        print(f"Error: Train or test set is empty after time split. "
              f"Last date: {last_date}, Split date: {split_date}. "
              "Check 'test_duration_weeks' and data date range.")
        return (None,) * 6 # type: ignore

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    train_dates_out, test_dates_out = dates.loc[train_mask], dates.loc[test_mask]

    print(f"Time-based split complete:")
    print(f"  Train period: {train_dates_out.min().date()} to {train_dates_out.max().date()} ({len(X_train)} samples)")
    print(f"  Test period:  {test_dates_out.min().date()} to {test_dates_out.max().date()} ({len(X_test)} samples)")
    return X_train, X_test, y_train, y_test, train_dates_out, test_dates_out


# --- Model Training & Evaluation Wrappers ---
def _train_evaluate_model(
    model_instance, # Already initialized model object
    model_name_base: str,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    model_params_for_log: dict, # Hyperparameters used for training
    model_name_suffix: str = ""
) -> tuple[dict, dict | None, object | None]:
    """
    Generic internal helper to train, evaluate, and save a given model instance.
    Handles different saving mechanisms based on model type.
    """
    model_full_name = f"{model_name_base}{model_name_suffix}"
    print(f"\n--- Training and evaluating {model_full_name} ---")
    print(f"Using parameters: {model_params_for_log}")

    try:
        start_time = datetime.now()
        if isinstance(model_instance, xgb.XGBRegressor):
            model_instance.fit(X_train, y_train, verbose=False)
        else: # For sklearn-compatible models (RandomForest, LightGBM)
            model_instance.fit(X_train, y_train)
        print(f"Model fitting completed in: {datetime.now() - start_time}")

        y_pred_test = model_instance.predict(X_test)
        metrics = _calculate_regression_metrics(y_test, y_pred_test)
        print(f"{model_full_name} Test Metrics: {metrics}")

        # Determine filename and save model
        timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
        model_filename_base = model_full_name.replace(' ', '_').replace('(', '').replace(')', '') # Clean name

        if isinstance(model_instance, RandomForestRegressor):
            model_filename = f"{model_filename_base}_{timestamp_str}.joblib"
            model_save_path = os.path.join(MODEL_STORE_PATH, model_filename)
            joblib.dump(model_instance, model_save_path)
        elif isinstance(model_instance, xgb.XGBRegressor):
            model_filename = f"{model_filename_base}_{timestamp_str}.json"
            model_save_path = os.path.join(MODEL_STORE_PATH, model_filename)
            model_instance.save_model(model_save_path)
        elif isinstance(model_instance, lgb.LGBMRegressor):
            model_filename = f"{model_filename_base}_{timestamp_str}.txt"
            model_save_path = os.path.join(MODEL_STORE_PATH, model_filename)
            model_instance.booster_.save_model(model_save_path) # Save booster for LightGBM
        else:
            print(f"Warning: Unknown model type '{type(model_instance)}'. Model not saved.")
            model_save_path = None

        if model_save_path:
            print(f"Saved trained model to: {model_save_path}")

        # Prepare parameters dictionary for logging (includes actual hyperparams and model_path)
        logged_params = model_params_for_log.copy()
        if model_save_path:
            logged_params['model_path'] = model_save_path
        return logged_params, metrics, model_instance

    except Exception as e:
        print(f"Error during {model_full_name} training or evaluation: {e}")
        # Return original params for logging, even on error, to record the attempt
        return model_params_for_log, None, None


def train_evaluate_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    model_params: dict | None = None,
    model_name_suffix: str = ""
) -> tuple[dict, dict | None, RandomForestRegressor | None]:
    """
    Initializes, trains, evaluates, and saves a Random Forest Regressor model.

    Args:
        X_train, y_train: Training features and target.
        X_test, y_test: Test features and target.
        model_params (dict, optional): Hyperparameters for RandomForestRegressor.
                                       If None, default parameters are used.
        model_name_suffix (str, optional): Suffix to append to model name for logging/saving.

    Returns:
        tuple: (logged_params_dict, metrics_dict, trained_model_object)
               Returns (params, None, None) on error.
    """
    default_rf_params = {
        'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 10,
        'min_samples_leaf': 5, 'random_state': 42, 'n_jobs': -1
    }
    actual_params_used = model_params.copy() if model_params is not None else default_rf_params
    model = RandomForestRegressor(**actual_params_used)
    return _train_evaluate_model(model, "RandomForestRegressor", X_train, y_train, X_test, y_test,
                                 actual_params_used, model_name_suffix)

def train_evaluate_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    model_params: dict | None = None,
    model_name_suffix: str = ""
) -> tuple[dict, dict | None, xgb.XGBRegressor | None]:
    """
    Initializes, trains, evaluates, and saves an XGBoost Regressor model.
    (See train_evaluate_random_forest for Args/Returns structure)
    """
    default_xgb_params = {
        'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.1, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'objective': 'reg:squarederror',
        'random_state': 42, 'n_jobs': -1
    }
    actual_params_used = model_params.copy() if model_params is not None else default_xgb_params
    model = xgb.XGBRegressor(**actual_params_used)
    return _train_evaluate_model(model, "XGBoostRegressor", X_train, y_train, X_test, y_test,
                                 actual_params_used, model_name_suffix)

def train_evaluate_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    model_params: dict | None = None,
    model_name_suffix: str = ""
) -> tuple[dict, dict | None, lgb.LGBMRegressor | None]:
    """
    Initializes, trains, evaluates, and saves a LightGBM Regressor model.
    (See train_evaluate_random_forest for Args/Returns structure)
    """
    default_lgb_params = {
        'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.1, 'num_leaves': 31,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'objective': 'regression_l1', # or 'regression' for L2
        'metric': 'mae', # or 'rmse'
        'random_state': 42, 'n_jobs': -1, 'verbose': -1 # Suppress LightGBM verbosity
    }
    actual_params_used = model_params.copy() if model_params is not None else default_lgb_params
    model = lgb.LGBMRegressor(**actual_params_used)
    return _train_evaluate_model(model, "LightGBMRegressor", X_train, y_train, X_test, y_test,
                                 actual_params_used, model_name_suffix)


# --- Main Orchestration Function ---
def main_regression_pipeline():
    """
    Main function to orchestrate the classical regression modeling pipeline.

    This involves loading data, preprocessing, splitting data by time,
    training multiple regression models (Random Forest, XGBoost, LightGBM),
    evaluating them, and logging the experiment results.
    """
    print("--- Starting Classical Regression Modeling Pipeline ---")
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm") # Suppress specific LightGBM warnings
    warnings.filterwarnings("ignore", category=FutureWarning) # Suppress general future warnings

    if not _PREPROCESSING_FUNC_LOADED: # pragma: no cover
        print("CRITICAL: Preprocessing function is not loaded. Cannot proceed with modeling.")
        return

    df_featured = _load_featured_data()
    if df_featured is None:
        print("Exiting pipeline: Failed to load featured data.")
        return

    # Preprocess data for regression
    # is_training_phase=True ensures scalers are fit, etc.
    X, y, dates_for_split, final_feature_list = preprocess_regression_features(
        df_featured,
        target_col='Weekly_Sales',
        is_training_phase=True
    )

    if X is None or y is None or dates_for_split is None or not final_feature_list :
        print("Exiting pipeline: Preprocessing of features and target failed.")
        return

    # Time-based split
    split_results = _time_based_split_regression(X, y, dates_for_split, test_duration_weeks=52)
    if any(res is None for res in split_results): # Check if any element in tuple is None
        print("Exiting pipeline: Time-based data splitting failed.")
        return
    X_train, X_test, y_train, y_test, train_dates, test_dates = split_results

    train_period_str = f"{train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')}"
    test_period_str = f"{test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')}"

    current_run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    GLOBAL_MODEL_STORE_ID = 0  # Convention for models trained on all stores/depts
    GLOBAL_MODEL_DEPT_ID = 0

    # Define model configurations to train
    model_configs = {
        "RandomForest": {
            "train_func": train_evaluate_random_forest,
            "params": {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 40,
                       'min_samples_leaf': 15, 'max_features': 0.6, 'random_state': 42, 'n_jobs': -1}
        },
        "XGBoost": {
            "train_func": train_evaluate_xgboost,
            "params": {'n_estimators': 250, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.75,
                       'colsample_bytree': 0.75, 'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1}
        },
        "LightGBM": {
            "train_func": train_evaluate_lightgbm,
            "params": {'n_estimators': 250, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 2**6 -1,
                       'subsample': 0.75, 'colsample_bytree': 0.75, 'objective': 'regression_l1',
                       'metric': 'mae', 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
        }
    }

    for model_key, config in model_configs.items():
        print(f"\n--- Processing Model: {model_key} ---")
        logged_params, metrics, _ = config["train_func"](
            X_train, y_train, X_test, y_test, model_params=config["params"]
        )
        if metrics: # Only log if training and evaluation were successful
            log_regression_experiment(
                model_name=model_key + "Regressor", # e.g. RandomForestRegressor
                params_dict_to_log=logged_params,   # This dict already includes 'model_path'
                metrics=metrics,
                train_period_str=train_period_str,
                test_period_str=test_period_str,
                features_used=final_feature_list,
                store_id=GLOBAL_MODEL_STORE_ID,
                dept_id=GLOBAL_MODEL_DEPT_ID,
                run_timestamp=current_run_timestamp
            )

    save_regression_experiment_log()
    print("\n--- Classical Regression Modeling Pipeline Finished ---")

if __name__ == '__main__':
    main_regression_pipeline()