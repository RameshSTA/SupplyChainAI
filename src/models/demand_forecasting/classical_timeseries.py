"""
Trains and evaluates classical time series forecasting models.

This script includes functions for:
- Setting up project paths.
- Logging experiment details (parameters, metrics, model paths).
- Loading pre-featured sales data.
- Calculating forecasting evaluation metrics (MAE, RMSE, MAPE).
- Performing time-based train-test splits for time series.
- Training, evaluating, and saving Naive, Seasonal Naive, ETS (best-fitting),
  SARIMA (auto-selected), and Prophet models.
- A main pipeline function to orchestrate model training for selected store-department pairs.
"""
import pandas as pd
import numpy as np
import os
# import itertools # Not used in the provided script
from datetime import datetime
import warnings
import joblib # For saving/loading models like ETS, SARIMA, Prophet

# Time Series specific libraries
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima # For SARIMAX model selection
from prophet import Prophet      # For Prophet models

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Configuration: Define base paths ---
_PROJECT_ROOT_TS = None # Module-level variable

def _get_project_root_classical_ts() -> str:
    """
    Determines the project root directory.

    Assumes this script (`classical_timeseries.py`) is located within a nested
    structure like: `PROJECT_ROOT/src/models/demand_forecasting/`.
    It navigates up four levels from this file's location to find the project root.

    Returns:
        str: The absolute path to the project root directory.
    """
    try:
        # Assumes file is at: PROJECT_ROOT/src/models/demand_forecasting/classical_timeseries.py
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
        return project_root
    except NameError:  # __file__ is not defined (e.g., interactive environment)
        # Fallback: Try to infer from CWD, common if running notebooks or scripts from project subdirs
        cwd = os.getcwd()
        # Heuristic: if 'data' and 'src' folders exist in CWD, CWD is likely project root.
        if os.path.exists(os.path.join(cwd, 'data')) and os.path.exists(os.path.join(cwd, 'src')):
            return cwd
        # If CWD is 'src', 'notebooks', 'app' etc., try parent.
        parent_dir = os.path.abspath(os.path.join(cwd, '..'))
        if os.path.exists(os.path.join(parent_dir, 'data')) and os.path.exists(os.path.join(parent_dir, 'src')):
            return parent_dir
        # Default to CWD with a warning if still unsure.
        print(
            f"Warning: `__file__` not defined. Using CWD '{cwd}' as PROJECT_ROOT. "
            "Ensure this is correct for data and model paths."
        )
        return cwd

_PROJECT_ROOT_TS = _get_project_root_classical_ts()

PROCESSED_DATA_PATH = os.path.join(_PROJECT_ROOT_TS, 'data', 'processed')
FEATURED_DATA_FILENAME = 'walmart_data_featured.parquet' # Or your CSV equivalent
REPORTS_PATH = os.path.join(_PROJECT_ROOT_TS, 'reports')
EXPERIMENT_LOGS_PATH = os.path.join(REPORTS_PATH, 'experiment_logs')
MODEL_STORE_PATH = os.path.join(_PROJECT_ROOT_TS, 'models_store', 'demand_forecasting')

# Ensure necessary directories exist
os.makedirs(EXPERIMENT_LOGS_PATH, exist_ok=True)
os.makedirs(MODEL_STORE_PATH, exist_ok=True)

# --- Experiment Logging ---
_CLASSICAL_TS_EXPERIMENT_LOG_FILE = os.path.join(EXPERIMENT_LOGS_PATH, 'classical_timeseries_experiments.csv')
_classical_ts_experiment_records = [] # Module-level list for records

def log_ts_experiment(
    store_id: int,
    dept_id: int,
    model_name: str,
    params: dict | None,
    metrics: dict | None,
    train_period: str,
    test_period: str,
    run_timestamp: str | None = None
):
    """
    Logs the details of a single time series model training experiment.

    Args:
        store_id: Identifier for the store.
        dept_id: Identifier for the department.
        model_name: Name of the model (e.g., "ETS (Best)", "SARIMA (auto)").
        params: Dictionary of parameters used for the model. Can include 'model_path'.
        metrics: Dictionary of evaluation metrics (e.g., {'MAE': val, 'RMSE': val}).
        train_period: String representation of the training period (e.g., "YYYY-MM-DD to YYYY-MM-DD").
        test_period: String representation of the test period.
        run_timestamp: Timestamp of the run; defaults to current time if None.
    """
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    params_to_log = params if params is not None else {}
    record = {
        'Timestamp': run_timestamp,
        'Store': store_id,
        'Dept': dept_id,
        'Model': model_name,
        'Parameters': str(params_to_log), # String representation of parameters
        'Train_Period': train_period,
        'Test_Period': test_period,
    }
    # Add metrics, ensuring keys exist and handling None for metrics
    for metric_key in ['MAE', 'RMSE', 'MAPE']:
        record[metric_key] = metrics.get(metric_key, np.nan) if metrics else np.nan

    _classical_ts_experiment_records.append(record)
    print(f"Logged experiment for {model_name} (Store: {store_id}, Dept: {dept_id})")

def save_ts_experiment_log():
    """Saves all accumulated classical time series experiment records to a CSV file."""
    if not _classical_ts_experiment_records:
        print("No new classical time series experiments were logged to save.")
        return

    log_df = pd.DataFrame(_classical_ts_experiment_records)
    try:
        # Check if file exists and has content to determine if header is needed
        header_needed = not (
            os.path.exists(_CLASSICAL_TS_EXPERIMENT_LOG_FILE) and
            os.path.getsize(_CLASSICAL_TS_EXPERIMENT_LOG_FILE) > 0
        )
        log_df.to_csv(_CLASSICAL_TS_EXPERIMENT_LOG_FILE, mode='a', header=header_needed, index=False)
        print(f"Classical time series experiment log saved/appended to '{_CLASSICAL_TS_EXPERIMENT_LOG_FILE}'")
        _classical_ts_experiment_records.clear() # Clear records after successful save
    except Exception as e:
        print(f"Error saving classical time series experiment log: {e}")

# --- Data Loading ---
def _load_featured_data_ts(file_path: str = os.path.join(PROCESSED_DATA_PATH, FEATURED_DATA_FILENAME)) -> pd.DataFrame | None:
    """
    Loads the preprocessed and feature-engineered Walmart dataset for time series modeling.

    Supports Parquet and CSV formats. Ensures 'Date' column is datetime.

    Args:
        file_path (str, optional): Full path to the featured data file.
            Defaults to the path constructed from global configuration.

    Returns:
        pd.DataFrame | None: Loaded DataFrame, or None if loading fails.
    """
    print(f"--- Loading featured data for time series from: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: Featured data file not found at '{file_path}'")
        # Attempt CSV fallback if Parquet was specified
        if file_path.lower().endswith('.parquet'):
            csv_fallback_path = file_path[:-len('.parquet')] + '.csv'
            if os.path.exists(csv_fallback_path):
                print(f"Parquet file not found. Attempting to load CSV fallback: '{csv_fallback_path}'")
                file_path = csv_fallback_path
            else:
                print(f"Neither Parquet ('{file_path}') nor CSV fallback ('{csv_fallback_path}') found.")
                return None
        else:
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
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Error loading featured data from '{file_path}': {e}")
        return None

# --- Model Evaluation Metrics ---
def _calculate_ts_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calculates MAE, RMSE, and MAPE for time series forecast evaluation.

    Handles empty series, mismatched lengths, and NaNs robustly.
    For MAPE, it replaces zeros in y_true with NaN before calculation to avoid
    division by zero, then uses nanmean.

    Args:
        y_true: Series of actual target values.
        y_pred: Series of predicted target values.

    Returns:
        dict: A dictionary containing 'MAE', 'RMSE', and 'MAPE' scores.
              MAPE is returned as a percentage. Returns NaNs if metrics
              cannot be calculated.
    """
    if y_true.empty or y_pred.empty or len(y_true) != len(y_pred):
        print("Warning: y_true or y_pred is empty or lengths mismatch in _calculate_ts_metrics. Returning NaN metrics.")
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

    # Align and drop NaNs consistently based on y_true's original non-NaN values
    y_true_common_idx = y_true.dropna().index
    y_pred_aligned = y_pred.reindex(y_true_common_idx).dropna()
    y_true_aligned = y_true.reindex(y_pred_aligned.index) # Re-align y_true to y_pred's post-NaN drop index

    if y_true_aligned.empty or y_pred_aligned.empty: # Check after alignment and NaN drop
        print("Warning: No common valid data points between y_true and y_pred after NaN handling. Returning NaN metrics.")
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))

    # For MAPE, handle cases where y_true is zero
    y_true_mape, y_pred_mape = y_true_aligned.copy(), y_pred_aligned.copy()
    zero_mask_mape = (y_true_mape == 0)
    y_true_mape[zero_mask_mape] = np.nan # Exclude these from mean calculation
    y_pred_mape[zero_mask_mape] = np.nan

    if np.all(np.isnan(y_true_mape)): # If all true values were zero or became NaN
        mape = np.nan
    else:
        mape = np.nanmean(np.abs((y_true_mape - y_pred_mape) / y_true_mape)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# --- Time Series Splitting ---
def _time_series_split_strict(series: pd.Series, test_size: int = 52) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    """
    Splits a time series into training and testing sets.

    Args:
        series: The time series data (pandas Series with DatetimeIndex).
        test_size: The number of observations to include in the test set
                   (taken from the end of the series). Defaults to 52.

    Returns:
        tuple: (train_series, test_series), or (None, None) if split is not possible.
    """
    if not isinstance(series, pd.Series):
        print("Error: Input must be a pandas Series for time series split.")
        return None, None
    if len(series) <= test_size:
        print(f"Error: Test size ({test_size}) is too large for series length ({len(series)}). Cannot split.")
        return None, None
    return series[:-test_size], series[-test_size:]

# --- Model Training and Evaluation Functions ---
def _train_evaluate_naive(train_series: pd.Series, test_series: pd.Series) -> tuple[dict, dict | None]:
    """Generates a naive (last value) forecast and evaluates it."""
    model_name = "NaiveForecast"
    params_log = {"method": "last_value"}
    print(f"Evaluating {model_name}...")
    if train_series.empty:
        params_log["error"] = "Empty training series provided."
        return params_log, None
    # Naive forecast: last value of train series repeated for length of test series
    last_value = train_series.iloc[-1]
    predictions = pd.Series(last_value, index=test_series.index)
    try:
        metrics = _calculate_ts_metrics(test_series, predictions)
        return params_log, metrics
    except Exception as e: # pragma: no cover
        params_log["error_calculating_metrics"] = str(e)
        return params_log, None

def _train_evaluate_seasonal_naive(
    train_series: pd.Series, test_series: pd.Series, seasonal_period: int = 52
) -> tuple[dict, dict | None]:
    """Generates a seasonal naive forecast and evaluates it."""
    model_name = "SeasonalNaiveForecast"
    params_log = {"seasonal_period": seasonal_period, "method": "last_season_value"}
    print(f"Evaluating {model_name} (Period: {seasonal_period})...")

    if len(train_series) < seasonal_period:
        params_log["error"] = f"Training series too short (length {len(train_series)}) for seasonal period {seasonal_period}."
        return params_log, None

    # Seasonal Naive: repeat the value from the same period in the last season
    predictions_list = []
    for i in range(len(test_series)):
        # Index into train_series: len(train) - seasonal_period + (current_forecast_step % seasonal_period)
        idx_in_train = len(train_series) - seasonal_period + (i % seasonal_period)
        predictions_list.append(train_series.iloc[idx_in_train])
    predictions = pd.Series(predictions_list, index=test_series.index)

    try:
        metrics = _calculate_ts_metrics(test_series, predictions)
        return params_log, metrics
    except Exception as e: # pragma: no cover
        params_log["error_calculating_metrics"] = str(e)
        return params_log, None

def _train_evaluate_ets(
    train_series: pd.Series, test_series: pd.Series,
    seasonal_periods_val: int = 52, store_id: int = 0, dept_id: int = 0
) -> tuple[dict | None, dict | None]:
    """
    Trains various Exponential Smoothing (ETS) models, selects the best based on RMSE,
    evaluates it, and saves the best performing model.

    Args:
        train_series: Training time series data.
        test_series: Test time series data for evaluation.
        seasonal_periods_val: The seasonal period length (e.g., 52 for yearly seasonality with weekly data).
        store_id: Store ID for logging and model naming.
        dept_id: Department ID for logging and model naming.

    Returns:
        tuple: (best_model_parameters_dict, best_model_metrics_dict).
               Returns (error_dict, None) if no model converges or an error occurs.
    """
    model_base_name = "ETS (Best Fit)"
    print(f"Training and selecting best {model_base_name} for Store {store_id}-Dept {dept_id}...")

    if train_series.empty or len(train_series) < max(seasonal_periods_val * 2, 20): # Need enough data
        return {'error': 'Training series too short for ETS modeling.'}, None

    # Define ETS configurations to try: (trend, seasonal, damped_trend)
    ets_configurations = [
        ('add', 'add', False), ('add', 'add', True), ('add', 'mul', False), ('add', 'mul', True),
        ('add', None, False), ('add', None, True),  # Additive trend, no seasonality
        ('mul', 'mul', False), ('mul', 'mul', True), # Multiplicative trend and seasonality
        (None, 'add', False), (None, 'mul', False),  # No trend, with seasonality
        (None, None, False) # Simple Exponential Smoothing (no trend, no seasonality)
    ]
    best_model_fit, best_params_log, best_metrics_log, best_rmse_val = None, None, None, float('inf')
    series_has_non_positive = (train_series <= 0).any() # Check for non-positive values for multiplicative models

    for trend_cfg, seasonal_cfg, damped_cfg in ets_configurations:
        current_model_params = {
            'trend': trend_cfg, 'seasonal': seasonal_cfg, 'damped_trend': damped_cfg,
            'seasonal_periods': seasonal_periods_val if seasonal_cfg else None
        }
        # Skip multiplicative models if data contains non-positive values
        if (seasonal_cfg == 'mul' or trend_cfg == 'mul') and series_has_non_positive:
            continue

        try:
            ets_model = ExponentialSmoothing(
                train_series,
                trend=trend_cfg,
                seasonal=seasonal_cfg,
                damped_trend=damped_cfg,
                seasonal_periods=current_model_params['seasonal_periods'],
                initialization_method='heuristic' # Common robust initialization
            ).fit()

            predictions = pd.Series(ets_model.forecast(len(test_series)), index=test_series.index)
            metrics = _calculate_ts_metrics(test_series, predictions)

            if metrics and metrics.get('RMSE', float('inf')) < best_rmse_val:
                best_rmse_val = metrics['RMSE']
                best_metrics_log = metrics
                best_params_log = current_model_params # Store the config that led to this model
                best_model_fit = ets_model
        except Exception: # Broad exception to catch convergence errors, etc.
            # print(f"  ETS config {current_model_params} failed: {e_ets}") # Optional: for debugging
            continue # Try next configuration

    if best_model_fit and best_params_log and best_metrics_log:
        model_log_name_ets = f"ETS_Best_S{store_id}D{dept_id}"
        model_filename_ets = f"{model_log_name_ets}_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib"
        model_save_path_ets = os.path.join(MODEL_STORE_PATH, model_filename_ets)
        try:
            joblib.dump(best_model_fit, model_save_path_ets)
            best_params_log['model_path'] = model_save_path_ets # Add path to logged params
            print(f"Best ETS (S{store_id}D{dept_id}) - Params: {best_params_log}, Metrics: {best_metrics_log}")
        except Exception as e_save: # pragma: no cover
            best_params_log['model_path'] = f"Error saving model: {e_save}"
            print(f"Error saving best ETS model: {e_save}")
    else:
        print(f"No suitable ETS model converged or performed well for Store {store_id}-Dept {dept_id}.")
        best_params_log = best_params_log if best_params_log is not None else {}
        best_params_log.setdefault('error', 'No ETS model converged or met criteria.')

    return best_params_log, best_metrics_log


def _train_evaluate_sarima(
    train_series: pd.Series, test_series: pd.Series,
    train_exog: pd.DataFrame | None = None, test_exog: pd.DataFrame | None = None,
    seasonal_period_val: int = 52, store_id: int = 0, dept_id: int = 0
) -> tuple[dict | None, dict | None]:
    """
    Trains and evaluates a SARIMA model using pmdarima.auto_arima for order selection.

    Handles exogenous variables, saves the best model, and logs parameters including
    which exogenous features were offered and ultimately used by the model.

    Args:
        train_series: Training time series data.
        test_series: Test time series data for evaluation.
        train_exog (optional): DataFrame of exogenous variables for training.
        test_exog (optional): DataFrame of exogenous variables for testing/forecasting.
        seasonal_period_val: The seasonal period length.
        store_id: Store ID for logging.
        dept_id: Department ID for logging.

    Returns:
        tuple: (model_parameters_dict, model_metrics_dict).
               Returns (error_dict, None) on failure.
    """
    model_name = "SARIMA (auto-selected)"
    print(f"Training {model_name} for Store {store_id}-Dept {dept_id}...")
    params_to_log = {'seasonal_period_configured': seasonal_period_val}

    if train_series.empty or len(train_series) < max(seasonal_period_val * 2, 60): # pmdarima needs sufficient data
        params_to_log['error'] = 'Training series too short for SARIMA.'
        return params_to_log, None

    # Prepare exogenous variables, ensuring alignment and no all-NaN columns
    current_train_exog, current_test_exog = None, None
    exog_names_offered = []
    if train_exog is not None and not train_exog.empty:
        current_train_exog = train_exog.copy().reindex(train_series.index).ffill().bfill()
        current_train_exog.dropna(axis=1, how='all', inplace=True) # Drop cols if all NaN
        current_train_exog.fillna(0, inplace=True) # Fill remaining NaNs with 0
        if not current_train_exog.empty:
            exog_names_offered = list(current_train_exog.columns)
            if test_exog is not None and not test_exog.empty:
                current_test_exog = test_exog.copy().reindex(columns=exog_names_offered, fill_value=0)
                current_test_exog = current_test_exog.reindex(test_series.index).ffill().bfill()
                current_test_exog.fillna(0, inplace=True)
                if current_test_exog.empty: current_test_exog = None # Reset if all processing leads to empty
        else: current_train_exog = None # Reset if all processing leads to empty
    params_to_log['sarima_exog_features_offered'] = exog_names_offered

    fitted_sarima_model, metrics = None, None
    try:
        # auto_arima finds the best SARIMA model according to AIC / BIC
        fitted_sarima_model = auto_arima(
            train_series,
            exogenous=current_train_exog,
            start_p=1, start_q=1, max_p=2, max_q=2, max_d=1, # Non-seasonal orders
            start_P=0, start_Q=0, max_P=1, max_Q=1, max_D=1, # Seasonal orders
            m=seasonal_period_val, seasonal=True,
            stepwise=True, suppress_warnings=True, D=None, d=None, # Let auto_arima find D/d
            trace=False, error_action='warn', random_state=42,
            n_fits=10, # Number of models to fit (can increase for more thorough search)
            information_criterion='aic'
        )
        # Log details of the selected model
        params_to_log.update({
            'order': fitted_sarima_model.order,
            'seasonal_order': fitted_sarima_model.seasonal_order,
            'trend': getattr(fitted_sarima_model, 'trend', None) # Trend component, if any
        })
        # Determine which exogenous variables were actually used by the final model
        exog_names_used_by_model = []
        if hasattr(fitted_sarima_model, 'model_') and hasattr(fitted_sarima_model.model_, 'exog_names'):
            exog_names_used_by_model = list(fitted_sarima_model.model_.exog_names)
        elif hasattr(fitted_sarima_model, 'exog_names_'): # For older pmdarima versions
             exog_names_used_by_model = list(fitted_sarima_model.exog_names_)
        params_to_log['sarima_exog_features_actually_used_by_model'] = exog_names_used_by_model

        print(f"  Best SARIMA order: {params_to_log.get('order')}, Seasonal: {params_to_log.get('seasonal_order')}")
        print(f"  SARIMA Exog features used: {exog_names_used_by_model}")

        # Prepare exogenous variables for prediction, ensuring they match what the model expects
        predict_exog_for_sarima_final = None
        if exog_names_used_by_model and current_test_exog is not None:
            missing_exog_for_pred = [col for col in exog_names_used_by_model if col not in current_test_exog.columns]
            if not missing_exog_for_pred:
                predict_exog_for_sarima_final = current_test_exog[exog_names_used_by_model]
                # Ensure index matches test_series for prediction
                if len(predict_exog_for_sarima_final) != len(test_series):
                     predict_exog_for_sarima_final = predict_exog_for_sarima_final.reindex(test_series.index).ffill().bfill().fillna(0)
            else:
                print(f"  Warning: Required exog features {missing_exog_for_pred} for SARIMA prediction not available in test_exog.")

        predictions, _ = fitted_sarima_model.predict(
            n_periods=len(test_series),
            X=predict_exog_for_sarima_final,
            return_conf_int=True # To get confidence intervals if needed later
        )
        predictions_series = pd.Series(predictions, index=test_series.index)
        metrics = _calculate_ts_metrics(test_series, predictions_series)
        print(f"  {model_name} Metrics: {metrics}")

    except Exception as e_sarima: # pragma: no cover
        print(f"  Error training/evaluating {model_name}: {e_sarima}")
        params_to_log['error_fitting_or_prediction'] = str(e_sarima)
        metrics = None # Ensure metrics are None on error

    if fitted_sarima_model:
        model_log_name_sarima = f"SARIMA_auto_S{store_id}D{dept_id}"
        model_filename_sarima = f"{model_log_name_sarima}_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib"
        model_save_path_sarima = os.path.join(MODEL_STORE_PATH, model_filename_sarima)
        try:
            joblib.dump(fitted_sarima_model, model_save_path_sarima)
            params_to_log['model_path'] = model_save_path_sarima
        except Exception as e_save: # pragma: no cover
            params_to_log['model_path'] = f"Error saving SARIMA model: {e_save}"
            print(f"Error saving SARIMA model: {e_save}")
            
    return params_to_log, metrics


def _train_evaluate_prophet(
    train_df_prophet: pd.DataFrame, test_df_prophet: pd.DataFrame,
    holiday_df_prophet: pd.DataFrame | None = None,
    exog_regressor_names: list | None = None,
    store_id: int = 0, dept_id: int = 0
) -> tuple[dict | None, dict | None]:
    """
    Trains and evaluates a Prophet forecasting model.

    Handles holidays and additional regressors (exogenous variables).
    Saves the fitted model and logs parameters including used regressors.

    Args:
        train_df_prophet: Training data in Prophet format (columns 'ds' and 'y', plus regressors).
        test_df_prophet: Test data in Prophet format for making future predictions.
        holiday_df_prophet (optional): DataFrame with holiday information ('ds', 'holiday').
        exog_regressor_names (optional): List of column names in train/test_df_prophet to be used as regressors.
        store_id: Store ID for logging.
        dept_id: Department ID for logging.

    Returns:
        tuple: (model_parameters_dict, model_metrics_dict).
               Returns (error_dict, None) on failure.
    """
    model_name = "Prophet"
    print(f"Training {model_name} for Store {store_id}-Dept {dept_id}...")
    # Default Prophet parameters; can be extended or made configurable
    params_to_log = {
        'seasonality_mode': 'multiplicative', 'yearly_seasonality': True,
        'weekly_seasonality': 'auto', 'daily_seasonality': 'auto', # Prophet often handles auto well
        'holidays_prior_scale': 10.0, 'changepoint_prior_scale': 0.05
    }
    if train_df_prophet.empty:
        params_to_log['error'] = 'Empty training data provided for Prophet.'
        return params_to_log, None

    prophet_model_instance = Prophet(
        seasonality_mode=params_to_log['seasonality_mode'],
        yearly_seasonality=params_to_log['yearly_seasonality'],
        weekly_seasonality=params_to_log['weekly_seasonality'],
        daily_seasonality=params_to_log['daily_seasonality'],
        holidays_prior_scale=params_to_log['holidays_prior_scale'],
        changepoint_prior_scale=params_to_log['changepoint_prior_scale'],
        holidays=holiday_df_prophet if holiday_df_prophet is not None and not holiday_df_prophet.empty else None
    )
    params_to_log['holidays_df_provided_to_model'] = holiday_df_prophet is not None and not holiday_df_prophet.empty

    # Add regressors if provided and available in training data
    actual_regressors_added = []
    if exog_regressor_names:
        for regressor_name in exog_regressor_names:
            if regressor_name in train_df_prophet.columns:
                try:
                    prophet_model_instance.add_regressor(regressor_name)
                    actual_regressors_added.append(regressor_name)
                except Exception as e_regressor: # pragma: no cover
                    print(f"  Warning: Could not add regressor '{regressor_name}' to Prophet: {e_regressor}")
    params_to_log['prophet_regressors_used_by_model'] = actual_regressors_added

    fitted_prophet_model, metrics = None, None
    try:
        # Suppress verbose Stan output and FutureWarnings from Prophet/pandas
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            fitted_prophet_model = prophet_model_instance.fit(train_df_prophet)

        # Prepare future DataFrame for prediction (must include 'ds' and all added regressors)
        if test_df_prophet.empty or 'ds' not in test_df_prophet.columns:
            raise ValueError("Test DataFrame for Prophet prediction is empty or missing 'ds' column.")
        
        # Ensure all regressors model was trained with are present in test_df_prophet for prediction
        future_df_for_prediction = test_df_prophet[['ds'] + actual_regressors_added].copy()
        if future_df_for_prediction.isnull().values.any():
            print("  Warning: NaNs found in future dataframe for Prophet regressors. Attempting ffill/bfill.")
            for reg_col in actual_regressors_added:
                 future_df_for_prediction[reg_col] = future_df_for_prediction[reg_col].ffill().bfill().fillna(0)


        forecast_output_df = fitted_prophet_model.predict(future_df_for_prediction)
        
        # Align predictions with actual test dates and target column 'y'
        predictions_series = forecast_output_df.set_index('ds')['yhat'].reindex(
            test_df_prophet.set_index('ds').index
        ).fillna(0) # Fill any misaligned prediction NaNs with 0
        true_values_series = test_df_prophet.set_index('ds')['y']

        metrics = _calculate_ts_metrics(true_values_series, predictions_series)
        print(f"  {model_name} Metrics: {metrics}")

    except Exception as e_prophet: # pragma: no cover
        print(f"  Error training/evaluating {model_name}: {e_prophet}")
        params_to_log['error_fitting_or_prediction'] = str(e_prophet)
        metrics = None # Ensure metrics are None on error

    if fitted_prophet_model:
        model_log_name_prophet = f"Prophet_S{store_id}D{dept_id}"
        # Prophet models are typically saved as JSON, but joblib can also work for the Python object.
        # For cross-platform/language, JSON is preferred via Prophet's built-in serialization.
        # Here, using joblib for consistency with other models in this script.
        model_filename_prophet = f"{model_log_name_prophet}_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib"
        model_save_path_prophet = os.path.join(MODEL_STORE_PATH, model_filename_prophet)
        try:
            joblib.dump(fitted_prophet_model, model_save_path_prophet)
            print(f"  Saved Prophet model to: {model_save_path_prophet}")
            params_to_log['model_path'] = model_save_path_prophet
        except Exception as e_save: # pragma: no cover
            params_to_log['model_path'] = f"Error saving Prophet model: {e_save}"
            print(f"Error saving Prophet model: {e_save}")
            
    return params_to_log, metrics

# --- Main Orchestration Function ---
def main_classical_ts_pipeline():
    """
    Main function to orchestrate the classical time series modeling pipeline.

    Iterates through selected Store-Department combinations, loads data,
    prepares data for time series models (including exogenous variables and
    holidays for SARIMA/Prophet), splits data, trains various classical
    time series models, evaluates them, and logs results.
    """
    print("--- Starting Classical Time Series Modeling Pipeline ---")
    # Suppress common warnings from statsmodels, pmdarima, prophet during fitting
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels") # Specific to statsmodels
    warnings.filterwarnings("ignore", category=UserWarning, module="pmdarima") # Specific to pmdarima

    df_featured_full = _load_featured_data_ts()
    if df_featured_full is None:
        print("Failed to load featured data. Exiting time series modeling pipeline.")
        return

    # Define a sample of Store-Department combinations to model for demonstration
    # In a production scenario, this would iterate over all relevant combinations or be configurable.
    if 'Store' not in df_featured_full.columns or 'Dept' not in df_featured_full.columns:
        print("Error: 'Store' or 'Dept' columns missing from featured data. Cannot proceed.")
        return
        
    example_stores = df_featured_full['Store'].unique()
    example_depts = df_featured_full['Dept'].unique()
    store_dept_combinations_to_model = [
        (1, 1), # A known combination
        (example_stores[0] if len(example_stores) > 0 else 1, # First available store
         example_depts[1] if len(example_depts) > 1 else (example_depts[0] if len(example_depts) > 0 else 1)) # Second available dept or first
    ]
    # Ensure unique combinations if dynamically generated ones overlap
    store_dept_combinations_to_model = sorted(list(set(store_dept_combinations_to_model)))

    # Define potential exogenous features available in the dataset
    potential_exog_feature_names = [
        'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
        'Size', 'Year', 'Month_sin', 'Month_cos', 'WeekOfYear_sin', 'WeekOfYear_cos'
    ]

    for store_id_iter, dept_id_iter in store_dept_combinations_to_model:
        print(f"\n--- Processing Time Series for Store: {store_id_iter}, Dept: {dept_id_iter} ---")
        series_df_current_sd = df_featured_full[
            (df_featured_full['Store'] == store_id_iter) & (df_featured_full['Dept'] == dept_id_iter)
        ].copy()

        if series_df_current_sd.empty or len(series_df_current_sd) < 104: # Min 2 years of data
            print(f"Not enough data for Store {store_id_iter}-Dept {dept_id_iter} "
                  f"(found {len(series_df_current_sd)} weeks, require at least 104). Skipping.")
            continue

        # Ensure 'Date' is datetime and set as index for target series
        series_df_current_sd['Date'] = pd.to_datetime(series_df_current_sd['Date'])
        series_df_current_sd.set_index('Date', inplace=True)
        series_df_current_sd.sort_index(inplace=True) # Ensure chronological order
        
        # Prepare target series, handling NaNs
        target_sales_series = series_df_current_sd['Weekly_Sales'].asfreq('W-FRI').interpolate(method='time') # Ensure weekly freq, interpolate small gaps
        if target_sales_series.isnull().any(): # If interpolation didn't fill all, ffill/bfill
            target_sales_series = target_sales_series.ffill().bfill()
        if target_sales_series.isnull().any(): # If still NaNs (e.g., all NaNs initially)
            print(f"Target series for S{store_id_iter}D{dept_id_iter} contains unfillable NaNs after interpolation and ffill/bfill. Skipping.")
            continue
        
        # Prepare exogenous data for SARIMA and regressors for Prophet
        current_exog_cols_available = [col for col in potential_exog_feature_names if col in series_df_current_sd.columns]
        exog_data_for_models = pd.DataFrame()
        if current_exog_cols_available:
            exog_data_for_models = series_df_current_sd[current_exog_cols_available].copy().asfreq('W-FRI').interpolate(method='time')
            exog_data_for_models = exog_data_for_models.ffill().bfill() # Fill any remaining NaNs
            # Drop columns that are all NaN after processing
            exog_data_for_models.dropna(axis=1, how='all', inplace=True)
            # Fill any individual remaining NaNs with 0 (or mean/median if preferred)
            exog_data_for_models.fillna(0, inplace=True)
            if exog_data_for_models.empty: # Reset if all columns were dropped
                 exog_data_for_models = None # type: ignore
            else: # Realign with target series index after processing
                 exog_data_for_models = exog_data_for_models.reindex(target_sales_series.index).ffill().bfill()


        # Prepare data for Prophet (ds, y, and regressors)
        prophet_input_df = pd.DataFrame({'ds': target_sales_series.index, 'y': target_sales_series.values})
        prophet_regressor_names_final = []
        if exog_data_for_models is not None:
            for col_exog_prophet in exog_data_for_models.columns:
                prophet_input_df[col_exog_prophet] = exog_data_for_models[col_exog_prophet].values
                prophet_regressor_names_final.append(col_exog_prophet)
        
        # Prepare holiday DataFrame for Prophet
        holidays_df_for_prophet = None
        if 'IsHoliday' in series_df_current_sd.columns: # Ensure IsHoliday was in original data
            holiday_dates_series = series_df_current_sd.index[series_df_current_sd['IsHoliday'] == 1]
            if not holiday_dates_series.empty:
                holidays_df_for_prophet = pd.DataFrame({
                    'ds': holiday_dates_series,
                    'holiday': 'SpecialHoliday' # Generic holiday name
                })

        # Split data
        num_test_periods = 52 # 1 year of weekly data for testing
        train_target, test_target = _time_series_split_strict(target_sales_series, test_size=num_test_periods)
        if train_target is None or test_target is None: # Check if split failed
            print(f"Data splitting failed for S{store_id_iter}D{dept_id_iter}. Skipping."); continue

        train_exog_for_sarima, test_exog_for_sarima = (None, None)
        if exog_data_for_models is not None and not exog_data_for_models.empty:
            train_exog_for_sarima, test_exog_for_sarima = _time_series_split_strict(exog_data_for_models, test_size=num_test_periods)

        train_prophet_input_df, test_prophet_input_df = _time_series_split_strict(prophet_input_df, test_size=num_test_periods)
        if train_prophet_input_df is None or test_prophet_input_df is None: # Check if split failed
             print(f"Prophet data splitting failed for S{store_id_iter}D{dept_id_iter}. Skipping Prophet model.");
        
        train_period_str_log = f"{train_target.index.min().strftime('%Y-%m-%d')} to {train_target.index.max().strftime('%Y-%m-%d')}"
        test_period_str_log = f"{test_target.index.min().strftime('%Y-%m-%d')} to {test_target.index.max().strftime('%Y-%m-%d')}"
        
        current_run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Model Training and Logging
        model_params, model_metrics = _train_evaluate_naive(train_target, test_target)
        log_ts_experiment(store_id_iter, dept_id_iter, 'NaiveForecast', model_params, model_metrics, train_period_str_log, test_period_str_log, current_run_timestamp)

        model_params, model_metrics = _train_evaluate_seasonal_naive(train_target, test_target, seasonal_period=52)
        log_ts_experiment(store_id_iter, dept_id_iter, 'SeasonalNaiveForecast', model_params, model_metrics, train_period_str_log, test_period_str_log, current_run_timestamp)
        
        model_params, model_metrics = _train_evaluate_ets(train_target, test_target, seasonal_periods_val=52, store_id=store_id_iter, dept_id=dept_id_iter)
        log_ts_experiment(store_id_iter, dept_id_iter, 'ETS (Best)', model_params, model_metrics, train_period_str_log, test_period_str_log, current_run_timestamp)
        
        model_params, model_metrics = _train_evaluate_sarima(
            train_target, test_target,
            train_exog=train_exog_for_sarima, test_exog=test_exog_for_sarima,
            seasonal_period_val=52, store_id=store_id_iter, dept_id=dept_id_iter
        )
        log_ts_experiment(store_id_iter, dept_id_iter, 'SARIMA (auto)', model_params, model_metrics, train_period_str_log, test_period_str_log, current_run_timestamp)
        
        if train_prophet_input_df is not None and test_prophet_input_df is not None: # Only run if data split was successful
            prophet_params, prophet_metrics = _train_evaluate_prophet(
                train_df_prophet=train_prophet_input_df,
                test_df_prophet=test_prophet_input_df,
                holiday_df_prophet=holidays_df_for_prophet,
                exog_regressor_names=prophet_regressor_names_final,
                store_id=store_id_iter, dept_id=dept_id_iter
            )
            log_ts_experiment(store_id_iter, dept_id_iter, 'Prophet', prophet_params, prophet_metrics, train_period_str_log, test_period_str_log, current_run_timestamp)

    save_ts_experiment_log()
    print("\n--- Classical Time Series Modeling Pipeline Finished ---")

if __name__ == '__main__':
    main_classical_ts_pipeline()