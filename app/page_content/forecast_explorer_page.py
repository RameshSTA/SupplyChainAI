"""
Renders the Forecast Explorer page for the Streamlit application.

This page allows users to visualize and compare different forecasting models
applied to Walmart sales data. It loads experimental logs, featured data,
and saved models, enabling users to select store, department, and model
to see historical performance and generate future predictions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn is imported, ensure it's used or remove if not.
import os
import sys
import joblib
import ast
import re

# --- Optional Deep Learning Imports ---
try:
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler # Commonly used with TF for scaling
    if tf.__version__.startswith('1.'):
        st.warning("TensorFlow 1.x detected. TensorFlow 2.x or higher is generally preferred.", icon="⚠️")
except ImportError:
    tf = None
    MinMaxScaler = None # Ensure MinMaxScaler is also None if sklearn is not found or tf is the trigger
    st.info("TensorFlow or scikit-learn (for MinMaxScaler) not found. LSTM model functionality will be limited or unavailable.")

# --- Optional Gradient Boosting Imports ---
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# --- Optional Prophet Import ---
try:
    from prophet import Prophet
    from prophet.serialize import model_from_json
except ImportError:
    Prophet = None
    model_from_json = None

# --- Optional pmdarima (Auto-ARIMA) Import ---
try:
    import pmdarima as pm # Used for auto_arima (SARIMA)
except ImportError:
    pm = None

# --- Initialize Utility Function Placeholders ---
preprocess_regression_features = None
engineer_all_demand_features = None
create_sequences = None # This might be from preprocessing or feature_engineering_utils

def _get_project_root_fc_explorer() -> str:
    """
    Determines the project root directory for this page.

    Assumes this script (`forecast_explorer_page.py`) is located within:
    `PROJECT_ROOT/app/page_content/`.
    It navigates up two directories from this file's current location.

    Returns:
        str: The absolute path to the project root directory.
    """
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up: app/page_content -> app -> PROJECT_ROOT
        return os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    except NameError: # pragma: no cover
        # This fallback is for environments where __file__ might not be defined
        st.warning(
            "Could not automatically determine project root using `__file__`. "
            "Falling back to current working directory. Module imports or data paths might be incorrect."
        )
        return os.getcwd()

# --- Project Path Setup & Custom Module Imports ---
_UTIL_FUNCS_LOADED_FC_EXPLORER = False
try:
    PROJECT_ROOT = _get_project_root_fc_explorer()
    SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)
    if PROJECT_ROOT not in sys.path: # For data loading or other project-relative paths
        sys.path.insert(0, PROJECT_ROOT)

    # Attempt to import utility functions
    from utils.preprocessing import preprocess_regression_features
    from utils.feature_engineering_utils import engineer_all_demand_features

    try:
        # Attempt primary location for create_sequences (often with preprocessing)
        from utils.preprocessing import create_sequences
    except ImportError:
        # Fallback location (as seen in user's original code for this page)
        from utils.feature_engineering_utils import create_sequences
    _UTIL_FUNCS_LOADED_FC_EXPLORER = True
    print("Forecast Explorer: Utility functions loaded successfully.")
except Exception as e_setup: # pragma: no cover
    _UTIL_FUNCS_LOADED_FC_EXPLORER = False
    st.error(f"CRITICAL ERROR during sys.path setup or utility imports in Forecast Explorer: {e_setup}")
    st.info(
        "This page requires utility functions from 'src/utils/'. "
        "Ensure this directory and its Python files (including '__init__.py' "
        "in 'src' and 'src/utils') are correctly structured and importable."
    )
    # Define dummy functions if critical imports failed
    if 'preprocess_regression_features' not in globals() or preprocess_regression_features is None:
        def preprocess_regression_features(*args, **kwargs):
            st.error("Utility 'preprocess_regression_features' is unavailable."); return None, None, None, None
    if 'create_sequences' not in globals() or create_sequences is None:
        def create_sequences(*args, **kwargs):
            st.error("Utility 'create_sequences' is unavailable."); return np.empty((0,0,0)), np.empty((0,))
    if 'engineer_all_demand_features' not in globals() or engineer_all_demand_features is None:
        def engineer_all_demand_features(df_input, **kwargs):
            st.error("Utility 'engineer_all_demand_features' is unavailable."); return df_input if df_input is not None else pd.DataFrame()


# --- Configuration Paths (derived from PROJECT_ROOT) ---
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
FEATURED_DATA_FILENAME = 'walmart_data_featured.parquet'
EXPERIMENT_LOGS_PATH = os.path.join(PROJECT_ROOT, 'reports', 'experiment_logs')
MODEL_STORE_PATH = os.path.join(PROJECT_ROOT, 'models_store', 'demand_forecasting')
SCALER_STORE_PATH = os.path.join(MODEL_STORE_PATH, 'scalers') # Path for LSTM scalers

@st.cache_data
def load_all_experiment_logs(log_path: str = EXPERIMENT_LOGS_PATH) -> pd.DataFrame | None:
    """
    Loads and combines all relevant experiment log CSV files.

    Searches for 'classical_timeseries_experiments.csv',
    'classical_regression_experiments.csv', and 'deep_learning_experiments.csv'.
    Parses parameters and feature lists from string representations in the logs.

    Args:
        log_path: Path to the directory containing experiment log CSV files.

    Returns:
        A pandas DataFrame combining all found logs, or None if no logs are loaded.
    """
    log_files_info = {
        'classical_ts': 'classical_timeseries_experiments.csv',
        'classical_reg': 'classical_regression_experiments.csv',
        'deep_learning': 'deep_learning_experiments.csv'
    }
    all_logs_list = []

    def _safe_literal_eval_fc(val_str, target_type=dict):
        """Safely evaluate a string representation of a Python literal (dict or list)."""
        if isinstance(val_str, target_type): return val_str
        if isinstance(val_str, str):
            try: return ast.literal_eval(val_str)
            except (ValueError, SyntaxError, TypeError): pass # Ignore errors, return default
        return target_type() # Return empty dict or list as default

    for log_type, file_name in log_files_info.items():
        full_path = os.path.join(log_path, file_name)
        if os.path.exists(full_path):
            try:
                df_log = pd.read_csv(full_path)
                essential_cols = ['Store', 'Dept', 'Model', 'Parameters', 'Test_Period', 'RMSE'] # Added RMSE
                missing_essentials = [col for col in essential_cols if col not in df_log.columns]
                if missing_essentials:
                    st.info(f"Note: Essential columns {missing_essentials} missing in log file '{file_name}'. Some display features may be affected.")
                df_log['Log_Type_Source'] = log_type
                all_logs_list.append(df_log)
            except Exception as e:
                st.warning(f"Could not load or process log file '{file_name}': {e}")
        else:
            st.info(f"Optional log file not found: '{full_path}'. This may be expected if not all model types were run.")

    if not all_logs_list:
        st.error("No experiment logs were found or successfully loaded. Forecast Explorer cannot operate.")
        return None

    df_combined = pd.concat(all_logs_list, ignore_index=True)

    # Parse 'Parameters' into 'Parameters_Dict' and extract 'model_path'
    if 'Parameters' in df_combined.columns:
        df_combined['Parameters_Dict'] = df_combined['Parameters'].apply(lambda x: _safe_literal_eval_fc(x, dict))
        df_combined['model_path'] = df_combined['Parameters_Dict'].apply(lambda p_dict: p_dict.get('model_path'))
    else:
        df_combined['Parameters_Dict'] = pd.Series([{} for _ in range(len(df_combined))])
        df_combined['model_path'] = None # Ensure column exists

    # Parse 'Features_List_Used' into 'Parsed_Features_List'
    df_combined['Parsed_Features_List'] = pd.Series([[] for _ in range(len(df_combined))], dtype=object) # Initialize
    if 'Features_List_Used' in df_combined.columns:
        df_combined['Parsed_Features_List'] = df_combined['Features_List_Used'].apply(lambda x: _safe_literal_eval_fc(x, list))

    def _consolidate_feature_list(row):
        """Consolidates feature lists from 'Parsed_Features_List' or common keys in 'Parameters_Dict'."""
        if isinstance(row['Parsed_Features_List'], list) and row['Parsed_Features_List']:
            return row['Parsed_Features_List']
        params_dict = row.get('Parameters_Dict', {})
        # Common keys where feature lists might be stored if 'Features_List_Used' was not populated
        feature_keys_in_params = [
            'features_used_list_in_params', 'features_used_for_lstm_input',
            'sarima_exog_features_actually_used_by_model', 'prophet_regressors_used',
            'features_used_list', 'features_used'
        ]
        for key in feature_keys_in_params:
            features = params_dict.get(key)
            if isinstance(features, list) and features:
                return features
        return [] # Default to empty list
    df_combined['Parsed_Features_List'] = df_combined.apply(_consolidate_feature_list, axis=1)

    # Standardize numeric and metric columns
    for col_name in ['Store', 'Dept']:
        if col_name in df_combined.columns:
            df_combined[col_name] = pd.to_numeric(df_combined[col_name], errors='coerce').fillna(-1).astype(int)
    for metric_name in ['MAE', 'RMSE', 'MAPE']:
        if metric_name in df_combined.columns:
            df_combined[metric_name] = pd.to_numeric(df_combined[metric_name], errors='coerce')
        else: # Ensure metric columns exist, even if all NaN, for consistency
            df_combined[metric_name] = np.nan
    return df_combined

@st.cache_data
def load_main_featured_data(data_path: str = PROCESSED_DATA_PATH, filename: str = FEATURED_DATA_FILENAME) -> pd.DataFrame | None:
    """
    Loads the main preprocessed and feature-engineered dataset.
    Supports Parquet and CSV. Ensures 'Date' is datetime and 'Type_Encoded' exists if 'Type' is present.
    """
    full_path = os.path.join(data_path, filename)
    if not os.path.exists(full_path):
        st.error(f"Featured data file not found: {full_path}")
        return None
    try:
        if full_path.lower().endswith('.parquet'):
            df = pd.read_parquet(full_path)
        elif full_path.lower().endswith('.csv'):
            df = pd.read_csv(full_path, parse_dates=['Date'])
        else:
            st.error(f"Unsupported data file format: {full_path}. Only .parquet and .csv are supported.")
            return None

        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        if 'Type' in df.columns and 'Type_Encoded' not in df.columns: # For regression model compatibility
            from sklearn.preprocessing import LabelEncoder # Safe local import
            le = LabelEncoder()
            df['Type_Encoded'] = le.fit_transform(df['Type'].astype(str))
        return df
    except Exception as e:
        st.error(f"Error loading featured data from '{full_path}': {e}")
        return None

@st.cache_resource # Models can be large, use cache_resource
def load_selected_model(model_path_str: str | None, model_name_to_load: str | None):
    """
    Loads a saved machine learning model based on its path and name hint.
    Enhanced with more detailed path checking and error reporting.
    """
    st.info(f"Attempting to load model: '{model_name_to_load}' from expected path: '{model_path_str}'")

    if not model_path_str or not isinstance(model_path_str, str):
        st.error("Model path is None or not a string. Cannot load model.")
        return None
    if not model_name_to_load: # Should not happen if UI populates correctly
        st.error("Model name (type hint) not provided for loading.")
        return None

    model_dir = os.path.dirname(model_path_str)
    if not os.path.isdir(model_dir): # Check if it's a directory
        st.error(f"The directory for the model does not exist or is not a directory: '{model_dir}'")
        st.info("Ensure training scripts ran and created this directory structure, "
                "and PROJECT_ROOT is consistent across scripts.")
        return None
    # else: # Optional: st.write(f"Confirmed model directory exists: '{model_dir}'")

    if not os.path.exists(model_path_str):
        st.error(f"Model file does not exist at the specified path: '{model_path_str}'")
        st.info(f"This usually means the training script for '{model_name_to_load}' "
                "has not been run, did not complete, or saved the model with a different name/location.")
        try:
            st.markdown(f"**Files found in the model directory (`{model_dir}`) for reference:**")
            files_in_dir = os.listdir(model_dir)
            if files_in_dir:
                st.json(files_in_dir[:20]) # Display up to 20 files
            else:
                st.write("No files found in this directory.")
        except Exception as e_listdir:
            st.write(f"Could not list files in directory '{model_dir}': {e_listdir}")
        return None

    # If path and file exist, proceed with loading
    try:
        model_name_lower = model_name_to_load.lower()
        if model_name_lower.startswith('lstm') and model_path_str.endswith(('.keras', '.h5')):
            if tf: return tf.keras.models.load_model(model_path_str)
            else: st.error("TensorFlow not available; cannot load LSTM model."); return None
        elif model_name_lower.startswith('xgboost') and model_path_str.endswith(('.json', '.ubj')):
            if xgb: model = xgb.XGBRegressor(); model.load_model(model_path_str); return model
            else: st.error("XGBoost not available; cannot load XGBoost model."); return None
        elif model_name_lower.startswith('lightgbm') and model_path_str.endswith('.txt'):
            if lgb: return lgb.Booster(model_file=model_path_str)
            else: st.error("LightGBM not available; cannot load LightGBM model."); return None
        elif model_name_lower.startswith('prophet') and model_path_str.endswith('.json'):
            if Prophet and model_from_json:
                with open(model_path_str, 'r') as fin: return model_from_json(fin.read())
            else: st.error("Prophet not available; cannot load Prophet model."); return None
        elif model_path_str.endswith('.joblib'): # For scikit-learn, pmdarima, etc.
            return joblib.load(model_path_str)
        else:
            st.warning(f"Unsupported model type or file extension for '{model_name_to_load}' from '{model_path_str}'.")
            return None
    except Exception as e: # Catch any exception during the actual model loading
        st.error(f"An error occurred while loading model '{model_name_to_load}' from '{model_path_str}': {e}")
        return None

# --- Main function for the Forecast Explorer page ---
def render_forecast_explorer_page():
    """
    Renders the main content for the Forecast Visualizer & Explorer page.

    Orchestrates UI for selecting store, department, and model, loads data/models,
    displays metrics, and generates/plots test period and future forecasts (placeholders for now).
    """
    st.header("🚀 Forecast Visualizer & Explorer")
    st.markdown("Select Store, Department, and Model to visualize its performance and generate future forecasts.")

    if not _UTIL_FUNCS_LOADED_FC_EXPLORER: # pragma: no cover
        st.error(
            "One or more critical utility functions (e.g., for preprocessing, feature engineering) "
            "failed to load during initial page setup. Forecast Explorer page functionality is severely limited."
        )
        # st.stop() # Consider if stopping is too abrupt or if page should partially render
        return # Exit rendering if utilities not loaded

    df_logs = load_all_experiment_logs()
    df_featured_full = load_main_featured_data()

    if df_logs is None or df_logs.empty or df_featured_full is None:
        st.warning(
            "Required experiment logs or the main featured dataset could not be loaded. "
            "Forecast Explorer functionality is limited or unavailable. "
            "Please ensure relevant training scripts have run and data/log paths are correct."
        )
        return

    # --- User Selections ---
    st.subheader("⚙️ Select Forecast Parameters")
    sel_col1, sel_col2, sel_col3, sel_col4 = st.columns(4)

    unique_stores = sorted([s for s in df_logs['Store'].unique() if s != -1 and s != 0]) # Exclude placeholders
    global_model_store_placeholder = 0 # Convention for global models

    selected_store = None
    if not unique_stores:
        with sel_col1: st.info("No store-specific forecast logs available.")
    else:
        with sel_col1:
            selected_store = st.selectbox("Store:", unique_stores, index=0, key="fc_store_select")

    selected_dept = None
    depts_for_store = []
    if selected_store:
        depts_for_store = sorted([
            d for d in df_logs[df_logs['Store'] == selected_store]['Dept'].unique() if d != -1 and d != 0
        ])
        if not depts_for_store:
            with sel_col2: st.info(f"No department-specific forecast logs for Store {selected_store}.")
        else:
            with sel_col2:
                selected_dept = st.selectbox("Department:", depts_for_store, index=0, key="fc_dept_select")
    elif unique_stores:
        with sel_col2: st.info("Select a Store to view Departments.")


    selected_model_name_from_log = None
    model_info_selected_row = None # Will store the selected model's full log row
    if selected_store and selected_dept:
        # Filter for store-specific models
        store_dept_models_df = df_logs[
            (df_logs['Store'] == selected_store) &
            (df_logs['Dept'] == selected_dept) &
            (df_logs['model_path'].notna()) & # Ensure model path exists
            (df_logs['model_path'] != "")     # Ensure model path is not an empty string
        ].copy()
        if not store_dept_models_df.empty:
            store_dept_models_df['Model_Display_Name_Base'] = store_dept_models_df['Model']

        # Filter for global models
        global_models_df = df_logs[
            (df_logs['Store'] == global_model_store_placeholder) &
            (df_logs['Dept'] == global_model_store_placeholder) &
            (df_logs['model_path'].notna()) &
            (df_logs['model_path'] != "")
        ].copy()
        if not global_models_df.empty:
            global_models_df['Model_Display_Name_Base'] = global_models_df['Model'] + " (Global)"

        # Combine and create display names
        all_relevant_models_df = pd.concat([store_dept_models_df, global_models_df], ignore_index=True)

        if not all_relevant_models_df.empty:
            # Ensure RMSE column exists and handle potential NaNs before formatting
            if 'RMSE' in all_relevant_models_df.columns:
                all_relevant_models_df['display_name_with_metrics'] = (
                    all_relevant_models_df['Model_Display_Name_Base'].astype(str) + " (RMSE: " +
                    all_relevant_models_df['RMSE'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A") + ")"
                )
            else: # Fallback if RMSE is missing
                all_relevant_models_df['display_name_with_metrics'] = all_relevant_models_df['Model_Display_Name_Base'].astype(str) + " (RMSE: N/A)"

            available_models_display_list = all_relevant_models_df['display_name_with_metrics'].unique().tolist()

            if available_models_display_list:
                with sel_col3:
                    selected_model_display_name = st.selectbox(
                        "Model:", available_models_display_list, index=0, key="fc_model_select"
                    )
                if selected_model_display_name: # Logic dependent on this selection
                    model_info_row_df = all_relevant_models_df[
                        all_relevant_models_df['display_name_with_metrics'] == selected_model_display_name
                    ]
                    if not model_info_row_df.empty:
                        model_info_selected_row = model_info_row_df.iloc[0].copy() # Get the Series for the selected model
                        selected_model_name_from_log = model_info_selected_row['Model']
            else:
                with sel_col3: st.info(f"No loadable models found for S{selected_store}-D{selected_dept} or globally.")
        else:
            with sel_col3: st.info(f"No models (store-specific or global) with valid model paths found for S{selected_store}-D{selected_dept}.")
    elif selected_store and not selected_dept and depts_for_store:
        with sel_col3: st.info("Select a Department to view available models.")
    elif not selected_store and unique_stores:
         with sel_col3: st.info("Select Store & Department to view models.")

    with sel_col4:
        future_forecast_weeks = st.number_input(
            "Future Weeks to Forecast:", min_value=0, max_value=104, value=12, step=4, key="fc_future_weeks",
            help="Number of future weeks to forecast beyond the available historical data."
        )
    st.markdown("---")

    # --- Main Display Area for Forecasts and Model Info ---
    if selected_store and selected_dept and selected_model_name_from_log and model_info_selected_row is not None:
        st.subheader(f"Results for Store {selected_store} - Dept {selected_dept} using {model_info_selected_row['Model_Display_Name_Base']}")

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            with st.expander("Logged Test Set Performance", expanded=True):
                metrics_to_show = {
                    k: model_info_selected_row[k] for k in ['MAE', 'RMSE', 'MAPE']
                    if k in model_info_selected_row and pd.notna(model_info_selected_row[k])
                }
                if metrics_to_show: st.json(metrics_to_show)
                else: st.write("Performance metrics not available or not logged for this model.")
        with exp_col2:
            with st.expander("Model Hyperparameters & Logged Info", expanded=True):
                params_dict_from_log_display = model_info_selected_row.get('Parameters_Dict', {})
                keys_to_exclude_display = [
                    'model_path', 'feature_scaler_path', 'target_scaler_path',
                    'features_used_list_in_params', 'features_used_for_lstm_input',
                    'features_used_list', 'features_used', 'train_period_in_params',
                    'test_period_in_params', 'sarima_exog_features_offered',
                    'sarima_exog_features_actually_used_by_model',
                    'prophet_regressors_used', 'input_shape_str', 'error',
                    'train_period_logged_in_params', 'test_period_logged_in_params', # Redundant if separate cols
                    'features_used_logged_in_params'
                ]
                params_to_show_ui = {
                    k: v for k, v in params_dict_from_log_display.items() if k not in keys_to_exclude_display
                }
                if params_to_show_ui: st.json(params_to_show_ui)
                else: st.write("Key hyperparameters not found or all were filtered for display.")
                st.caption(f"Full Parameters String (from log): `{model_info_selected_row.get('Parameters', 'Not available')}`")
                st.caption(f"Features Used (parsed from log): `{model_info_selected_row.get('Parsed_Features_List', [])}`")

        model_path_from_log = model_info_selected_row.get('model_path')
        st.markdown(f"**Source Log Entry Details:** Log Type: `{model_info_selected_row.get('Log_Type_Source', 'N/A')}`, Model Name in Log: `{selected_model_name_from_log}`")
        st.markdown(f"Attempting to load model from logged path: `{model_path_from_log}`")


        loaded_model_object = load_selected_model(model_path_from_log, selected_model_name_from_log)

        if loaded_model_object:
            st.success(f"Model '{selected_model_name_from_log}' loaded successfully!")

            # Prepare historical data for plotting for the selected store-dept
            historical_data_sd = df_featured_full[
                (df_featured_full['Store'] == selected_store) &
                (df_featured_full['Dept'] == selected_dept)
            ].copy()
            if 'Date' not in historical_data_sd.columns or 'Weekly_Sales' not in historical_data_sd.columns :
                st.error("Critical error: 'Date' or 'Weekly_Sales' column missing in filtered historical data for plotting.")
                return # Stop if essential columns are missing

            historical_sales_series_plot = historical_data_sd.set_index('Date')['Weekly_Sales'].sort_index()

            # Initialize series for predictions
            test_period_predictions_series, future_period_predictions_series = None, None
            actuals_on_test_period_plot, test_start_date, test_end_date = None, None, None

            # Attempt to parse test period dates from logs to plot actuals
            params_dict_from_log_for_dates = model_info_selected_row.get('Parameters_Dict', {})
            # Prioritize explicit Test_Period column, then from Parameters_Dict
            test_period_str_from_log_val = model_info_selected_row.get('Test_Period',
                                                                  params_dict_from_log_for_dates.get('test_period_in_params',
                                                                                                     params_dict_from_log_for_dates.get('test_period')))
            if isinstance(test_period_str_from_log_val, str):
                cleaned_test_period_str = re.sub(r'^[^\w\s\d:-]+|[^\w\s\d:-]+$', '', test_period_str_from_log_val.strip())
                if ' to ' in cleaned_test_period_str:
                    try:
                        start_test_str, end_test_str = cleaned_test_period_str.split(' to ')
                        test_start_date = pd.to_datetime(start_test_str.strip())
                        test_end_date = pd.to_datetime(end_test_str.strip())
                        actuals_on_test_period_plot = historical_sales_series_plot[
                            (historical_sales_series_plot.index >= test_start_date) &
                            (historical_sales_series_plot.index <= test_end_date)
                        ]
                        if actuals_on_test_period_plot.empty:
                             st.info(f"No actual sales data found in the logged test period ({test_start_date.date()} to {test_end_date.date()}) for S{selected_store}D{selected_dept}.")
                    except Exception as e_date_parse:
                        st.warning(f"Could not parse test period dates ('{test_period_str_from_log_val}') from logs: {e_date_parse}")
                        test_start_date, test_end_date = None, None # Reset on error
            else:
                st.info("Test period dates not clearly defined or not a string in logs for this model run. Cannot plot test period actuals accurately from log.")

            # --- Placeholder for Generating Test Period & Future Forecasts ---
            # This section needs to be implemented with actual prediction logic for each model type.
            # It would use `loaded_model_object`, `historical_data_sd`, `model_info_selected_row` (for features, scalers),
            # and utility functions like `preprocess_regression_features`, `engineer_all_demand_features`, `create_sequences`.

            st.markdown("---")
            st.markdown("### 🔮 Forecast Generation (Placeholders)")
            st.info(
                "The actual forecast generation logic for the test period and future predictions "
                "needs to be implemented here based on the selected model type (Classical Time Series, "
                "Global Regression, or LSTM)."
            )

            if test_start_date and test_end_date and actuals_on_test_period_plot is not None and not actuals_on_test_period_plot.empty :
                st.markdown("#### Test Period Predictions (Illustrative Placeholder)")
                # Example: test_period_predictions_series = generate_test_preds(loaded_model_object, test_data_features, ...)
                # For now, create a NaN series to show on plot
                test_period_predictions_series = pd.Series(np.nan, index=actuals_on_test_period_plot.index, name="Test Predictions (Placeholder)")
                st.caption("Displaying NaN for test period predictions as the generation logic is a placeholder.")
            else:
                 st.caption("Test period actuals not available from logs to overlay predictions.")


            if future_forecast_weeks > 0 and not historical_sales_series_plot.empty:
                st.markdown("#### Future Forecast (Illustrative Placeholder)")
                # Example: future_period_predictions_series = generate_future_preds(loaded_model_object, historical_data_sd, future_forecast_weeks, ...)
                last_hist_date = historical_sales_series_plot.index.max()
                # Infer frequency of historical data, default to Weekly-Friday if not inferable
                data_freq = pd.infer_freq(historical_sales_series_plot.index) or 'W-FRI'
                future_pred_dates = pd.date_range(
                    start=last_hist_date + pd.Timedelta(days=1 if data_freq[0]!='W' else 7), # Start after last known date
                    periods=future_forecast_weeks,
                    freq=data_freq
                )
                future_period_predictions_series = pd.Series(np.nan, index=future_pred_dates, name="Future Forecast (Placeholder)")
                st.caption("Displaying NaN for future forecasts as the generation logic is a placeholder.")
            elif future_forecast_weeks > 0:
                st.warning("Cannot generate future forecast placeholder as historical sales data is empty.")

            # --- Plotting ---
            st.markdown("---")
            st.markdown("### 📊 Sales Forecast vs Actuals Visualization")
            if not historical_sales_series_plot.empty:
                fig, ax = plt.subplots(figsize=(18, 7)) # Adjusted size for better readability
                ax.plot(historical_sales_series_plot.index, historical_sales_series_plot.values,
                        label='Actual Historical Sales', color='cornflowerblue', alpha=0.8, linewidth=1.5)

                if actuals_on_test_period_plot is not None and not actuals_on_test_period_plot.empty:
                    ax.plot(actuals_on_test_period_plot.index, actuals_on_test_period_plot.values,
                            label='Actual (Logged Test Period)', color='darkgreen', linestyle='-', marker='o', markersize=4, linewidth=2, alpha=0.9)

                if test_period_predictions_series is not None and not test_period_predictions_series.empty:
                    ax.plot(test_period_predictions_series.index, test_period_predictions_series.values,
                            label=f'{model_info_selected_row["Model_Display_Name_Base"]} (Test Period Pred. - Placeholder)',
                            color='darkorange', linestyle='--', linewidth=2)

                if future_period_predictions_series is not None and not future_period_predictions_series.empty:
                    ax.plot(future_period_predictions_series.index, future_period_predictions_series.values,
                            label=f'{model_info_selected_row["Model_Display_Name_Base"]} (Future Forecast - Placeholder)',
                            color='purple', linestyle=':', linewidth=2)

                ax.set_title(f"Sales Forecast vs Actuals for S{selected_store}-D{selected_dept} ({model_info_selected_row['Model_Display_Name_Base']})", fontsize=16)
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("Weekly Sales", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig) # Important to free memory
            else:
                st.warning("No historical sales data available for the selected Store-Department to plot.")
        else:
            # Error messages from load_selected_model will be displayed above this.
            st.markdown(
                "**Further Troubleshooting if Model Loading Failed:**"
                "\n1. **Verify Training Completion:** Ensure the specific training script (e.g., for "
                f"'{selected_model_name_from_log}') ran without errors and saved the model artifact."
                "\n2. **Check Experiment Logs:** Open the relevant CSV log file in "
                "`reports/experiment_logs/`. Find the entry for this model, Store, and Dept. "
                "Confirm the `model_path` listed there is correct, accessible, and points to an existing file. "
                "It should ideally be an absolute path or a path easily resolvable from the project root."
                "\n3. **File System Check:** Manually navigate to the `models_store/demand_forecasting/` "
                "directory and verify the model file exists with the exact name and extension."
                "\n4. **PROJECT_ROOT Consistency:** This app and all training scripts must determine `PROJECT_ROOT` "
                "identically for paths logged by training scripts to be valid when loaded by the app."
            )
            st.write("Full parameters dictionary from log for this model selection (contains paths logged during training):")
            st.json(model_info_selected_row.get('Parameters_Dict', {}))


    else: # If initial selections for store, dept, model are not all made
        st.info("Please select a Store, Department, and a trained Model using the controls above to view detailed forecasts and performance.")


if __name__ == "__main__": # pragma: no cover
    # This block allows for standalone testing of this page module.
    # It's crucial that PROJECT_ROOT is correctly inferred or paths are adjusted
    # for data, logs, and utility function loading.
    # Main application typically handles global page configuration like title and layout.
    # st.set_page_config(layout="wide", page_title="Forecast Explorer") # Usually in main_app.py

    if not _UTIL_FUNCS_LOADED_FC_EXPLORER:
         st.error(
            "Standalone run: Critical utility functions (preprocessing, feature engineering) "
            "failed to load. Page functionality will be severely impaired or non-functional."
        )
    render_forecast_explorer_page()