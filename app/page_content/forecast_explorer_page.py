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
import seaborn as sns # Seaborn is imported but not explicitly used in the provided snippets. Kept for potential use.
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
    MinMaxScaler = None
    st.info("TensorFlow or scikit-learn not found. LSTM model functionality will be unavailable.")

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
# These will be populated by imports from the 'src' directory.
# Defining them as None initially helps in robust checking later.
preprocess_regression_features = None
engineer_all_demand_features = None
create_sequences = None

def _get_project_root_fc_explorer() -> str:
    """
    Determines the project root directory for this page.

    Assumes this script (`forecast_explorer_page.py`) is located within:
    `PROJECT_ROOT/app/page_content/`.
    It navigates up two directories from this file's current location.

    Returns:
        str: The absolute path to the project root directory.

    Raises:
        NameError: If `__file__` is not defined, preventing path determination.
                   A Streamlit warning is also issued in this case.
    """
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up: app/page_content -> app -> PROJECT_ROOT
        return os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    except NameError: # pragma: no cover
        st.warning(
            "Could not determine project root via `__file__` (it's undefined). "
            "Falling back to current working directory. Module imports or data paths might fail."
        )
        return os.getcwd()

# --- Project Path Setup & Custom Module Imports ---
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
        # Attempt primary location for create_sequences
        from utils.preprocessing import create_sequences
    except ImportError:
        # Fallback location if not found in preprocessing (or if structure changed)
        from utils.feature_engineering_utils import create_sequences
    UTIL_FUNCS_LOADED = True
except Exception as e_setup: # pragma: no cover
    UTIL_FUNCS_LOADED = False
    st.error(f"CRITICAL ERROR during sys.path setup or utility imports: {e_setup}")
    st.info(
        "This page requires utility functions from 'src/utils/'. "
        "Ensure this directory and its Python files (including '__init__.py' "
        "in 'src' and 'src/utils') are correctly structured and importable."
    )
    # Define dummy functions if critical imports failed, to allow app to show error gracefully
    if 'preprocess_regression_features' not in globals() or preprocess_regression_features is None:
        def preprocess_regression_features(*args, **kwargs):
            st.error("Utility 'preprocess_regression_features' is unavailable."); return None, None, None, None
    if 'create_sequences' not in globals() or create_sequences is None:
        def create_sequences(*args, **kwargs):
            st.error("Utility 'create_sequences' is unavailable."); return np.empty((0,0,0)), np.empty((0,))
    if 'engineer_all_demand_features' not in globals() or engineer_all_demand_features is None:
        def engineer_all_demand_features(df_input, **kwargs):
            st.error("Utility 'engineer_all_demand_features' is unavailable."); return df_input

# --- Configuration Paths (derived from PROJECT_ROOT) ---
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
FEATURED_DATA_FILENAME = 'walmart_data_featured.parquet'
EXPERIMENT_LOGS_PATH = os.path.join(PROJECT_ROOT, 'reports', 'experiment_logs')
MODEL_STORE_PATH = os.path.join(PROJECT_ROOT, 'models_store', 'demand_forecasting')
# SCALER_STORE_PATH = os.path.join(MODEL_STORE_PATH, 'scalers') # Defined but not used in provided snippet

@st.cache_data
def load_all_experiment_logs(log_path: str = EXPERIMENT_LOGS_PATH) -> pd.DataFrame | None:
    """
    Loads and combines all relevant experiment log CSV files.

    It searches for 'classical_timeseries_experiments.csv',
    'classical_regression_experiments.csv', and 'deep_learning_experiments.csv'
    in the specified log_path. It parses parameters and feature lists from
    string representations in the logs.

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

    def _safe_literal_eval(val_str, target_type=dict):
        """Safely evaluate a string representation of a Python literal (dict or list)."""
        if isinstance(val_str, target_type): return val_str
        if isinstance(val_str, str):
            try: return ast.literal_eval(val_str)
            except (ValueError, SyntaxError, TypeError): pass
        return target_type() # Return empty dict or list

    for log_type, file_name in log_files_info.items():
        full_path = os.path.join(log_path, file_name)
        if os.path.exists(full_path):
            try:
                df_log = pd.read_csv(full_path)
                essential_cols = ['Store', 'Dept', 'Model', 'Parameters', 'Test_Period']
                missing_essentials = [col for col in essential_cols if col not in df_log.columns]
                if missing_essentials:
                    st.warning(f"Essential columns {missing_essentials} missing in '{file_name}'. Some functionality may be affected.")
                df_log['Log_Type_Source'] = log_type
                all_logs_list.append(df_log)
            except Exception as e:
                st.warning(f"Could not load or process log file '{file_name}': {e}")
        else:
            st.info(f"Optional log file not found: '{full_path}'.")

    if not all_logs_list:
        st.error("No experiment logs found or loaded. Cannot proceed with forecast exploration.")
        return None

    df_combined = pd.concat(all_logs_list, ignore_index=True)

    # Parse parameters and feature lists
    if 'Parameters' in df_combined.columns:
        df_combined['Parameters_Dict'] = df_combined['Parameters'].apply(lambda x: _safe_literal_eval(x, dict))
        df_combined['model_path'] = df_combined['Parameters_Dict'].apply(lambda p_dict: p_dict.get('model_path'))
    else:
        df_combined['Parameters_Dict'] = pd.Series([{} for _ in range(len(df_combined))])
        df_combined['model_path'] = None

    df_combined['Parsed_Features_List'] = pd.Series([[] for _ in range(len(df_combined))], dtype=object)
    if 'Features_List_Used' in df_combined.columns:
        df_combined['Parsed_Features_List'] = df_combined['Features_List_Used'].apply(lambda x: _safe_literal_eval(x, list))

    def get_features_from_various_sources(row):
        """Consolidates feature lists from 'Parsed_Features_List' or various keys in 'Parameters_Dict'."""
        if isinstance(row['Parsed_Features_List'], list) and row['Parsed_Features_List']:
            return row['Parsed_Features_List']
        params_dict = row.get('Parameters_Dict', {})
        # Check common keys where feature lists might be stored in parameters
        keys_to_check = [
            'features_used_list_in_params', 'features_used_for_lstm_input',
            'sarima_exog_features_actually_used_by_model', 'prophet_regressors_used',
            'features_used_list', 'features_used'
        ]
        for key in keys_to_check:
            features = params_dict.get(key)
            if isinstance(features, list) and features:
                return features
        return []
    df_combined['Parsed_Features_List'] = df_combined.apply(get_features_from_various_sources, axis=1)

    # Standardize numeric columns
    for col in ['Store', 'Dept']:
        if col in df_combined.columns:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce').fillna(-1).astype(int)
    for metric in ['MAE', 'RMSE', 'MAPE']:
        if metric in df_combined.columns:
            df_combined[metric] = pd.to_numeric(df_combined[metric], errors='coerce')
    return df_combined

@st.cache_data
def load_main_featured_data(data_path: str = PROCESSED_DATA_PATH, filename: str = FEATURED_DATA_FILENAME) -> pd.DataFrame | None:
    """
    Loads the main preprocessed and feature-engineered dataset.

    Supports Parquet and CSV formats. Ensures 'Date' column is datetime
    and 'Type_Encoded' exists if 'Type' column is present.

    Args:
        data_path: Directory path containing the data file.
        filename: Name of the data file (e.g., 'data.parquet' or 'data.csv').

    Returns:
        A pandas DataFrame with the featured data, or None if loading fails.
    """
    full_path = os.path.join(data_path, filename)
    if not os.path.exists(full_path):
        st.error(f"Featured data file not found: {full_path}")
        return None
    try:
        if full_path.endswith('.parquet'):
            df = pd.read_parquet(full_path)
        elif full_path.endswith('.csv'):
            df = pd.read_csv(full_path, parse_dates=['Date'])
        else:
            st.error(f"Unsupported data file format: {full_path}. Only .parquet and .csv are supported.")
            return None

        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure Type_Encoded for regression models if 'Type' exists and 'Type_Encoded' doesn't
        if 'Type' in df.columns and 'Type_Encoded' not in df.columns:
            from sklearn.preprocessing import LabelEncoder # Local import is fine here
            le = LabelEncoder()
            df['Type_Encoded'] = le.fit_transform(df['Type'].astype(str)) # Ensure 'Type' is string for LE
        return df
    except Exception as e:
        st.error(f"Error loading featured data from '{full_path}': {e}")
        return None

@st.cache_resource # Models can be large, use cache_resource
def load_selected_model(model_path_str: str | None, model_name_to_load: str | None):
    """
    Loads a saved machine learning model based on its path and name hint.

    Supports TensorFlow/Keras (.keras, .h5), XGBoost (.json, .ubj),
    LightGBM (.txt), Prophet (.json), and generic .joblib models.

    Args:
        model_path_str: Filesystem path to the saved model.
        model_name_to_load: A name indicating the type of model (e.g., 'LSTM', 'XGBoost').

    Returns:
        The loaded model object, or None if loading fails or model type is unsupported.
    """
    if not model_path_str or not isinstance(model_path_str, str) or not os.path.exists(model_path_str):
        st.error(f"Model path is invalid or file does not exist: '{model_path_str}'")
        return None
    if not model_name_to_load:
        st.error("Model name (type hint) not provided for loading.")
        return None

    st.info(f"Attempting to load model: '{model_name_to_load}' from path: '{model_path_str}'")
    try:
        model_name_lower = model_name_to_load.lower()
        if model_name_lower.startswith('lstm') and model_path_str.endswith(('.keras', '.h5')):
            if tf: return tf.keras.models.load_model(model_path_str)
            else: st.error("TensorFlow not available for LSTM model loading."); return None
        elif model_name_lower.startswith('xgboost') and model_path_str.endswith(('.json', '.ubj')):
            if xgb: model = xgb.XGBRegressor(); model.load_model(model_path_str); return model
            else: st.error("XGBoost not available for XGBoost model loading."); return None
        elif model_name_lower.startswith('lightgbm') and model_path_str.endswith('.txt'):
            if lgb: return lgb.Booster(model_file=model_path_str)
            else: st.error("LightGBM not available for LightGBM model loading."); return None
        elif model_name_lower.startswith('prophet') and model_path_str.endswith('.json'):
            if Prophet and model_from_json:
                with open(model_path_str, 'r') as fin: return model_from_json(fin.read())
            else: st.error("Prophet not available for Prophet model loading."); return None
        elif model_path_str.endswith('.joblib'): # Generic fallback for scikit-learn, pmdarima, etc.
            return joblib.load(model_path_str)
        else:
            st.warning(f"Unsupported model type or file extension for '{model_name_to_load}' from '{model_path_str}'.")
            return None
    except Exception as e:
        st.error(f"Error loading model '{model_name_to_load}' from '{model_path_str}': {e}")
        return None

# --- Main function for the Forecast Explorer page ---
def render_forecast_explorer_page():
    """
    Renders the main content for the Forecast Visualizer & Explorer page.

    This function orchestrates UI for selecting store, department, and model,
    loads relevant data and models, displays performance metrics, and
    (if implemented) generates and plots test period and future forecasts.
    """
    st.header("🚀 Forecast Visualizer & Explorer")
    st.markdown("Select Store, Department, and Model to visualize its performance and generate future forecasts.")

    if not UTIL_FUNCS_LOADED: # pragma: no cover
        st.error(
            "One or more critical utility functions (e.g., for preprocessing, feature engineering) "
            "failed to load during the initial page setup. The Forecast Explorer page cannot operate correctly. "
            "Please check the console output and the import statements at the top of this script."
        )
        st.stop() # Halt further execution of this page if setup failed

    df_logs = load_all_experiment_logs()
    df_featured_full = load_main_featured_data()

    if df_logs is None or df_logs.empty or df_featured_full is None:
        st.warning(
            "Required experiment logs or the main featured dataset could not be loaded. "
            "Forecast Explorer functionality is limited or unavailable. "
            "Please check data paths and log file generation."
        )
        return # Stop if essential data isn't available

    # --- User Selections ---
    st.subheader("⚙️ Select Forecast Parameters")
    sel_col1, sel_col2, sel_col3, sel_col4 = st.columns(4)

    unique_stores = sorted([s for s in df_logs['Store'].unique() if s != -1 and s != 0])
    global_model_store_placeholder = 0 # Convention for global models

    selected_store = None
    if not unique_stores:
        with sel_col1: st.warning("No specific store data found in logs.")
    else:
        with sel_col1:
            selected_store = st.selectbox(
                "Store:", unique_stores, index=0, key="fc_store_select"
            )

    selected_dept = None
    depts_for_store = [] # Initialize to handle cases where selected_store might become None
    if selected_store:
        depts_for_store = sorted([
            d for d in df_logs[df_logs['Store'] == selected_store]['Dept'].unique() if d != -1 and d != 0
        ])
        if not depts_for_store:
            with sel_col2: st.warning(f"No specific department data for Store {selected_store} in logs.")
        else:
            with sel_col2:
                selected_dept = st.selectbox(
                    "Department:", depts_for_store, index=0, key="fc_dept_select"
                )
    elif unique_stores: # Only show if stores were available but none selected yet
        with sel_col2: st.info("Select a Store to see available Departments.")


    selected_model_name_from_log = None
    model_info = None
    if selected_store and selected_dept:
        store_specific_models_df = df_logs[
            (df_logs['Store'] == selected_store) &
            (df_logs['Dept'] == selected_dept) &
            (df_logs['model_path'].notna())
        ].copy()
        if not store_specific_models_df.empty:
            store_specific_models_df['Model_Display_Name_Base'] = store_specific_models_df['Model']

        global_models_df = df_logs[
            (df_logs['Store'] == global_model_store_placeholder) &
            (df_logs['Dept'] == global_model_store_placeholder) & # Assuming global models also use a placeholder for Dept
            (df_logs['model_path'].notna())
        ].copy()
        if not global_models_df.empty:
            global_models_df['Model_Display_Name_Base'] = global_models_df['Model'] + " (Global)"

        all_relevant_models_df = pd.concat([store_specific_models_df, global_models_df], ignore_index=True)

        if not all_relevant_models_df.empty:
            all_relevant_models_df['display_name_with_metrics'] = (
                all_relevant_models_df['Model_Display_Name_Base'] + " (RMSE: " +
                all_relevant_models_df['RMSE'].round(2).astype(str) + ")"
            )
            available_models_display_list = all_relevant_models_df['display_name_with_metrics'].unique().tolist()

            if available_models_display_list:
                with sel_col3:
                    selected_model_display_name = st.selectbox(
                        "Model:", available_models_display_list, index=0, key="fc_model_select"
                    )
                # This logic must be outside 'with sel_col3' to correctly use the selected value
                if selected_model_display_name:
                    model_info_row = all_relevant_models_df[
                        all_relevant_models_df['display_name_with_metrics'] == selected_model_display_name
                    ]
                    if not model_info_row.empty:
                        model_info = model_info_row.iloc[0].copy()
                        selected_model_name_from_log = model_info['Model']
            else:
                with sel_col3: st.warning(f"No loadable models found for S{selected_store}-D{selected_dept} or globally.")
        else:
            with sel_col3: st.warning(f"No models (store-specific or global) found for S{selected_store}-D{selected_dept} with model paths.")
    elif selected_store and not selected_dept and depts_for_store: # Store selected, departments were available for it
        with sel_col3: st.info("Select a Department to see available models.")
    elif not selected_store and unique_stores: # No store selected, but stores are available
         with sel_col3: st.info("Select Store and Department.")


    with sel_col4:
        future_forecast_weeks = st.number_input(
            "Future Weeks to Forecast:", min_value=0, max_value=52, value=12, step=4, key="fc_future_weeks"
        )

    st.markdown("---")

    # --- Main Display Area for Forecasts and Model Info ---
    if selected_store and selected_dept and selected_model_name_from_log and model_info is not None:
        st.subheader(f"Results for Store {selected_store} - Dept {selected_dept} using {model_info['Model_Display_Name_Base']}")

        # Display Metrics and Parameters in Expanders
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            with st.expander("Logged Test Set Performance", expanded=True):
                metrics_to_show = {
                    k: model_info[k] for k in ['MAE', 'RMSE', 'MAPE']
                    if k in model_info and pd.notna(model_info[k])
                }
                if metrics_to_show: st.json(metrics_to_show)
                else: st.write("Performance metrics not available or not logged for this model.")
        with exp_col2:
            with st.expander("Model Hyperparameters & Logged Info", expanded=True):
                params_dict_to_display = model_info.get('Parameters_Dict', {})
                # Keys to exclude for a cleaner display of primary hyperparameters
                keys_to_exclude = [
                    'model_path', 'feature_scaler_path', 'target_scaler_path',
                    'features_used_list_in_params', 'features_used_for_lstm_input',
                    'features_used_list', 'features_used', 'train_period_in_params',
                    'test_period_in_params', 'sarima_exog_features_offered',
                    'sarima_exog_features_actually_used_by_model',
                    'prophet_regressors_used', 'input_shape_str', 'error'
                ]
                params_to_show = {
                    k: v for k, v in params_dict_to_display.items() if k not in keys_to_exclude
                }
                if params_to_show: st.json(params_to_show)
                else: st.write("Key hyperparameters not found or all were filtered from display.")
                st.caption(f"Full Parameters String (from log): `{model_info.get('Parameters', 'Not available')}`")
                st.caption(f"Features Used (parsed): `{model_info.get('Parsed_Features_List', [])}`")

        model_path = model_info.get('model_path')
        loaded_model = load_selected_model(model_path, selected_model_name_from_log)

        if loaded_model:
            st.success(f"Model '{selected_model_name_from_log}' loaded successfully from '{model_path}'!")

            historical_data_for_store_dept = df_featured_full[
                (df_featured_full['Store'] == selected_store) &
                (df_featured_full['Dept'] == selected_dept)
            ].copy()
            if 'Date' not in historical_data_for_store_dept.columns:
                st.error("Critical error: 'Date' column missing in filtered historical data.")
                return
            
            historical_data_series = historical_data_for_store_dept.set_index('Date')['Weekly_Sales'].sort_index()

            test_predictions_series, future_predictions_series = None, None
            test_period_actuals_for_plot, start_test_date, end_test_date = None, None, None

            # Attempt to parse test period from logs
            params_dict_from_log = model_info.get('Parameters_Dict', {})
            test_period_str_from_log = params_dict_from_log.get('test_period_in_params',
                                                                params_dict_from_log.get('test_period',
                                                                                         model_info.get('Test_Period')))
            if isinstance(test_period_str_from_log, str):
                # Clean potential non-standard characters from string before splitting
                cleaned_test_period_str = re.sub(r'^[^\w\s\d:-]+|[^\w\s\d:-]+$', '', test_period_str_from_log.strip())
                if ' to ' in cleaned_test_period_str:
                    try:
                        start_test_str, end_test_str = cleaned_test_period_str.split(' to ')
                        start_test_date = pd.to_datetime(start_test_str.strip())
                        end_test_date = pd.to_datetime(end_test_str.strip())
                        test_period_actuals_for_plot = historical_data_series[
                            (historical_data_series.index >= start_test_date) &
                            (historical_data_series.index <= end_test_date)
                        ]
                    except Exception as e_dates:
                        st.warning(f"Could not parse test period dates ('{test_period_str_from_log}') from logs: {e_dates}")
                        start_test_date, end_test_date = None, None
            else:
                st.info("Test period dates not clearly defined in logs for this model run.")

            n_test_periods_from_log = 0
            if start_test_date and end_test_date:
                try:
                    # Infer frequency if not set, default to weekly Friday
                    freq = historical_data_series.index.freqstr or pd.infer_freq(historical_data_series.index) or 'W-FRI'
                    forecast_dates_test_period = pd.date_range(start=start_test_date, end=end_test_date, freq=freq)
                    n_test_periods_from_log = len(forecast_dates_test_period)
                except Exception as e_freq:
                    st.warning(f"Could not determine frequency for test period date range: {e_freq}.")


            # --- Placeholder for Test Period & Future Forecast Generation Logic ---
            # This is where the complex prediction logic for each model type would go.
            # It would use loaded_model, historical_data_for_store_dept, utility functions, etc.
            # For now, these will remain None or empty.

            st.info(
                "Prediction logic for test period and future forecasts is currently a placeholder. "
                "Implement specific forecasting routines for each model type (Classical TS, Global Regression, LSTM) here."
            )
            if start_test_date and end_test_date and n_test_periods_from_log > 0:
                st.markdown("#### Test Period Predictions (Placeholder)")
                # Example: test_predictions_series = generate_test_predictions(loaded_model, ...)
                if test_period_actuals_for_plot is not None and not test_period_actuals_for_plot.empty:
                     test_predictions_series = pd.Series(np.nan, index=test_period_actuals_for_plot.index) # Placeholder
                     st.caption("Displaying NaN for test predictions as logic is not yet implemented.")


            if future_forecast_weeks > 0:
                st.markdown("#### Future Forecast (Placeholder)")
                # Example: future_predictions_series = generate_future_forecasts(loaded_model, ...)
                if not historical_data_series.empty:
                    last_historical_date = historical_data_series.index.max()
                    future_dates = pd.date_range(
                        start=last_historical_date + pd.Timedelta(days=7 if (historical_data_series.index.freqstr or 'W') == 'W-FRI' else 1), # Adjust based on freq
                        periods=future_forecast_weeks,
                        freq=historical_data_series.index.freqstr or 'W-FRI'
                    )
                    future_predictions_series = pd.Series(np.nan, index=future_dates) # Placeholder
                    st.caption("Displaying NaN for future forecasts as logic is not yet implemented.")


            # --- Plotting ---
            if not historical_data_series.empty:
                fig, ax = plt.subplots(figsize=(18, 6))
                ax.plot(historical_data_series.index, historical_data_series.values,
                        label='Actual Historical Sales', color='blue', alpha=0.6)

                if test_period_actuals_for_plot is not None and not test_period_actuals_for_plot.empty:
                    ax.plot(test_period_actuals_for_plot.index, test_period_actuals_for_plot.values,
                            label='Actual (Logged Test Period)', color='green', linestyle='-', linewidth=2, alpha=0.8)

                if test_predictions_series is not None and not test_predictions_series.empty:
                    ax.plot(test_predictions_series.index, test_predictions_series.values,
                            label=f'{model_info["Model_Display_Name_Base"]} (Test Period Pred.)', color='orange', linestyle='--')

                if future_predictions_series is not None and not future_predictions_series.empty:
                    ax.plot(future_predictions_series.index, future_predictions_series.values,
                            label=f'{model_info["Model_Display_Name_Base"]} (Future Forecast)', color='purple', linestyle=':')

                ax.set_title(f"Sales Forecast vs Actuals for S{selected_store}-D{selected_dept} ({model_info['Model_Display_Name_Base']})")
                ax.set_xlabel("Date")
                ax.set_ylabel("Weekly Sales")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig) # Important to free memory
            else:
                st.warning("No historical sales data available for the selected Store-Department to plot.")
        else:
            st.error(f"Could not load the selected model: {selected_model_name_from_log}. "
                     f"Path attempted: {model_path}")
    else:
        st.info("Please select a Store, Department, and Model using the controls above to view forecasts and model details.")


if __name__ == "__main__": # pragma: no cover
    # This block allows for standalone testing of this page module.
    # Ensure that PROJECT_ROOT is correctly inferred or paths are adjusted
    # for data and utility function loading.
    # Main application typically handles page configuration.
    # st.set_page_config(layout="wide", page_title="Forecast Explorer")
    if not UTIL_FUNCS_LOADED:
         st.error("Standalone run: Critical utility functions failed to load. Page may not function.")
    render_forecast_explorer_page()