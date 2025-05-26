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
import seaborn as sns
import os
import sys
import joblib
import ast
import re

# --- Optional Deep Learning Imports ---
# Wrapped in try-except to allow the app to run even if some are missing,
# with functionality gracefully degraded or noted to the user.
try:
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler # Often used with neural networks
    if tf.__version__.startswith('1.'): # pragma: no cover
        # Using st.sidebar for app-level warnings that don't clutter the main page
        st.sidebar.warning("TensorFlow 1.x detected. TensorFlow 2.x or higher is generally preferred for new projects.", icon="⚠️")
except ImportError: # pragma: no cover
    tf = None
    MinMaxScaler = None
    st.sidebar.info("TensorFlow or scikit-learn (for MinMaxScaler) not fully available. LSTM model functionality will be limited or unavailable.")

# --- Optional Gradient Boosting Imports ---
try:
    import xgboost as xgb
except ImportError: # pragma: no cover
    xgb = None
try:
    import lightgbm as lgb
except ImportError: # pragma: no cover
    lgb = None

# --- Optional Prophet Import ---
try:
    from prophet import Prophet
    from prophet.serialize import model_from_json
except ImportError: # pragma: no cover
    Prophet = None
    model_from_json = None


# --- Utility Function Placeholders (if main imports fail) ---
# These act as fallbacks if the primary utility imports fail due to path issues.
def _placeholder_preprocess_regression_features(*args, **kwargs):
    st.error("CRITICAL: Utility 'preprocess_regression_features' is unavailable due to import issues."); return None, None, None, None
def _placeholder_engineer_all_demand_features(df_input, **kwargs):
    st.error("CRITICAL: Utility 'engineer_all_demand_features' is unavailable."); return df_input if df_input is not None else pd.DataFrame()
def _placeholder_create_sequences(*args, **kwargs):
    st.error("CRITICAL: Utility 'create_sequences' is unavailable."); return np.empty((0,0,0)), np.empty((0,))

preprocess_regression_features = _placeholder_preprocess_regression_features
engineer_all_demand_features = _placeholder_engineer_all_demand_features
create_sequences = _placeholder_create_sequences


def _get_project_root_fc_explorer() -> str:
    """
    Determines the project root directory for the Forecast Explorer page.

    This function assumes the script is located within a subdirectory like
    `PROJECT_ROOT/app/page_content/` and navigates up to the project root.
    This is essential for consistent relative path resolution for data, models,
    and utility modules, especially when running pages standalone or as part
    of a larger application.

    Returns:
        str: The absolute path to the determined project root directory.
             Returns current working directory as a fallback if `__file__` is undefined,
             which may occur in some non-standard execution environments.
    """
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up: app/page_content -> app -> PROJECT_ROOT
        return os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    except NameError: # pragma: no cover
        st.warning(
            "Could not automatically determine project root using `__file__` (it was undefined). "
            "Falling back to current working directory. Path-dependent features (data/model loading, utils) "
            "may not function correctly if this page is run standalone without the project root as CWD."
        )
        return os.getcwd()

# --- Project Path Setup & Custom Module Imports ---
_UTIL_FUNCS_LOADED_FC_EXPLORER = False
PROJECT_ROOT = _get_project_root_fc_explorer()
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

# Dynamically add project root and src to sys.path if not already present
if SRC_PATH not in sys.path: # pragma: no cover
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path: # pragma: no cover
    sys.path.insert(0, PROJECT_ROOT)

try:
    # Attempt to import actual utility functions
    from utils.preprocessing import preprocess_regression_features as actual_preprocess_reg
    from utils.feature_engineering_utils import engineer_all_demand_features as actual_engineer_all_demand
    try:
        from utils.preprocessing import create_sequences as actual_create_sequences
    except ImportError: # Fallback as per original user structure
        from utils.feature_engineering_utils import create_sequences as actual_create_sequences
    
    # If imports succeed, replace placeholders with actual functions
    preprocess_regression_features = actual_preprocess_reg
    engineer_all_demand_features = actual_engineer_all_demand
    create_sequences = actual_create_sequences
    _UTIL_FUNCS_LOADED_FC_EXPLORER = True
    print("Forecast Explorer: Core utility functions loaded successfully from 'src/utils/'.")
except Exception as e_setup: # pragma: no cover
    _UTIL_FUNCS_LOADED_FC_EXPLORER = False
    # Placeholders defined above will be used. Error message below informs user.
    st.error(
        f"CRITICAL ERROR initializing Forecast Explorer utilities from 'src/utils/': {e_setup}. "
        "This page requires specific utility functions for data preprocessing and feature engineering. "
        "Please ensure the 'src' directory, its sub-modules ('preprocessing.py', 'feature_engineering_utils.py'), "
        "and necessary '__init__.py' files are correctly structured and importable from the project root. "
        "Page functionality will be severely limited, using placeholder functions."
    )

# --- Configuration Paths (derived from PROJECT_ROOT) ---
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
FEATURED_DATA_FILENAME = 'walmart_data_featured.parquet' # Or .csv as per your data
EXPERIMENT_LOGS_PATH = os.path.join(PROJECT_ROOT, 'reports', 'experiment_logs')
MODEL_STORE_PATH = os.path.join(PROJECT_ROOT, 'models_store', 'demand_forecasting')
# SCALER_STORE_PATH = os.path.join(MODEL_STORE_PATH, 'scalers') # Defined but not explicitly used in provided load_selected_model

@st.cache_data(show_spinner="Loading and consolidating experiment logs...")
def load_all_experiment_logs(log_path: str = EXPERIMENT_LOGS_PATH) -> pd.DataFrame | None:
    """
    Loads, combines, and preprocesses all relevant experiment log CSV files.

    Aggregates logs from classical time series, classical regression, and deep
    learning experiments. Standardizes key columns, parses string representations
    of parameters and feature lists, and ensures metric columns are numeric.
    This consolidated log drives model selection and metadata retrieval.

    Args:
        log_path: Path to the directory containing experiment log CSV files.

    Returns:
        A pandas DataFrame combining all found logs, or None if loading fails.
    """
    log_files_info = {
        'classical_ts': 'classical_timeseries_experiments.csv',
        'classical_reg': 'classical_regression_experiments.csv',
        'deep_learning': 'deep_learning_experiments.csv'
    }
    all_logs_list = []

    def _safe_literal_eval_fc(val_str, target_type=dict):
        """Safely evaluate a string representation of a Python literal."""
        if isinstance(val_str, target_type): return val_str
        if isinstance(val_str, str):
            try: return ast.literal_eval(val_str)
            except (ValueError, SyntaxError, TypeError): pass
        return target_type()

    for log_type, file_name in log_files_info.items():
        full_path = os.path.join(log_path, file_name)
        if os.path.exists(full_path):
            try:
                df_log = pd.read_csv(full_path)
                df_log['Log_Type_Source'] = log_type
                all_logs_list.append(df_log)
            except Exception as e:
                st.warning(f"Could not load/process log file '{file_name}': {e}")
        else:
            st.info(f"Note: Log file '{file_name}' not found at '{full_path}'. This may be expected if not all model types were run or logged.")

    if not all_logs_list:
        st.error("CRITICAL: No experiment logs were found or loaded. Forecast Explorer functionality is unavailable.")
        return None

    df_combined = pd.concat(all_logs_list, ignore_index=True)

    # Robust parsing of Parameters and Features
    if 'Parameters' in df_combined.columns:
        df_combined['Parameters_Dict'] = df_combined['Parameters'].apply(lambda x: _safe_literal_eval_fc(x, dict))
        df_combined['model_path'] = df_combined['Parameters_Dict'].apply(lambda p_dict: p_dict.get('model_path') if isinstance(p_dict, dict) else None)
    else:
        df_combined['Parameters_Dict'] = pd.Series([{} for _ in range(len(df_combined))])
        df_combined['model_path'] = None

    df_combined['Parsed_Features_List'] = pd.Series([[] for _ in range(len(df_combined))], dtype=object)
    if 'Features_List_Used' in df_combined.columns:
        df_combined['Parsed_Features_List'] = df_combined['Features_List_Used'].apply(lambda x: _safe_literal_eval_fc(x, list))

    def _consolidate_feature_list(row):
        """Consolidates feature lists for display and later use."""
        parsed_list = row.get('Parsed_Features_List', [])
        if isinstance(parsed_list, list) and parsed_list: return parsed_list
        
        params_dict = row.get('Parameters_Dict', {})
        if not isinstance(params_dict, dict): return [] # Ensure params_dict is a dict

        feature_keys_in_params = [
            'features_used_list_in_params', 'features_used_for_lstm_input', 'features_used_for_regression',
            'sarima_exog_features_actually_used_by_model', 'prophet_regressors_used',
            'features_used_list', 'features_used', 'exog_features', 'regressors' # Added 'regressors'
        ]
        for key in feature_keys_in_params:
            features = params_dict.get(key)
            if isinstance(features, list) and features: return features
        return []
    df_combined['Parsed_Features_List'] = df_combined.apply(_consolidate_feature_list, axis=1)

    # Standardize key numeric and metric columns
    for col_name in ['Store', 'Dept']:
        if col_name in df_combined.columns:
            df_combined[col_name] = pd.to_numeric(df_combined[col_name], errors='coerce').fillna(-1).astype(int)
        else: df_combined[col_name] = -1 # Ensure column exists for consistent filtering
    
    for metric_name in ['MAE', 'RMSE', 'MAPE']:
        if metric_name in df_combined.columns:
            df_combined[metric_name] = pd.to_numeric(df_combined[metric_name], errors='coerce')
        else: df_combined[metric_name] = np.nan # Ensure column exists
    return df_combined

@st.cache_data(show_spinner="Loading primary featured dataset...")
def load_main_featured_data(data_path: str = PROCESSED_DATA_PATH, filename: str = FEATURED_DATA_FILENAME) -> pd.DataFrame | None:
    """
    Loads the main preprocessed and feature-engineered dataset.

    This dataset contains historical sales and all engineered features
    required for model prediction. Supports Parquet and CSV formats.
    Crucially parses the 'Date' column to datetime objects.

    Args:
        data_path: Directory path of the featured data file.
        filename: Name of the featured data file.

    Returns:
        A Pandas DataFrame with featured data, or None on failure.
    """
    full_path = os.path.join(data_path, filename)
    if not os.path.exists(full_path):
        st.error(f"CRITICAL: Main featured data file ('{filename}') not found at expected location: '{data_path}'. This file is essential for forecast generation.")
        return None
    try:
        if full_path.lower().endswith('.parquet'):
            df = pd.read_parquet(full_path)
        elif full_path.lower().endswith('.csv'):
            df = pd.read_csv(full_path) # Robust date parsing handled next
        else:
            st.error(f"Unsupported data file format: '{filename}'. Only .parquet and .csv are currently supported.")
            return None

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            if df['Date'].isnull().any():
                st.warning("Warning: Some 'Date' values in the featured dataset could not be parsed and were set to NaT. This may affect time series operations.")
        else:
            st.error("CRITICAL: The 'Date' column is missing from the featured dataset. Time series analysis and forecasting cannot proceed without it.")
            return None

        if 'Type' in df.columns and 'Type_Encoded' not in df.columns: # For regression model compatibility
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['Type_Encoded'] = le.fit_transform(df['Type'].astype(str))
            st.sidebar.info("Note: 'Type_Encoded' column was dynamically created from 'Type' for model compatibility.")
        return df
    except Exception as e:
        st.error(f"Error loading or processing main featured data from '{full_path}': {e}")
        return None

@st.cache_resource(ttl=3600, show_spinner="Loading selected model artifact from store...")
def load_selected_model(model_path_str: str | None, model_name_to_load: str | None):
    """
    Loads a saved machine learning model artifact from the specified path.

    Handles various model types (scikit-learn, TensorFlow/Keras, XGBoost,
    LightGBM, Prophet) based on model name hints and file extensions.
    Provides detailed error reporting for loading failures.

    Args:
        model_path_str: The string path to the saved model file.
        model_name_to_load: Name/type hint of the model for loading mechanism.

    Returns:
        The loaded model object if successful, otherwise None.
    """
    # Initial st.info moved to within the main function after path resolution for clarity
    if not model_path_str or not isinstance(model_path_str, str) or not model_path_str.strip():
        st.error("Invalid model path (None, empty, or not a string). Cannot load model.")
        return None
    if not model_name_to_load or not isinstance(model_name_to_load, str) or not model_name_to_load.strip():
        st.error("Invalid model name/type hint. Cannot determine how to load model.")
        return None

    # Resolve path: if not absolute, assume relative to PROJECT_ROOT
    resolved_model_path = model_path_str
    if not os.path.isabs(model_path_str):
        potential_abs_path = os.path.join(PROJECT_ROOT, model_path_str)
        if os.path.exists(potential_abs_path): # Check if this resolved path exists
            resolved_model_path = potential_abs_path
        # If potential_abs_path doesn't exist, we'll let the os.path.exists(resolved_model_path) below handle the error.
    
    st.info(f"Attempting to load model: '{model_name_to_load}' from resolved path: '{resolved_model_path}'")

    model_dir = os.path.dirname(resolved_model_path)
    if not os.path.isdir(model_dir): # Check directory after path resolution
        st.error(f"Model directory does not exist or is not valid: '{model_dir}'.")
        st.markdown(
            "**Troubleshooting Suggestions:**\n"
            "- Confirm model training scripts completed and saved artifacts to this structured location.\n"
            "- Ensure `PROJECT_ROOT` consistency between training environment and this application.\n"
            "- Verify the path in experiment logs is correct (absolute or reliably relative)."
        )
        return None

    if not os.path.exists(resolved_model_path):
        st.error(f"Model file artifact not found at: '{resolved_model_path}'.")
        st.markdown(
             f"**Troubleshooting Suggestions:**\n"
             f"- The training for '{model_name_to_load}' may not have run, failed, or saved the model elsewhere.\n"
             f"- Manually verify the model artifact's existence and path."
        )
        try:
            st.markdown(f"**Files found in expected model directory (`{model_dir}`):**")
            files_in_dir = os.listdir(model_dir)
            st.json(files_in_dir[:20] if files_in_dir else "No files found in this directory.")
        except Exception as e_listdir:
            st.warning(f"Could not list files in directory '{model_dir}': {e_listdir}")
        return None

    try:
        model_name_lower = model_name_to_load.lower()
        st.write(f"Loading '{model_name_lower}' artifact: `{os.path.basename(resolved_model_path)}`...")

        if model_name_lower.startswith('lstm') and (resolved_model_path.endswith(('.keras', '.h5')) or '.keras' in resolved_model_path or '.h5' in resolved_model_path):
            if tf: return tf.keras.models.load_model(resolved_model_path)
            else: st.error("TensorFlow library unavailable; cannot load LSTM model."); return None
        elif model_name_lower.startswith('xgboost') and (resolved_model_path.endswith(('.json', '.ubj')) or '.json' in resolved_model_path):
            if xgb: model = xgb.XGBRegressor(); model.load_model(resolved_model_path); return model
            else: st.error("XGBoost library unavailable; cannot load XGBoost model."); return None
        elif model_name_lower.startswith('lightgbm') and resolved_model_path.endswith('.txt'):
            if lgb: return lgb.Booster(model_file=resolved_model_path)
            else: st.error("LightGBM library unavailable; cannot load LightGBM model."); return None
        elif model_name_lower.startswith('prophet') and resolved_model_path.endswith('.json'):
            if Prophet and model_from_json:
                with open(resolved_model_path, 'r') as fin: return model_from_json(fin.read())
            else: st.error("Prophet library unavailable; cannot load Prophet model."); return None
        elif resolved_model_path.endswith('.joblib'):
            return joblib.load(resolved_model_path)
        else:
            st.warning(f"Model type '{model_name_to_load}' from file '{os.path.basename(resolved_model_path)}' "
                       "has an unsupported extension or unrecognized loading mechanism. Please check conventions.")
            return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model '{model_name_to_load}' from '{resolved_model_path}':")
        st.exception(e) # Provides full traceback for debugging
        return None

def render_forecast_explorer_page():
    """
    Renders the interactive Forecast Visualizer & Model Explorer page.

    This page facilitates a deep dive into individual forecasting model
    performance. Users select a store, department, and a specific trained model.
    The application then loads the model artifact, displays its logged
    performance metrics and key hyperparameters, and generates a comparative
    visualization of historical actual sales against (currently placeholder)
    model predictions for both the original test period and a user-defined
    future horizon.

    The primary value of this explorer is to enable qualitative assessment of
    forecast quality, complementing the quantitative metrics found on the
    Model Performance Comparison page. It helps understand how well a model
    captures specific time series patterns like trend, seasonality, and responses
    to special events for a given sales series.
    """
    st.header("🚀 Forecast Visualiser & Model Explorer") # Changed icon for consistency
    st.markdown("""
    Welcome to the **Forecast Explorer**! This interactive interface is designed for in-depth qualitative
    analysis of individual forecasting models. By visualising a model's predictions against actual historical data,
    you can gain a deeper understanding of its behaviour, strengths, and weaknesses for specific sales series.

    **How to Use This Page Effectively:**
    1.  **Define Scope:** Use the dropdowns below to select the `Store` and `Department` for the sales series you wish to analyse.
    2.  **Choose Model:** Select a `Model` from the list populated based on available trained artifacts for your chosen scope (or global models). Models are listed with their logged Root Mean Squared Error (RMSE) for quick reference.
    3.  **Set Forecast Horizon:** Specify the number of `Future Weeks to Forecast` to see how the model projects sales beyond the available historical data.
    
    The explorer will then load the selected model, display its documented test-set performance and hyperparameters,
    and generate a plot showing historical sales alongside (currently illustrative placeholder) model predictions for
    the test period and future forecasts. This visual inspection is crucial for:
    - Assessing how well the model captures trends, seasonality, and responses to special events.
    - Identifying potential biases or systematic errors in forecasts.
    - Building confidence in model selection beyond just quantitative metrics.
    """)

    if not _UTIL_FUNCS_LOADED_FC_EXPLORER:
        st.error(
            "CRITICAL: Core utility functions (for preprocessing/feature engineering) did not load. "
            "The Forecast Explorer's ability to prepare data for models and generate new forecasts "
            "will be non-functional or severely impaired. Please check `sys.path` setup and `src/utils/` imports."
        )
        # Consider st.stop() if the page is truly unusable
        # For now, allow it to proceed to load data/logs to show some UI.
        # return

    # Load essential data; if fails, provide clear messages and halt.
    df_logs = load_all_experiment_logs()
    if df_logs is None or df_logs.empty:
        st.error("Essential experiment logs could not be loaded. Forecast Explorer cannot operate without them. Please check log paths and generation.")
        return
        
    df_featured_full = load_main_featured_data()
    if df_featured_full is None or df_featured_full.empty:
        st.error("The main featured dataset (containing historical sales and features) could not be loaded. Forecast Explorer cannot operate. Please check data paths.")
        return

    st.markdown("---")
    st.subheader("⚙️ Step 1: Configure Your Forecast Exploration Parameters")
    st.markdown("Select the specific sales series (Store-Department), the trained forecasting model, and the desired future forecast period.")

    sel_col1, sel_col2, sel_col3, sel_col4 = st.columns(4)

    unique_stores_for_ui = sorted([s for s in df_logs['Store'].unique() if s != -1 and s != 0])
    global_model_store_identifier = 0

    selected_store_ui = None # Use a different variable name for UI selection
    if not unique_stores_for_ui:
        with sel_col1: st.info("No store-specific forecast logs currently available for selection.")
    else:
        with sel_col1:
            selected_store_ui = st.selectbox(
                "Select Store:", unique_stores_for_ui, index=0, key="fc_store_select_final",
                help="Choose the specific store for which you want to explore sales forecasts."
            )

    selected_dept_ui = None
    depts_for_store_ui_list = []
    if selected_store_ui:
        depts_for_store_ui_list = sorted([
            d for d in df_logs[df_logs['Store'] == selected_store_ui]['Dept'].unique() if d != -1 and d != 0
        ])
        if not depts_for_store_ui_list:
            with sel_col2: st.info(f"No department-specific forecast logs found for Store {selected_store_ui}.")
        else:
            with sel_col2:
                selected_dept_ui = st.selectbox(
                    "Select Department:", depts_for_store_ui_list, index=0, key="fc_dept_select_final",
                    help="Choose the department within the selected store."
                )
    elif unique_stores_for_ui :
        with sel_col2: st.info("Select a Store to populate available Departments.")


    selected_model_name_from_log_ui = None
    model_info_selected_row_ui = None
    
    if selected_store_ui and selected_dept_ui:
        # Filter for models specific to the selected Store-Dept
        store_dept_models_df_ui = df_logs[
            (df_logs['Store'] == selected_store_ui) &
            (df_logs['Dept'] == selected_dept_ui) &
            (df_logs['model_path'].notna()) & (df_logs['model_path'].str.strip() != "")
        ].copy()
        if not store_dept_models_df_ui.empty:
            store_dept_models_df_ui['Model_Display_Name_Base'] = store_dept_models_df_ui['Model']

        # Filter for global models (Store=0, Dept=0 by convention)
        global_models_df_ui = df_logs[
            (df_logs['Store'] == global_model_store_identifier) &
            (df_logs['Dept'] == global_model_store_identifier) &
            (df_logs['model_path'].notna()) & (df_logs['model_path'].str.strip() != "")
        ].copy()
        if not global_models_df_ui.empty:
            global_models_df_ui['Model_Display_Name_Base'] = global_models_df_ui['Model'] + " (Global)"

        # Combine, ensuring no duplicates if a model name could appear in both (unlikely with current logic)
        all_relevant_models_df_ui = pd.concat([store_dept_models_df_ui, global_models_df_ui], ignore_index=True)
        all_relevant_models_df_ui.drop_duplicates(subset=['Model_Display_Name_Base', 'model_path'], inplace=True)


        if not all_relevant_models_df_ui.empty:
            if 'RMSE' in all_relevant_models_df_ui.columns:
                all_relevant_models_df_ui['display_name_with_metrics'] = (
                    all_relevant_models_df_ui['Model_Display_Name_Base'].astype(str) + " (RMSE: " +
                    all_relevant_models_df_ui['RMSE'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A") + ")"
                )
            else:
                all_relevant_models_df_ui['display_name_with_metrics'] = all_relevant_models_df_ui['Model_Display_Name_Base'].astype(str) + " (RMSE: Not Logged)"
                st.sidebar.warning("RMSE not found in logs for some models; model selection display may lack this metric.")

            available_models_display_list_ui = sorted(all_relevant_models_df_ui['display_name_with_metrics'].unique().tolist())

            if available_models_display_list_ui:
                with sel_col3:
                    selected_model_display_name_ui = st.selectbox(
                        "Select Model:", available_models_display_list_ui, index=0, key="fc_model_select_final",
                        help="Choose a trained model. 'Global' models are trained on all data. Logged RMSE is shown for quick reference."
                    )
                if selected_model_display_name_ui:
                    model_info_row_df_matches_ui = all_relevant_models_df_ui[
                        all_relevant_models_df_ui['display_name_with_metrics'] == selected_model_display_name_ui
                    ]
                    if not model_info_row_df_matches_ui.empty:
                        model_info_selected_row_ui = model_info_row_df_matches_ui.iloc[0].copy()
                        selected_model_name_from_log_ui = model_info_selected_row_ui['Model']
            else:
                with sel_col3: st.info(f"No loadable models with valid paths found for S{selected_store_ui}-D{selected_dept_ui} or globally after processing.")
        else:
            with sel_col3: st.info(f"No models (store-specific or global) with valid model paths found for S{selected_store_ui}-D{selected_dept_ui}.")
    elif selected_store_ui and not selected_dept_ui and depts_for_store_ui_list:
        with sel_col3: st.info("Select a Department to populate available models.")
    elif not selected_store_ui and unique_stores_for_ui:
         with sel_col3: st.info("Select both Store & Department to view available models.")


    with sel_col4:
        future_forecast_weeks_ui = st.number_input(
            "Future Weeks to Forecast:", min_value=0, max_value=104, value=12, step=4, key="fc_future_weeks_final",
            help="Define the number of future weeks for which to generate a new forecast, extending beyond the last known data point."
        )
    st.markdown("---")

    # --- Main Display Area: Model Details, Forecast Generation, and Visualisation ---
    if selected_store_ui and selected_dept_ui and selected_model_name_from_log_ui and model_info_selected_row_ui is not None:
        st.subheader(f"🔍 Detailed Analysis for: Store {selected_store_ui} - Department {selected_dept_ui}")
        st.markdown(f"**Using Model Configuration:** `{model_info_selected_row_ui['Model_Display_Name_Base']}`")

        # Display Model Information
        st.markdown("#### Step 2: Review Selected Model's Profile")
        st.markdown("Key information about the selected model, including its logged test performance and hyperparameters.")
        
        info_col1_ui, info_col2_ui = st.columns(2)
        with info_col1_ui:
            st.markdown("**Logged Test Set Performance**")
            st.caption("""
            These are the performance metrics (MAE, RMSE, MAPE) recorded when this model was originally evaluated on its
            designated test dataset. They provide a quantitative baseline of the model's expected accuracy.
            Lower values indicate better performance.
            """)
            metrics_to_show_final_ui = {
                k.upper(): model_info_selected_row_ui[k] for k in ['MAE', 'RMSE', 'MAPE'] # Use uppercase for display
                if k in model_info_selected_row_ui and pd.notna(model_info_selected_row_ui[k])
            }
            if metrics_to_show_final_ui:
                for metric_label, metric_value in metrics_to_show_final_ui.items():
                    st.metric(label=metric_label, value=f"{metric_value:.3f}" if isinstance(metric_value, float) else metric_value)
            else: st.info("Performance metrics were not available or not recorded for this specific model run.")

        with info_col2_ui:
            st.markdown("**Key Model Hyperparameters & Information**")
            st.caption("""
            Displays critical hyperparameters and other relevant information captured during the model's training.
            This aids in understanding the specific model configuration being visualised and is essential for reproducibility.
            """)
            params_dict_ui = model_info_selected_row_ui.get('Parameters_Dict', {})
            keys_to_exclude_final = [ # Curated list for cleaner UI
                'model_path', 'feature_scaler_path', 'target_scaler_path', 'input_shape_str', 'error',
                'train_period_in_params', 'test_period_in_params', 'features_used_list_in_params',
                'train_period_logged_in_params', 'test_period_logged_in_params', 'features_used_logged_in_params'
            ]
            # Further refine which parameters from params_dict_ui are most relevant for display
            relevant_params_for_ui = {
                k: v for k, v in params_dict_ui.items() 
                if k not in keys_to_exclude_final and not k.startswith('sarima_exog_features_') # Example of more filtering
            }
            if relevant_params_for_ui: st.json(relevant_params_for_ui, expanded=False)
            else: st.info("Key hyperparameters not found in logs or all were filtered for this concise display.")
            
            with st.expander("View Full Logged Parameters String & Parsed Features List", expanded=False):
                st.markdown("**Full Raw Parameters String (from log):**")
                st.code(model_info_selected_row_ui.get('Parameters', 'Not available'), language='text') # Display as plain text
                st.markdown("**Features Utilised by Model (parsed from log):**")
                st.json(model_info_selected_row_ui.get('Parsed_Features_List', []))


        model_path_for_loading = model_info_selected_row_ui.get('model_path')
        loaded_model_object = load_selected_model(model_path_for_loading, selected_model_name_from_log_ui)

        if loaded_model_object:
            st.success(f"Model artifact **'{selected_model_name_from_log_ui}'** successfully loaded from model store!")
            st.markdown("---")
            st.markdown("#### Step 3: Forecast Generation & Visualisation")
            st.markdown("""
            This section visualises the performance of the loaded model. It will display:
            1.  **Actual Historical Sales:** The ground truth data for the selected Store-Department.
            2.  **Actual Sales during Logged Test Period (if available):** Sales data from the original test period defined in the logs.
            3.  **(Placeholder) Model Predictions on Test Period:** How the model performed on its original test data.
            4.  **(Placeholder) New Future Forecasts:** Projections for the number of future weeks specified.
            
            **Purpose of this Visualisation:** To qualitatively assess if the model's predictions align well with actual sales patterns,
            capturing trends, seasonality, and responses to special events. This complements the quantitative metrics by providing
            a visual understanding of the forecast quality.
            """)

            historical_data_for_plot = df_featured_full[
                (df_featured_full['Store'] == selected_store_ui) &
                (df_featured_full['Dept'] == selected_dept_ui)
            ].copy()

            if 'Date' not in historical_data_for_plot.columns or 'Weekly_Sales' not in historical_data_for_plot.columns:
                st.error("Critical error for plotting: 'Date' or 'Weekly_Sales' column is missing in the filtered historical data for the selected Store-Department.")
                return

            historical_sales_series_for_plot_final = historical_data_for_plot.set_index('Date')['Weekly_Sales'].sort_index()

            # Initialise series for predictions
            test_period_predictions_series_final, future_period_predictions_series_final = None, None
            actuals_on_test_period_final, test_start_date_final, test_end_date_final = None, None, None

            # Attempt to parse test period dates from logs for plotting actuals
            params_dict_for_dates_final = model_info_selected_row_ui.get('Parameters_Dict', {})
            test_period_str_log_final = model_info_selected_row_ui.get('Test_Period',
                                                                  params_dict_for_dates_final.get('test_period_in_params',
                                                                                                     params_dict_for_dates_final.get('test_period')))
            if isinstance(test_period_str_log_final, str):
                cleaned_test_period_str_final = re.sub(r'^[^\w\s\d:-]+|[^\w\s\d:-]+$', '', test_period_str_log_final.strip())
                if ' to ' in cleaned_test_period_str_final:
                    try:
                        start_test_str_final, end_test_str_final = cleaned_test_period_str_final.split(' to ')
                        test_start_date_final = pd.to_datetime(start_test_str_final.strip(), errors='coerce')
                        test_end_date_final = pd.to_datetime(end_test_str_final.strip(), errors='coerce')
                        if pd.notna(test_start_date_final) and pd.notna(test_end_date_final):
                            actuals_on_test_period_final = historical_sales_series_for_plot_final[
                                (historical_sales_series_for_plot_final.index >= test_start_date_final) &
                                (historical_sales_series_for_plot_final.index <= test_end_date_final)
                            ]
                            if actuals_on_test_period_final.empty:
                                 st.caption(f"Note: No actual sales data found for S{selected_store_ui}-D{selected_dept_ui} within the logged test period ({test_start_date_final.date()} to {test_end_date_final.date()}).")
                        else: st.warning("Could not fully parse valid test period start/end dates from logs. Test period actuals may not be displayed accurately.")
                    except Exception as e_date_parse_final:
                        st.warning(f"Error parsing test period dates ('{test_period_str_log_final}') from logs for plotting: {e_date_parse_final}")
                        test_start_date_final, test_end_date_final = None, None # Reset on error
            else:
                st.info("Test period (e.g., 'YYYY-MM-DD to YYYY-MM-DD') not clearly defined or not found as a string in logs for this model run. Cannot accurately overlay specific test period actuals from log definition.")

            # --- Placeholder for Generating Test Period & Future Forecasts ---
            st.markdown("##### Generating Forecasts (Currently Using Illustrative Placeholder Data)")
            st.warning(
                "**Important Note:** The forecast lines displayed below (for test period and future) are currently generated "
                "using **random placeholder data** for illustrative purposes only. The actual, model-specific forecast generation "
                "logic needs to be implemented within this script. This would involve:\n"
                "1. Selecting appropriate historical data based on the model type (e.g., full series for time series models, specific feature sets for regression/LSTM).\n"
                "2. Applying any necessary preprocessing (e.g., scaling, feature engineering) consistent with how the model was trained, using the `preprocess_regression_features`, `engineer_all_demand_features`, and `create_sequences` utility functions.\n"
                "3. Using the loaded model's `.predict()` or `.forecast()` method.\n"
                "4. Post-processing predictions if necessary (e.g., inverse scaling)."
            )

            if test_start_date_final and test_end_date_final and actuals_on_test_period_final is not None and not actuals_on_test_period_final.empty:
                test_period_predictions_series_final = pd.Series(
                    np.random.uniform(actuals_on_test_period_final.min() * 0.7, actuals_on_test_period_final.max() * 1.3, size=len(actuals_on_test_period_final)),
                    index=actuals_on_test_period_final.index, name="Test Predictions (Placeholder Data)"
                )
                st.caption("Displaying **random placeholder data** for test period predictions. Replace with actual model output.")
            else:
                 st.caption("Test period actuals (from log definition) not available to overlay predictions; placeholder test predictions not generated.")

            if future_forecast_weeks_ui > 0 and not historical_sales_series_for_plot_final.empty:
                last_hist_date_final = historical_sales_series_for_plot_final.index.max()
                data_freq_final = pd.infer_freq(historical_sales_series_for_plot_final.index)
                if data_freq_final is None:
                    data_freq_final = 'W-FRI' # Default if uninferrable
                    st.caption(f"Could not infer data frequency for future date generation; defaulted to '{data_freq_final}'.")
                
                try:
                    future_pred_dates_final = pd.date_range(
                        start=last_hist_date_final + pd.tseries.frequencies.to_offset(data_freq_final),
                        periods=future_forecast_weeks_ui,
                        freq=data_freq_final
                    )
                    future_period_predictions_series_final = pd.Series(
                        np.random.uniform(historical_sales_series_for_plot_final.quantile(0.15), historical_sales_series_for_plot_final.quantile(0.85), size=len(future_pred_dates_final)),
                        index=future_pred_dates_final, name="Future Forecast (Placeholder Data)"
                    )
                    st.caption("Displaying **random placeholder data** for future forecasts. Replace with actual model output.")
                except Exception as e_fut_date:
                    st.error(f"Error generating future date range (freq: {data_freq_final}): {e_fut_date}")
            elif future_forecast_weeks_ui > 0:
                st.warning("Cannot generate future forecast placeholder as historical sales data is empty for this Store-Department selection.")

            # --- Plotting ---
            st.markdown("##### Forecast Visualisation")
            if not historical_sales_series_for_plot_final.empty:
                fig_forecast_final, ax_forecast_final = plt.subplots(figsize=(20, 9)) # Wider plot
                
                # Plot Actual Historical Sales
                ax_forecast_final.plot(historical_sales_series_for_plot_final.index, historical_sales_series_for_plot_final.values,
                        label='Actual Historical Sales', color='dodgerblue', alpha=0.85, linewidth=2)

                # Plot Actual Sales during Logged Test Period (if available)
                if actuals_on_test_period_final is not None and not actuals_on_test_period_final.empty:
                    ax_forecast_final.plot(actuals_on_test_period_final.index, actuals_on_test_period_final.values,
                            label='Actual (Logged Test Period)', color='green', linestyle='-', marker='o', markersize=4, linewidth=2.2, alpha=0.9)

                # Plot Model Predictions on Test Period (Placeholder)
                if test_period_predictions_series_final is not None and not test_period_predictions_series_final.empty:
                    ax_forecast_final.plot(test_period_predictions_series_final.index, test_period_predictions_series_final.values,
                            label=f'{model_info_selected_row_ui["Model_Display_Name_Base"]} (Test Pred. - Placeholder)',
                            color='darkorange', linestyle='--', linewidth=2.5, alpha=0.9)

                # Plot Future Forecasts (Placeholder)
                if future_period_predictions_series_final is not None and not future_period_predictions_series_final.empty:
                    ax_forecast_final.plot(future_period_predictions_series_final.index, future_period_predictions_series_final.values,
                            label=f'{model_info_selected_row_ui["Model_Display_Name_Base"]} (Future Forecast - Placeholder)',
                            color='purple', linestyle=':', linewidth=2.5, alpha=0.9)

                ax_forecast_final.set_title(f"Sales Forecast vs. Actuals for S{selected_store_ui}-D{selected_dept_ui} (Model: {model_info_selected_row_ui['Model_Display_Name_Base']})", fontsize=18)
                ax_forecast_final.set_xlabel("Date", fontsize=15)
                ax_forecast_final.set_ylabel("Weekly Sales", fontsize=15)
                ax_forecast_final.legend(fontsize=12, loc='best')
                ax_forecast_final.grid(True, linestyle='--', alpha=0.6)
                plt.xticks(fontsize=11, rotation=30, ha='right')
                plt.yticks(fontsize=11)
                plt.tight_layout()
                st.pyplot(fig_forecast_final)
                plt.close(fig_forecast_final)
            else:
                st.warning("No historical sales data is available for the selected Store-Department to generate the main forecast plot.")
        else: # Model loading failed
            st.error(f"The selected model artifact ('{selected_model_name_from_log_ui}') could not be loaded. Therefore, forecast generation and detailed visualisation are not possible for this selection.")
            st.markdown(
                "**Please review any error messages displayed above from the model loading attempt. Common troubleshooting steps include:**"
                "\n1.  **Verify Training Completion & Artifacts:** Ensure the specific training script for "
                f"'{selected_model_name_from_log_ui}' ran successfully and correctly saved the model artifact."
                "\n2.  **Check Experiment Log Path (`model_path`):** In the `reports/experiment_logs/` CSV files, locate the entry for this model, Store, and Dept. "
                "Confirm that the `model_path` value is accurate (absolute path preferred for logs, or reliably relative to `PROJECT_ROOT`), accessible, and points to an existing file."
                "\n3.  **File System Verification:** Manually navigate to the `models_store/demand_forecasting/` directory (and any subdirectories indicated in the logged path) and confirm the model file exists with the exact name and extension."
                "\n4.  **`PROJECT_ROOT` Consistency:** Ensure that the `PROJECT_ROOT` variable is determined identically by both your model training scripts (which save models and log their paths) and this Streamlit application (which attempts to load them)."
                "\n5.  **Library Dependencies & Environment:** Check that all necessary libraries for the specific model type (e.g., TensorFlow, XGBoost, Prophet, scikit-learn of the correct version) are installed and accessible in the environment where this Streamlit app is running."
            )
            st.markdown("**Logged Parameters Dictionary for this model selection (contains `model_path` as logged during training):**")
            st.json(model_info_selected_row_ui.get('Parameters_Dict', {}))

    else: # Initial state before all necessary selections (Store, Dept, Model) are made
        st.info(
            "**Getting Started:** Please use the selection boxes under 'Step 1' above to choose a **Store**, **Department**, and a **Trained Model**. "
            "Once all are selected, details about the model, its (placeholder) forecasts, and visualisations will be displayed. "
            "You can also specify the number of future weeks to forecast."
        )

if __name__ == "__main__": # pragma: no cover
    # This block facilitates standalone testing of this page module.
    # It's crucial that PROJECT_ROOT is correctly inferred or paths are adjusted
    # for data, logs, and utility function loading to work as expected.
    if os.path.basename(os.getcwd()) == 'page_content': # If running from within page_content
        os.chdir(os.path.join("..", "..")) # Navigate up to project root

    # Re-initialize project root and paths for standalone context
    PROJECT_ROOT = os.getcwd()
    SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
    if SRC_PATH not in sys.path: sys.path.insert(0, SRC_PATH)
    if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
    
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
    EXPERIMENT_LOGS_PATH = os.path.join(PROJECT_ROOT, 'reports', 'experiment_logs')
    MODEL_STORE_PATH = os.path.join(PROJECT_ROOT, 'models_store', 'demand_forecasting')

    # Attempt to re-check utility loading for standalone run context
    _UTIL_FUNCS_LOADED_FC_EXPLORER_STANDALONE = False
    try:
        from utils.preprocessing import preprocess_regression_features as actual_preprocess_reg
        from utils.feature_engineering_utils import engineer_all_demand_features as actual_engineer_all_demand
        try: from utils.preprocessing import create_sequences as actual_create_sequences
        except ImportError: from utils.feature_engineering_utils import create_sequences as actual_create_sequences
        
        preprocess_regression_features = actual_preprocess_reg
        engineer_all_demand_features = actual_engineer_all_demand
        create_sequences = actual_create_sequences
        _UTIL_FUNCS_LOADED_FC_EXPLORER_STANDALONE = True
        print("Standalone Run: Utility functions re-checked and loaded.")
    except Exception as e_standalone_utils:
         st.error(f"Standalone Run: Failed to load critical utility functions: {e_standalone_utils}. Page functionality will be impaired.")

    st.set_page_config(layout="wide", page_title="SupplyChainAI - Forecast Explorer")
    render_forecast_explorer_page()