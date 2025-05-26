"""
Renders the Inventory Strategy & Optimization page for the Streamlit application.

This page provides tools for calculating optimal inventory parameters (EOQ,
Safety Stock, Reorder Point), simulating inventory policies, and performing
sensitivity analysis on key input parameters. It can leverage demand
characteristics derived from previously logged forecasting model experiments.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import ast
import re

# --- Project Path Setup & Custom Module Imports ---
_PROJECT_ROOT_INV_OPT = None

def _get_project_root_for_inventory_page() -> str:
    """
    Determines the project root directory for the inventory optimization page.

    Assumes this script is located within `PROJECT_ROOT/app/page_content/`.
    Navigates up two directories to establish the project root, crucial for
    locating data, logs, and source modules consistently.

    Returns:
        str: The absolute path to the project root directory. Falls back to
             the current working directory if `__file__` is undefined.
    """
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    except NameError: # pragma: no cover
        st.warning(
            "Could not determine project root via `__file__` (undefined in this context). "
            "Falling back to current working directory. Module imports or data/log paths might fail "
            "if this page is run standalone without the project root as CWD."
        )
        return os.getcwd()

_PROJECT_ROOT_INV_OPT = _get_project_root_for_inventory_page()

_CORE_MODELS_LOADED = False
try:
    SRC_PATH_INV = os.path.join(_PROJECT_ROOT_INV_OPT, 'src')
    if SRC_PATH_INV not in sys.path:
        sys.path.insert(0, SRC_PATH_INV)
    if _PROJECT_ROOT_INV_OPT not in sys.path: # Ensure project root is also available
        sys.path.insert(0, _PROJECT_ROOT_INV_OPT)

    from models.inventory_optimization.core_models import (
        calculate_eoq,
        calculate_safety_stock,
        calculate_reorder_point,
        get_z_score,
        run_inventory_simulation
    )
    _CORE_MODELS_LOADED = True
    print("Inventory Optimisation Page: Core inventory models loaded successfully.")
except ImportError as e_import: # pragma: no cover
    _CORE_MODELS_LOADED = False
    st.error(
        f"CRITICAL IMPORT ERROR loading core inventory models: {e_import}. "
        "This page relies on functions from 'src/models/inventory_optimization/core_models.py'. "
        "Please ensure this file exists and that all relevant directories (like 'src', 'src/models', "
        "'src/models/inventory_optimization') have '__init__.py' files to be recognizable as packages. "
        "Page functionality will be severely impaired."
    )
    # Define dummy functions if critical imports failed to allow partial app load
    def calculate_eoq(*args, **kwargs): st.error("EOQ calculation function is unavailable due to import error."); return None
    def calculate_safety_stock(*args, **kwargs): st.error("Safety Stock calculation function is unavailable."); return None
    def calculate_reorder_point(*args, **kwargs): st.error("Reorder Point calculation function is unavailable."); return None
    def get_z_score(*args, **kwargs): st.error("Z-score utility is unavailable."); return None
    def run_inventory_simulation(*args, **kwargs): st.error("Inventory simulation function is unavailable."); return pd.DataFrame()
except Exception as e_path: # pragma: no cover
    _CORE_MODELS_LOADED = False
    st.error(f"Unexpected error during sys.path setup or core model imports for inventory page: {e_path}")


# --- Configuration Paths ---
EXPERIMENT_LOGS_PATH = os.path.join(_PROJECT_ROOT_INV_OPT, 'reports', 'experiment_logs')
PROCESSED_DATA_PATH = os.path.join(_PROJECT_ROOT_INV_OPT, 'data', 'processed')
FEATURED_DATA_FILENAME = 'walmart_data_featured.parquet' # Or your CSV filename

@st.cache_data(show_spinner="Loading experiment logs for inventory inputs...")
def _load_experiment_logs_for_inventory(log_path: str = EXPERIMENT_LOGS_PATH) -> pd.DataFrame | None:
    """
    Loads and combines experiment log CSV files, extracting data relevant
    for populating inventory calculation inputs (e.g., RMSE for demand std. dev.).

    Args:
        log_path: Path to the directory containing experiment log CSV files.

    Returns:
        A Pandas DataFrame combining relevant logs, or None if loading fails.
    """
    log_files_info = {
        'classical_ts': 'classical_timeseries_experiments.csv',
        'classical_reg': 'classical_regression_experiments.csv',
        'deep_learning': 'deep_learning_experiments.csv'
    }
    all_logs_list = []

    def _safe_literal_eval_dict_inv(val_str):
        if isinstance(val_str, dict): return val_str
        if isinstance(val_str, str):
            try: return ast.literal_eval(val_str)
            except: pass
        return {}

    for log_type, file_name in log_files_info.items():
        full_path = os.path.join(log_path, file_name)
        if os.path.exists(full_path):
            try:
                df_log = pd.read_csv(full_path)
                df_log['Log_Type_Source'] = log_type
                all_logs_list.append(df_log)
            except Exception as e:
                st.warning(f"Could not load or process log file '{file_name}': {e}")
        else:
            st.info(f"Note: Log file '{file_name}' not found at '{full_path}'. Auto-population from this source will be unavailable.")

    if not all_logs_list:
        st.warning("No experiment logs were found or loaded. Auto-population of demand characteristics from model logs will be unavailable. Please enter demand inputs manually.")
        return None

    df_combined = pd.concat(all_logs_list, ignore_index=True)
    if 'Parameters' in df_combined.columns:
        df_combined['Parameters_Dict'] = df_combined['Parameters'].apply(_safe_literal_eval_dict_inv)
    else: df_combined['Parameters_Dict'] = pd.Series([{} for _ in range(len(df_combined))])

    for col in ['Store', 'Dept']: # Ensure these columns exist for filtering
        if col in df_combined.columns:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce').fillna(-1).astype(int)
        else: df_combined[col] = -1
    if 'RMSE' in df_combined.columns: # RMSE is key for std. dev. proxy
        df_combined['RMSE'] = pd.to_numeric(df_combined['RMSE'], errors='coerce')
    else: df_combined['RMSE'] = np.nan # Add if missing for consistency
    return df_combined

@st.cache_data(show_spinner="Loading historical sales data...")
def _load_historical_sales_for_inventory(data_path: str = PROCESSED_DATA_PATH, filename: str = FEATURED_DATA_FILENAME) -> pd.DataFrame | None:
    """
    Loads historical sales data, selecting only essential columns for inventory analysis
    and simulation (Date, Store, Dept, Weekly_Sales).

    Args:
        data_path: Directory path containing the data file.
        filename: Name of the data file.

    Returns:
        Pandas DataFrame with essential sales data, or None if loading fails.
    """
    full_path = os.path.join(data_path, filename)
    if not os.path.exists(full_path):
        st.error(f"CRITICAL: Historical sales data file ('{filename}') not found at '{data_path}'. Simulation with historical demand will be disabled.")
        return None
    try:
        if full_path.lower().endswith('.parquet'):
            df = pd.read_parquet(full_path)
        elif full_path.lower().endswith('.csv'):
            df = pd.read_csv(full_path) # Date parsing handled next
        else:
            st.error(f"Unsupported historical sales data file format: '{filename}'. Only .parquet or .csv supported.")
            return None

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            if df['Date'].isnull().any():
                st.warning("Warning: Some 'Date' values in historical sales data could not be parsed to datetime.")
        else:
            st.error("CRITICAL: 'Date' column missing in historical sales data. Time-based operations will fail.")
            return None # Essential for any time-based processing

        required_cols = ['Date', 'Store', 'Dept', 'Weekly_Sales']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Historical sales data is missing the following required columns for inventory analysis: {missing_cols}.")
            return None
        return df[required_cols]
    except Exception as e:
        st.error(f"Error loading historical sales data from '{full_path}': {e}")
        return None

def render_inventory_optimization_page():
    """
    Renders the Inventory Strategy & Optimisation page with detailed explanations.

    This interactive module allows users to:
    1. Optionally leverage demand characteristics (average demand, standard deviation)
       derived from previously logged forecasting model experiments.
    2. Calculate optimal inventory parameters: Economic Order Quantity (EOQ),
       Safety Stock (SS), and Reorder Point (ROP) using classical inventory theory.
    3. Simulate the performance of these inventory policies over time under
       various demand scenarios (constant average or historical patterns).
    4. Conduct sensitivity analysis to understand how changes in key input
       parameters affect the optimal inventory levels and associated costs.

    The page aims to provide actionable insights for balancing service levels
    against inventory holding and ordering costs.
    """
    if not _CORE_MODELS_LOADED:
        st.error(
            "CRITICAL FAILURE: Core inventory calculation models (EOQ, Safety Stock, etc.) "
            "could not be loaded due to earlier import errors. This page cannot function. "
            "Please resolve the import issues related to 'src/models/inventory_optimization/core_models.py'."
        )
        st.stop() # Halt execution of this page if core logic is missing

    sns.set_theme(style="whitegrid", palette="pastel") # Consistent visual theme
    st.header("🧮 Strategic Inventory Optimisation Engine")
    st.markdown(
        """
        Welcome to the **Strategic Inventory Optimisation Engine**. Effective inventory management is pivotal for balancing
        customer service levels with operational costs. This module provides a suite of tools to:
        - Determine statistically sound inventory parameters.
        - Simulate policy performance under different demand conditions.
        - Understand the sensitivity of your inventory strategy to key business variables.
        
        The goal is to empower data-driven decisions that enhance profitability and supply chain resilience.
        """
    )

    # Load necessary data sources
    df_logs_inv = _load_experiment_logs_for_inventory() # Renamed for clarity
    df_historical_sales_inv = _load_historical_sales_for_inventory() # Renamed

    if df_logs_inv is None or df_logs_inv.empty:
        st.info(
            "**Note on Demand Inputs:** Experiment logs containing forecasting model performance (e.g., RMSE) "
            "were not available or loaded. You will need to enter all demand characteristics (Average Weekly Demand, "
            "Standard Deviation of Weekly Demand) manually in the calculator below."
        )
    if df_historical_sales_inv is None or df_historical_sales_inv.empty:
        st.info(
            "**Note on Simulation:** Historical sales data was not available or loaded. The inventory simulation feature "
            "will only support using a constant average weekly demand pattern. Simulation with actual historical demand "
            "patterns will be disabled for the selected Store-Department if applicable."
        )

    st.markdown("---")
    st.subheader("🔗 Step 1: Link to Forecast Insights (Optional)")
    st.markdown(
        """
        To derive more data-driven inventory parameters, you can optionally link to the outputs of your forecasting experiments.
        By selecting a specific `Store`, `Department`, and `Forecasting Model`, this section can:
        - Use the model's **Root Mean Squared Error (RMSE)** from its test period as a proxy for the **Standard Deviation of Weekly Demand**. RMSE reflects the typical magnitude of forecast errors.
        - Calculate the **Average Weekly Demand** based on the actual sales observed during that model's specific logged test period.
        
        If no model is selected, or if the required information isn't available in the logs, you can enter these demand characteristics manually in the calculator.
        """
    )

    sel_col1_inv, sel_col2_inv, sel_col3_inv = st.columns(3)
    selected_store_inv_ui, selected_dept_inv_ui, selected_model_info_inv_ui = None, None, None
    depts_for_store_inv_list = []

    unique_stores_inv_list = []
    if df_logs_inv is not None and 'Store' in df_logs_inv.columns:
        unique_stores_inv_list = sorted([s for s in df_logs_inv['Store'].unique() if s != -1 and s != 0])

    with sel_col1_inv:
        if not unique_stores_inv_list:
            st.info("No store-specific forecast logs available for auto-populating demand inputs.")
        else:
            selected_store_inv_ui = st.selectbox(
                "Select Store (for Demand Auto-Population):", unique_stores_inv_list, index=0, key="inv_store_select_ui",
                help="Choose the store whose forecast data you might want to use for inventory inputs."
            )
    with sel_col2_inv:
        if selected_store_inv_ui and df_logs_inv is not None and 'Dept' in df_logs_inv.columns:
            depts_for_store_inv_list = sorted([
                d for d in df_logs_inv[df_logs_inv['Store'] == selected_store_inv_ui]['Dept'].unique() if d != -1 and d != 0
            ])
            if not depts_for_store_inv_list:
                st.info(f"No department-specific forecast logs found for Store {selected_store_inv_ui}.")
            else:
                selected_dept_inv_ui = st.selectbox(
                    "Select Department (for Demand Auto-Population):", depts_for_store_inv_list, index=0, key="inv_dept_select_ui",
                    help="Choose the department within the selected store."
                )
        elif selected_store_inv_ui is None and unique_stores_inv_list: # Store selection available but not made
            st.info("Select a Store to view relevant Departments for forecast linking.")

    with sel_col3_inv:
        if selected_store_inv_ui and selected_dept_inv_ui and df_logs_inv is not None:
            relevant_models_df_inv = df_logs_inv[
                (df_logs_inv['Store'] == selected_store_inv_ui) &
                (df_logs_inv['Dept'] == selected_dept_inv_ui) &
                (df_logs_inv['RMSE'].notna()) # Critical for std. dev. proxy
            ].copy()
            if not relevant_models_df_inv.empty:
                relevant_models_df_inv['display_name_inv'] = (
                    relevant_models_df_inv['Model'] + " (RMSE: " +
                    relevant_models_df_inv['RMSE'].round(2).astype(str) + ")"
                )
                model_display_names_inv = relevant_models_df_inv['display_name_inv'].tolist()
                selected_model_display_inv_ui = st.selectbox(
                    "Select Forecasting Model (for Demand Inputs):", model_display_names_inv, index=0, key="inv_model_select_ui",
                    help="Select a model. Its RMSE will be used as a proxy for Std.Dev of Demand, and its test period actuals to estimate Avg. Demand."
                )
                if selected_model_display_inv_ui:
                    selected_model_info_inv_ui = relevant_models_df_inv[
                        relevant_models_df_inv['display_name_inv'] == selected_model_display_inv_ui
                    ].iloc[0] # Get the series for the selected model
            else:
                st.info(f"No models with logged RMSE suitable for auto-population found for S{selected_store_inv_ui}-D{selected_dept_inv_ui}.")
        elif selected_store_inv_ui and not selected_dept_inv_ui and depts_for_store_inv_list:
            st.info("Select a Department to view available models for forecast linking.")
        elif selected_store_inv_ui is None and unique_stores_inv_list:
             st.info("Select Store & Department to view models for forecast linking.")

    # Auto-populate Demand Parameters if Model Selected
    default_avg_weekly_demand_val, default_std_dev_weekly_demand_val = 100.0, 20.0 # Global defaults
    if selected_model_info_inv_ui is not None:
        logged_rmse_val = selected_model_info_inv_ui.get('RMSE')
        if pd.notna(logged_rmse_val):
            default_std_dev_weekly_demand_val = round(logged_rmse_val, 2)
            st.success(f"Std. Dev. of Demand auto-populated from selected model's RMSE: **{default_std_dev_weekly_demand_val}**")

        test_period_str_val = selected_model_info_inv_ui.get('Test_Period')
        params_dict_val = selected_model_info_inv_ui.get('Parameters_Dict', {})
        if pd.isna(test_period_str_val) and isinstance(params_dict_val, dict):
             test_period_str_val = params_dict_val.get('test_period_in_params', params_dict_val.get('test_period'))

        if isinstance(test_period_str_val, str) and df_historical_sales_inv is not None:
            cleaned_test_period_str_val = re.sub(r'^[^\w\s\d:-]+|[^\w\s\d:-]+$', '', test_period_str_val.strip())
            if ' to ' in cleaned_test_period_str_val:
                try:
                    start_str_val, end_str_val = cleaned_test_period_str_val.split(' to ')
                    start_date_val = pd.to_datetime(start_str_val.strip(), errors='coerce')
                    end_date_val = pd.to_datetime(end_str_val.strip(), errors='coerce')
                    if pd.notna(start_date_val) and pd.notna(end_date_val):
                        actuals_in_test_period_val = df_historical_sales_inv[
                            (df_historical_sales_inv['Store'] == selected_store_inv_ui) &
                            (df_historical_sales_inv['Dept'] == selected_dept_inv_ui) &
                            (df_historical_sales_inv['Date'] >= start_date_val) &
                            (df_historical_sales_inv['Date'] <= end_date_val)
                        ]['Weekly_Sales']
                        if not actuals_in_test_period_val.empty:
                            default_avg_weekly_demand_val = round(actuals_in_test_period_val.mean(), 2)
                            st.success(f"Avg. Weekly Demand auto-populated from actual sales ({len(actuals_in_test_period_val)} weeks) during selected model's test period: **{default_avg_weekly_demand_val}**")
                        else:
                            st.caption(f"No historical sales data found for S{selected_store_inv_ui}-D{selected_dept_inv_ui} during the logged test period ({test_period_str_val}). Avg. Demand not auto-populated from historicals.")
                    else: st.warning(f"Could not parse valid start/end dates from logged test period '{test_period_str_val}'.")
                except Exception as e_date_parse_val:
                    st.caption(f"Error parsing test period '{test_period_str_val}' to auto-fill Avg. Demand: {e_date_parse_val}.")
            elif selected_model_info_inv_ui is not None: # If test_period_str_val was not 'X to Y'
                 st.caption(f"Test period string format ('{test_period_str_val}') from logs is not recognized for date parsing. Avg. Demand not auto-populated.")
        elif selected_model_info_inv_ui is not None and df_historical_sales_inv is None: # Historical data is missing
            st.caption("Historical sales data is not loaded; Avg. Weekly Demand cannot be auto-populated from model's test period actuals.")
        elif selected_model_info_inv_ui is not None: # Test period string was NaN or not a string
             st.caption("Test period not defined or not a parseable string for the selected model in logs; Avg. Demand cannot be auto-populated.")

    st.markdown("---")
    st.subheader("🧮 Step 2: Interactive Inventory Parameter Calculator")
    st.markdown("""
    Define the key characteristics of your product's demand, associated costs, supplier lead times, and desired customer service level.
    Based on these inputs, the calculator will compute standard inventory control parameters.
    """)
    st.markdown("#### Input Parameters for Calculation")
    # Use unique session state keys for this page to avoid conflicts
    if 'inv_opt_baseline_params' not in st.session_state:
        st.session_state.inv_opt_baseline_params = {}

    input_col1_calc, input_col2_calc = st.columns(2)
    with input_col1_calc:
        st.markdown("**Demand Characteristics (per week)**")
        avg_weekly_demand_calc = st.number_input(
            "Average Weekly Demand (Units):", min_value=0.0,
            value=st.session_state.inv_opt_baseline_params.get('avg_weekly_demand', default_avg_weekly_demand_val),
            step=10.0, format="%.2f", key="inv_avg_demand_calc_input",
            help="The expected average number of units sold or consumed per week."
        )
        std_dev_weekly_demand_calc = st.number_input(
            "Standard Deviation of Weekly Demand (Units):", min_value=0.0,
            value=st.session_state.inv_opt_baseline_params.get('std_dev_weekly_demand', default_std_dev_weekly_demand_val),
            step=1.0, format="%.2f", key="inv_std_demand_calc_input",
            help="Measures the variability or uncertainty in weekly demand. Higher values indicate more fluctuation. Can be proxied by forecast RMSE."
        )
        st.markdown("**Cost Parameters**")
        item_cost_per_unit_calc = st.number_input(
            "Cost per Unit ($):", min_value=0.01,
            value=st.session_state.inv_opt_baseline_params.get('item_cost_per_unit', 10.0),
            step=0.5, format="%.2f", key="inv_item_cost_calc_input",
            help="The direct cost to acquire or produce one unit of the item."
        )
        ordering_cost_per_order_calc = st.number_input(
            "Cost per Order ($) (Fixed Administrative Cost):", min_value=0.0,
            value=st.session_state.inv_opt_baseline_params.get('ordering_cost_per_order', 50.0),
            step=5.0, format="%.2f", key="inv_order_cost_calc_input",
            help="The fixed administrative and logistical cost incurred each time an order is placed (e.g., shipping, processing)."
        )
    with input_col2_calc:
        annual_holding_rate_percent_calc = st.slider(
            "Annual Holding Cost Rate (% of Item Cost):", min_value=0, max_value=50,
            value=st.session_state.inv_opt_baseline_params.get('annual_holding_rate_percent', 20),
            step=1, key="inv_holding_rate_calc_input",
            help="The cost to hold one unit of inventory for a year, expressed as a percentage of the item's cost (includes storage, insurance, obsolescence, capital cost)."
        )
        st.markdown("**Lead Time Characteristics (in weeks)**")
        avg_lead_time_weeks_calc = st.number_input(
            "Average Supplier Lead Time (Weeks):", min_value=0.0,
            value=st.session_state.inv_opt_baseline_params.get('avg_lead_time_weeks', 4.0),
            step=0.5, format="%.1f", key="inv_avg_lt_calc_input",
            help="The average time it takes from placing an order with a supplier to receiving the goods."
        )
        std_dev_lead_time_weeks_calc = st.number_input(
            "Standard Deviation of Lead Time (Weeks):", min_value=0.0,
            value=st.session_state.inv_opt_baseline_params.get('std_dev_lead_time_weeks', 1.0),
            step=0.1, format="%.2f", key="inv_std_lt_calc_input",
            help="Measures the variability in supplier lead time. Higher values indicate less reliable delivery times."
        )
        st.markdown("**Target Service Level**")
        target_service_level_percent_calc = st.slider(
            "Target Service Level (Cycle Service Level %):", min_value=50, max_value=99, # Common range, can be 99.9
            value=st.session_state.inv_opt_baseline_params.get('target_service_level_percent', 95),
            step=1, key="inv_service_level_calc_input",
            help="The desired probability of not stocking out during a replenishment cycle. Higher levels require more safety stock."
        )

    # Update session state with current inputs for calculator
    st.session_state.inv_opt_baseline_params = {
        'avg_weekly_demand': avg_weekly_demand_calc, 'std_dev_weekly_demand': std_dev_weekly_demand_calc,
        'item_cost_per_unit': item_cost_per_unit_calc, 'ordering_cost_per_order': ordering_cost_per_order_calc,
        'annual_holding_rate_percent': annual_holding_rate_percent_calc,
        'avg_lead_time_weeks': avg_lead_time_weeks_calc, 'std_dev_lead_time_weeks': std_dev_lead_time_weeks_calc,
        'target_service_level_percent': target_service_level_percent_calc
    }
    if 'inv_opt_calc_results' not in st.session_state:
        st.session_state.inv_opt_calc_results = {}

    if st.button("Calculate Optimal Inventory Parameters", key="calc_inventory_btn_detailed", type="primary"):
        st.session_state.inv_opt_calc_results = {} # Reset
        params_calc = st.session_state.inv_opt_baseline_params
        annual_demand_calc_val = params_calc['avg_weekly_demand'] * 52
        annual_holding_cost_pu_val = params_calc['item_cost_per_unit'] * (params_calc['annual_holding_rate_percent'] / 100.0)

        eoq_calc_result = None
        if annual_demand_calc_val > 0 and params_calc['ordering_cost_per_order'] >= 0 and annual_holding_cost_pu_val > 0:
            eoq_calc_result = calculate_eoq(annual_demand_calc_val, params_calc['ordering_cost_per_order'], annual_holding_cost_pu_val)
        else:
            st.warning(
                "EOQ cannot be calculated. Ensure: Avg. Weekly Demand > 0, "
                "Ordering Cost >= 0, and (Item Cost > 0 AND Holding Cost Rate > 0)."
            )
        ss_calc_result = calculate_safety_stock(
            params_calc['avg_weekly_demand'], params_calc['std_dev_weekly_demand'],
            params_calc['avg_lead_time_weeks'], params_calc['std_dev_lead_time_weeks'],
            float(params_calc['target_service_level_percent']) # Ensure float for get_z_score
        )
        rop_calc_result = None
        if ss_calc_result is not None:
            rop_calc_result = calculate_reorder_point(
                params_calc['avg_weekly_demand'], params_calc['avg_lead_time_weeks'], ss_calc_result
            )
        st.session_state.inv_opt_calc_results = {
            'eoq': eoq_calc_result, 'safety_stock': ss_calc_result, 'rop': rop_calc_result,
            'inputs_used_for_calc': params_calc.copy()
        }

    if st.session_state.inv_opt_calc_results and any(val is not None for val in st.session_state.inv_opt_calc_results.values()):
        results_calc = st.session_state.inv_opt_calc_results
        inputs_used_disp = results_calc.get('inputs_used_for_calc', {})
        st.markdown("---"); st.subheader("📊 Calculated Inventory Parameters & Financial Implications")
        
        st.markdown("##### Core Inventory Control Parameters:")
        res_col1_disp, res_col2_disp, res_col3_disp = st.columns(3)
        with res_col1_disp:
            eoq_display_val = f"{results_calc.get('eoq', 0):.0f} units" if results_calc.get('eoq') is not None else "N/A (Check Inputs)"
            st.metric(label="Economic Order Quantity (EOQ)", value=eoq_display_val)
            st.caption("EOQ aims to minimise total ordering and holding costs.")
        with res_col2_disp:
            ss_display_val = f"{results_calc.get('safety_stock', 0):.0f} units" if results_calc.get('safety_stock') is not None else "N/A"
            st.metric(label="Safety Stock (SS)", value=ss_display_val)
            if results_calc.get('safety_stock') is not None:
                z_val_disp = get_z_score(float(inputs_used_disp.get('target_service_level_percent', 0)))
                z_disp_str = f" (Z ≈ {z_val_disp:.2f})" if z_val_disp is not None else ""
                st.caption(f"For {inputs_used_disp.get('target_service_level_percent',0)}% Service Level{z_disp_str}. Buffers against demand/lead time uncertainty.")
        with res_col3_disp:
            rop_display_val = f"{results_calc.get('rop', 0):.0f} units" if results_calc.get('rop') is not None else "N/A"
            st.metric(label="Reorder Point (ROP)", value=rop_display_val)
            if results_calc.get('rop') is not None:
                avg_demand_during_lt_disp = inputs_used_disp.get('avg_weekly_demand',0) * inputs_used_disp.get('avg_lead_time_weeks',0)
                st.caption(f"Trigger to order. (Avg. Demand during LT: {avg_demand_during_lt_disp:.0f} + SS)")

        # Illustrative Cost Breakdown
        if all(results_calc.get(k) is not None for k in ['eoq', 'safety_stock']) and \
           all(inputs_used_disp.get(k) is not None for k in ['item_cost_per_unit', 'ordering_cost_per_order', 'annual_holding_rate_percent', 'avg_weekly_demand']):
            
            annual_demand_disp = inputs_used_disp['avg_weekly_demand'] * 52
            item_cost_disp = inputs_used_disp['item_cost_per_unit']
            ordering_cost_disp = inputs_used_disp['ordering_cost_per_order']
            holding_rate_disp_dec = inputs_used_disp['annual_holding_rate_percent'] / 100.0
            annual_holding_cost_pu_disp = item_cost_disp * holding_rate_disp_dec
            eoq_val_disp = results_calc['eoq']
            ss_val_disp_cost = results_calc['safety_stock']

            if eoq_val_disp > 0 and annual_holding_cost_pu_disp >= 0: # Allow zero holding cost for some scenarios
                num_orders_disp = annual_demand_disp / eoq_val_disp if eoq_val_disp > 0 else 0
                annual_ordering_cost_disp = num_orders_disp * ordering_cost_disp
                avg_cycle_stock_disp = eoq_val_disp / 2
                annual_holding_cycle_disp = avg_cycle_stock_disp * annual_holding_cost_pu_disp
                annual_holding_safety_disp = ss_val_disp_cost * annual_holding_cost_pu_disp
                total_holding_disp = annual_holding_cycle_disp + annual_holding_safety_disp
                total_relevant_cost_disp = annual_ordering_cost_disp + total_holding_disp

                st.markdown("##### Illustrative Annual Cost Implications:")
                st.markdown("This breakdown helps visualise the trade-offs inherent in inventory policy. The EOQ seeks an optimal balance between ordering frequency (driving ordering costs) and average inventory levels (driving holding costs). Safety stock adds to holding costs but protects service levels.")
                cost_data_disp = {
                    "Cost Component": [
                        "Annual Ordering Costs (Fixed)", "Annual Holding Costs (from Cycle Stock)",
                        "Annual Holding Costs (from Safety Stock)", "**Total Annual Holding Costs**",
                        "**Total Relevant Inventory Costs (Ordering + Holding)**"
                    ],
                    "Estimated Value ($)": [
                        f"{annual_ordering_cost_disp:,.2f}", f"{annual_holding_cycle_disp:,.2f}",
                        f"{annual_holding_safety_disp:,.2f}", f"**{total_holding_disp:,.2f}**",
                        f"**{total_relevant_cost_disp:,.2f}**"
                    ]
                }
                st.table(pd.DataFrame(cost_data_disp))
        st.markdown("---")


    # --- Inventory Simulation Section ---
    st.subheader("🧪 Step 3: Inventory Policy Simulation")
    st.markdown(
        """
        Inventory simulation provides a dynamic view of how your chosen inventory policy (defined by parameters like EOQ and ROP,
        which can be taken from the calculator above or adjusted) would perform under specific demand conditions over a defined period.
        **Value:** It allows you to stress-test your strategy, visually identify potential stockout risks or periods of excess inventory,
        and understand the practical implications of your parameters before real-world implementation. This is crucial for validating
        theoretical calculations against more realistic, fluctuating demand.
        """
    )
    sim_inputs_col1_ui, sim_inputs_col2_ui = st.columns(2)
    with sim_inputs_col1_ui:
        simulation_weeks_ui = st.number_input(
            "Simulation Duration (Weeks):", min_value=4, max_value=260, value=52, step=4, key="sim_weeks_ui",
            help="Length of the period over which to simulate inventory levels."
        )
        # Default initial inventory to ROP or a fallback
        default_sim_initial_inv_ui = int(st.session_state.get('inv_opt_calc_results', {}).get('rop', 150)
                                     if st.session_state.get('inv_opt_calc_results', {}).get('rop') is not None
                                     else 150)
        initial_inventory_ui = st.number_input(
            "Initial Inventory Level at Start of Simulation (Units):", min_value=0,
            value=default_sim_initial_inv_ui, step=10, key="sim_initial_inv_ui",
            help="The on-hand inventory quantity at the beginning of the simulation."
        )
    with sim_inputs_col2_ui:
        demand_pattern_options_ui = ["Use Constant Average Weekly Demand (from calculator inputs)"]
        hist_demand_label_ui = "Use Actual Historical Demand (select Store/Dept in Step 1)"
        # Enable historical demand only if data and selections are valid
        can_use_historical_demand = (
            df_historical_sales_inv is not None and
            selected_store_inv_ui is not None and
            selected_dept_inv_ui is not None
        )
        if can_use_historical_demand:
            demand_pattern_options_ui.append(hist_demand_label_ui)
        else:
            st.caption("To simulate with historical demand, first select a Store and Department in 'Step 1' above and ensure historical sales data is loaded.")

        selected_demand_pattern_ui = st.selectbox(
            "Demand Pattern for Simulation:", options=demand_pattern_options_ui, key="sim_demand_pattern_ui",
            help="Choose whether to simulate with a constant average demand or actual historical demand patterns (if available for selected Store/Dept)."
        )

    if st.button("Run Inventory Simulation", key="run_sim_btn_detailed", type="primary"):
        # Fetch latest calculated parameters for simulation
        calc_results_sim = st.session_state.get('inv_opt_calc_results', {})
        baseline_params_sim = st.session_state.get('inv_opt_baseline_params', {}) # From calculator inputs

        # Ensure essential parameters for simulation are calculated/available
        eoq_for_sim_run = calc_results_sim.get('eoq')
        rop_for_sim_run = calc_results_sim.get('rop')
        avg_lt_for_sim = baseline_params_sim.get('avg_lead_time_weeks')

        if eoq_for_sim_run is None or rop_for_sim_run is None or avg_lt_for_sim is None:
            st.warning("Essential inventory parameters (EOQ, ROP from calculator; Average Lead Time from inputs) are missing. "
                       "Please calculate/input them first before running the simulation.")
        else:
            eoq_for_sim_run = float(eoq_for_sim_run)
            rop_for_sim_run = float(rop_for_sim_run)
            lead_time_for_sim_run = int(round(avg_lt_for_sim)) # Simulation typically uses integer periods

            demand_series_for_sim_run = []
            if selected_demand_pattern_ui.startswith("Use Constant Average"):
                avg_demand_sim_input = baseline_params_sim.get('avg_weekly_demand', 100) # Fallback if not in baseline
                demand_series_for_sim_run = [avg_demand_sim_input] * simulation_weeks_ui
                st.caption(f"Simulation using constant average weekly demand of {avg_demand_sim_input:.2f} units.")
            elif selected_demand_pattern_ui.startswith("Use Actual Historical Demand") and df_historical_sales_inv is not None and can_use_historical_demand:
                # Ensure selected_store_inv_ui and selected_dept_inv_ui are used here
                hist_sales_query_df = df_historical_sales_inv[
                    (df_historical_sales_inv['Store'] == selected_store_inv_ui) &
                    (df_historical_sales_inv['Dept'] == selected_dept_inv_ui)
                ].copy() # Use specific selections from Step 1
                
                if not hist_sales_query_df.empty:
                    hist_sales_query_df.sort_values('Date', inplace=True) # Ensure chronological order
                    hist_sales_values = hist_sales_query_df['Weekly_Sales'].values
                    
                    if len(hist_sales_values) >= simulation_weeks_ui:
                        demand_series_for_sim_run = hist_sales_values[:simulation_weeks_ui]
                    elif len(hist_sales_values) > 0: # Tile if historical data is shorter
                        demand_series_for_sim_run = np.tile(
                            hist_sales_values, (simulation_weeks_ui // len(hist_sales_values)) + 1
                        )[:simulation_weeks_ui]
                        st.caption(f"Historical data ({len(hist_sales_values)} weeks) was tiled to meet simulation duration of {simulation_weeks_ui} weeks.")
                    else: # Should not happen if can_use_historical_demand was true and selections made
                        st.warning(f"No historical sales data actually found for S{selected_store_inv_ui}-D{selected_dept_inv_ui} for simulation. Reverting to constant average demand.")
                        demand_series_for_sim_run = [baseline_params_sim.get('avg_weekly_demand', 100)] * simulation_weeks_ui
                else:
                    st.warning(f"No historical sales data found for S{selected_store_inv_ui}-D{selected_dept_inv_ui} for simulation. Reverting to constant average demand.")
                    demand_series_for_sim_run = [baseline_params_sim.get('avg_weekly_demand', 100)] * simulation_weeks_ui
            
            # Final check on demand_series_for_sim_run
            if not (isinstance(demand_series_for_sim_run, (list, np.ndarray)) and len(demand_series_for_sim_run) == simulation_weeks_ui and simulation_weeks_ui > 0):
                st.error(
                    f"Could not prepare a valid demand series for simulation. "
                    f"Expected {simulation_weeks_ui} weeks of demand data. Please check inputs and historical data availability."
                )
            else:
                with st.spinner("Running inventory simulation... This can take a moment for longer durations."):
                    sim_results_df_output = run_inventory_simulation(
                        demand_series=np.array(demand_series_for_sim_run, dtype=float),
                        initial_inventory=float(initial_inventory_ui),
                        order_quantity_eoq=float(eoq_for_sim_run),
                        reorder_point_rop=float(rop_for_sim_run),
                        lead_time_periods=lead_time_for_sim_run,
                        simulation_periods=simulation_weeks_ui
                    )

                if sim_results_df_output is not None and not sim_results_df_output.empty:
                    st.success("Inventory simulation completed successfully!")
                    st.markdown("#### Simulation Performance Summary Metrics:")
                    st.markdown("These metrics summarize the overall performance of the inventory policy during the simulated period.")
                    
                    summary_col1_sim, summary_col2_sim, summary_col3_sim = st.columns(3)
                    total_demand_sim_val = sim_results_df_output['Demand'].sum()
                    total_sold_sim_val = sim_results_df_output['Units_Sold'].sum()
                    total_stockouts_sim_val = sim_results_df_output['Stockout_Units'].sum()
                    num_orders_sim_val = len(sim_results_df_output[sim_results_df_output['Order_Placed_Qty'] > 0])

                    summary_col1_sim.metric("Total Demand During Simulation", f"{total_demand_sim_val:,.0f} units")
                    summary_col1_sim.metric("Total Units Fulfilled (Sold)", f"{total_sold_sim_val:,.0f} units")
                    summary_col2_sim.metric("Total Unfulfilled Demand (Stockout Units)", f"{total_stockouts_sim_val:,.0f} units",
                                        delta_color="inverse" if total_stockouts_sim_val > 0 else "normal",
                                        help="Lower is better. Indicates demand that could not be met from on-hand stock.")
                    if total_demand_sim_val > 0:
                        fill_rate_val = (total_sold_sim_val / total_demand_sim_val) * 100
                        summary_col2_sim.metric("Achieved Service Level (In-Stock Fill Rate)", f"{fill_rate_val:.2f}%",
                                                help="Percentage of demand fulfilled directly from stock. Higher is better.")
                    summary_col3_sim.metric("Number of Replenishment Orders Placed", f"{num_orders_sim_val}",
                                            help="Indicates ordering frequency. Relates to total ordering costs.")
                    
                    avg_inv_level_sim = sim_results_df_output['Inventory_End_Period'].mean()
                    summary_col3_sim.metric("Average End-of-Period Inventory", f"{avg_inv_level_sim:,.0f} units",
                                            help="Average inventory held. Relates to total holding costs.")


                    st.markdown("#### Visualisation: Simulated Inventory Levels Over Time")
                    st.markdown("""
                    This plot illustrates the week-by-week fluctuation of on-hand inventory. 
                    **Interpretation:**
                    - **Inventory Level (Blue Line):** Shows the on-hand stock at the end of each week.
                    - **Reorder Point (ROP - Orange Dashed Line):** When inventory drops to or below this level, a replenishment order (of EOQ size) is triggered.
                    - **Safety Stock (SS - Green Dotted Line, if calculated):** The buffer stock maintained to mitigate stockouts due to demand or lead time variability. Ideally, inventory should rarely drop below this level.
                    - **Stockout Markers (Red 'X'):** Indicate weeks where demand exceeded available inventory, resulting in a stockout.
                    Observe how inventory cycles with replenishments and how effectively the ROP and SS protect against stockouts given the demand pattern.
                    """)
                    plt.style.use('seaborn-v0_8-darkgrid') # Consistent style
                    fig_sim_plot, ax_sim_plot = plt.subplots(figsize=(18, 8)) # Enhanced size
                    ax_sim_plot.plot(sim_results_df_output['Period'], sim_results_df_output['Inventory_End_Period'],
                                label='On-Hand Inventory Level', color='dodgerblue', marker='.', markersize=4,
                                linestyle='-', linewidth=1.8, zorder=5)
                    ax_sim_plot.axhline(y=rop_for_sim_run, color='darkorange', linestyle='--', linewidth=2.2,
                                   label=f'Reorder Point (ROP ≈ {rop_for_sim_run:.0f})', zorder=3)
                    
                    ss_val_for_plot = calc_results_sim.get('safety_stock') # Get from original calculation if available
                    if ss_val_for_plot is not None:
                         ax_sim_plot.axhline(y=ss_val_for_plot, color='forestgreen', linestyle=':', linewidth=2.2,
                                        label=f'Safety Stock (SS ≈ {ss_val_for_plot:.0f})', zorder=3)

                    stockout_periods_df_plot = sim_results_df_output[sim_results_df_output['Stockout_Units'] > 0]
                    if not stockout_periods_df_plot.empty:
                        ax_sim_plot.scatter(stockout_periods_df_plot['Period'],
                                       sim_results_df_output.loc[stockout_periods_df_plot.index, 'Inventory_End_Period'],
                                       color='red', marker='X', s=100, label='Stockout Event', zorder=10, edgecolors='black')

                    ax_sim_plot.set_xlabel("Simulation Week Number", fontsize=14)
                    ax_sim_plot.set_ylabel("Units in Inventory", fontsize=14)
                    ax_sim_plot.set_title("Simulated Inventory Level Dynamics Over Time", fontsize=17, fontweight='bold')
                    ax_sim_plot.legend(fontsize=11, loc='best')
                    ax_sim_plot.grid(True, linestyle='--', alpha=0.7)
                    ax_sim_plot.tick_params(axis='both', which='major', labelsize=11)
                    plt.fill_between(sim_results_df_output['Period'], 0, sim_results_df_output['Inventory_End_Period'],
                                     alpha=0.1, color='dodgerblue') # Subtle fill
                    plt.tight_layout()
                    st.pyplot(fig_sim_plot)
                    plt.close(fig_sim_plot)

                    if st.checkbox("Show Detailed Week-by-Week Simulation Results Table", False, key="show_sim_table_detailed"):
                        st.markdown("The table below shows the inventory flow for each week of the simulation:")
                        st.dataframe(sim_results_df_output.style.format("{:.0f}", subset=pd.IndexSlice[:, ['Demand', 'Inventory_Start_Period', 'Inventory_End_Period', 'Units_Sold', 'Order_Received_Qty', 'Order_Placed_Qty', 'Stockout_Units']]))
                else:
                    st.error("Inventory simulation did not produce any valid results. Please check your input parameters, demand series preparation, and console for potential errors from the `run_inventory_simulation` function.")

    st.markdown("---")
    # --- Sensitivity Analysis Section ---
    st.subheader("🔬 Step 4: Sensitivity Analysis of Inventory Parameters")
    st.markdown(
        """
        Sensitivity analysis is a powerful technique to understand how robust your calculated inventory parameters 
        (EOQ, Safety Stock, Reorder Point) are to variations in key input assumptions. 
        By varying one input parameter at a time while holding others constant (at their current values from the calculator above), 
        we can observe the impact on the optimal inventory levels.

        **Value:** This analysis helps identify which inputs exert the most influence on your inventory policy. For instance, if ROP is highly
        sensitive to changes in lead time variability, then efforts to stabilize lead times become more critical. It aids in risk assessment
        and focusing data improvement efforts where they matter most.
        """
    )
    baseline_params_for_sens = st.session_state.get('inv_opt_baseline_params', {}).copy() # Use current calculator inputs
    if not baseline_params_for_sens or not all(k in baseline_params_for_sens for k in ['avg_weekly_demand', 'item_cost_per_unit', 'annual_holding_rate_percent']):
        st.info("Please ensure all baseline parameters are entered in the 'Interactive Inventory Calculator' (Step 2) above to enable sensitivity analysis.")
    else:
        param_to_vary_options_sens = {
            "Annual Holding Cost Rate (% of Item Cost)": "annual_holding_rate_percent",
            "Ordering Cost per Order ($)": "ordering_cost_per_order",
            "Target Service Level (%)": "target_service_level_percent",
            "Average Weekly Demand (Units)": "avg_weekly_demand",
            "Std. Dev. of Weekly Demand (Units)": "std_dev_weekly_demand",
            "Average Supplier Lead Time (Weeks)": "avg_lead_time_weeks",
            "Std. Dev. of Lead Time (Weeks)": "std_dev_lead_time_weeks"
        }
        selected_param_display_name_sens = st.selectbox(
            "Select Input Parameter to Vary for Sensitivity Analysis:",
            options=list(param_to_vary_options_sens.keys()), key="sens_param_select_ui"
        )
        selected_param_key_sens = param_to_vary_options_sens[selected_param_display_name_sens]

        default_val_sens = baseline_params_for_sens.get(selected_param_key_sens, 0)
        # Define more robust default ranges, preventing zero or negative where inappropriate
        default_range_sens_config = {
            "annual_holding_rate_percent": (max(1.0, default_val_sens * 0.5), min(100.0, default_val_sens * 1.5), 5.0, "%.1f"),
            "ordering_cost_per_order": (max(0.0, default_val_sens * 0.5), default_val_sens * 1.5 + 1, 10.0, "%.2f"), # Add 1 to max to avoid min=max if default is 0
            "target_service_level_percent": (max(50.0, default_val_sens - 20), min(99.9, default_val_sens + 20), 2.0, "%.1f"),
            "avg_weekly_demand": (max(1.0, default_val_sens * 0.5), default_val_sens * 1.5 + 1, max(1.0, float(default_val_sens*0.1)), "%.2f"),
            "std_dev_weekly_demand": (max(0.0, default_val_sens * 0.5), default_val_sens * 1.5 + 1, max(0.5, float(default_val_sens*0.1)), "%.2f"),
            "avg_lead_time_weeks": (max(0.1, default_val_sens * 0.5), default_val_sens * 1.5 + 0.1, 0.5, "%.1f"),
            "std_dev_lead_time_weeks": (max(0.0, default_val_sens * 0.5), default_val_sens * 1.5 + 0.1, 0.2, "%.2f")
        }
        current_range_sens = default_range_sens_config.get(selected_param_key_sens, (0, 100, 10, "%.2f")) # Fallback

        sens_range_col1_ui, sens_range_col2_ui, sens_range_col3_ui = st.columns(3)
        with sens_range_col1_ui:
            vary_min_ui = st.number_input("Minimum Value for Variation:", value=float(current_range_sens[0]), format=current_range_sens[3], key="sens_min_ui", help=f"Current baseline value for {selected_param_display_name_sens} is {default_val_sens:.2f}")
        with sens_range_col2_ui:
            vary_max_ui = st.number_input("Maximum Value for Variation:", value=float(current_range_sens[1]), format=current_range_sens[3], key="sens_max_ui")
        with sens_range_col3_ui:
            num_steps_ui = st.number_input("Number of Steps for Variation:", min_value=3, max_value=50, value=11, step=1, key="sens_steps_ui", help="Number of distinct values of the parameter to test within the range.")

        if st.button("Run Sensitivity Analysis", key="run_sens_analysis_btn_detailed", type="primary"):
            if vary_min_ui >= vary_max_ui:
                st.warning("Minimum value for variation must be strictly less than the Maximum value.")
            else:
                param_values_to_test_sens = np.linspace(vary_min_ui, vary_max_ui, int(num_steps_ui))
                sensitivity_results_list_ui = []
                with st.spinner(f"Running sensitivity analysis for '{selected_param_display_name_sens}' across {len(param_values_to_test_sens)} values..."):
                    for val_iter in param_values_to_test_sens:
                        current_params_iter_sens = baseline_params_for_sens.copy()
                        current_params_iter_sens[selected_param_key_sens] = val_iter

                        annual_demand_iter_sens = current_params_iter_sens['avg_weekly_demand'] * 52
                        annual_holding_cost_pu_iter_sens = current_params_iter_sens['item_cost_per_unit'] * \
                                                      (current_params_iter_sens['annual_holding_rate_percent'] / 100.0)
                        
                        eoq_sens_iter_val = None
                        if annual_demand_iter_sens > 0 and current_params_iter_sens['ordering_cost_per_order'] >= 0 and annual_holding_cost_pu_iter_sens > 0:
                            eoq_sens_iter_val = calculate_eoq(
                                annual_demand_iter_sens, current_params_iter_sens['ordering_cost_per_order'], annual_holding_cost_pu_iter_sens
                            )
                        
                        ss_sens_iter_val = calculate_safety_stock(
                            current_params_iter_sens['avg_weekly_demand'], current_params_iter_sens['std_dev_weekly_demand'],
                            current_params_iter_sens['avg_lead_time_weeks'], current_params_iter_sens['std_dev_lead_time_weeks'],
                            float(current_params_iter_sens['target_service_level_percent'])
                        )
                        
                        rop_sens_iter_val = None
                        if ss_sens_iter_val is not None: # ROP calculation depends on SS
                            rop_sens_iter_val = calculate_reorder_point(
                                current_params_iter_sens['avg_weekly_demand'], current_params_iter_sens['avg_lead_time_weeks'], ss_sens_iter_val
                            )
                        
                        sensitivity_results_list_ui.append({
                            selected_param_display_name_sens: val_iter, # The varied parameter value
                            'EOQ': eoq_sens_iter_val,
                            'Safety_Stock': ss_sens_iter_val,
                            'ROP': rop_sens_iter_val
                        })
                results_df_sens_ui = pd.DataFrame(sensitivity_results_list_ui)

                st.markdown("#### Sensitivity Analysis Results (Tabular)")
                st.markdown(f"Shows how EOQ, Safety Stock, and ROP change as `{selected_param_display_name_sens}` varies, while other inputs are held at their baseline values from the calculator.")
                st.dataframe(results_df_sens_ui.style.format("{:.1f}", na_rep="N/A", subset=pd.IndexSlice[:, ['EOQ', 'Safety_Stock', 'ROP']]).format({selected_param_display_name_sens: current_range_sens[3]}))

                st.markdown("#### Sensitivity Visualisations")
                st.markdown("""
                These plots graphically illustrate the sensitivity of each key inventory parameter (EOQ, Safety Stock, ROP)
                to changes in the selected input variable.
                **Interpretation:**
                - The **x-axis** shows the range of the input parameter being varied.
                - The **y-axis** shows the resulting value of the inventory parameter.
                - A **steep slope** in a line indicates high sensitivity, meaning small changes in the input parameter lead to large changes in the inventory parameter. This highlights critical inputs that require careful estimation and monitoring.
                - A **flat slope** indicates low sensitivity.
                """)
                fig_sens_plot, axes_sens_plot = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
                plot_vars_sens = ['EOQ', 'Safety_Stock', 'ROP']
                colors_sens = ['crimson', 'dodgerblue', 'forestgreen']
                titles_sens = ["Sensitivity of Economic Order Quantity (EOQ)",
                               "Sensitivity of Safety Stock (SS)",
                               "Sensitivity of Reorder Point (ROP)"]

                for i, var_to_plot_sens in enumerate(plot_vars_sens):
                    if var_to_plot_sens in results_df_sens_ui.columns:
                        sns.lineplot(data=results_df_sens_ui, x=selected_param_display_name_sens, y=var_to_plot_sens,
                                     ax=axes_sens_plot[i], marker='o', markersize=5, linewidth=2, color=colors_sens[i])
                        axes_sens_plot[i].set_ylabel(f"{var_to_plot_sens.replace('_',' ')} (Units)", fontsize=12)
                        axes_sens_plot[i].set_title(titles_sens[i], fontsize=15)
                        axes_sens_plot[i].grid(True, linestyle=':', alpha=0.7)
                        axes_sens_plot[i].tick_params(axis='both', which='major', labelsize=10)
                    else:
                        axes_sens_plot[i].text(0.5, 0.5, f"{var_to_plot_sens} data not available for plotting.",
                                          horizontalalignment='center', verticalalignment='center',
                                          transform=axes_sens_plot[i].transAxes, fontsize=12, color='gray')

                axes_sens_plot[-1].set_xlabel(f"Varied Input Parameter: {selected_param_display_name_sens}", fontsize=13)
                fig_sens_plot.suptitle(f"Impact of Varying '{selected_param_display_name_sens}' on Inventory Parameters", fontsize=18, fontweight='bold', y=1.03)
                plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout for suptitle
                st.pyplot(fig_sens_plot)
                plt.close(fig_sens_plot)

    st.markdown("---")
    st.caption(
        "**Important Note:** For accurate safety stock and reorder point calculations, ensure that all demand-related inputs "
        "(Average Weekly Demand, Std. Dev. of Weekly Demand) and lead time inputs (Average Lead Time, Std. Dev. of Lead Time) "
        "are expressed in the **same time unit** (e.g., all weekly)."
    )

if __name__ == "__main__": # pragma: no cover
    # This block facilitates standalone testing of this page.
    # It's essential that PROJECT_ROOT is correctly inferred or paths are adjusted
    # for data files and core_models.py to be imported/loaded correctly.
    # st.set_page_config(layout="wide", page_title="SupplyChainAI - Inventory Optimisation") # Typically in main_app.py

    if not _CORE_MODELS_LOADED:
        st.error(
            "Standalone Run: Core inventory models failed to load. "
            "This page's functionality will be severely limited or non-functional."
        )
    render_inventory_optimization_page()