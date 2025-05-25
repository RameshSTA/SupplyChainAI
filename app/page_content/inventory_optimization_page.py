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
_PROJECT_ROOT_INV_OPT = None # Module-level variable for project root

def _get_project_root_for_inventory_page() -> str:
    """
    Determines the project root directory for this inventory optimization page.

    Assumes this script (`inventory_optimization_page.py`) is located within:
    `PROJECT_ROOT/app/page_content/`.
    Navigates up two directories from this file's location.

    Returns:
        str: The absolute path to the project root directory.
    """
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up: app/page_content -> app -> PROJECT_ROOT
        return os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    except NameError: # pragma: no cover
        st.warning(
            "Could not determine project root via `__file__` (it was undefined). "
            "Falling back to current working directory. Module imports or data paths might fail."
        )
        return os.getcwd()

_PROJECT_ROOT_INV_OPT = _get_project_root_for_inventory_page()

# Attempt to import core inventory model functions
try:
    SRC_PATH_INV = os.path.join(_PROJECT_ROOT_INV_OPT, 'src')
    if SRC_PATH_INV not in sys.path:
        sys.path.insert(0, SRC_PATH_INV)
    # Add project root as well if other parts of the app might need it (e.g. for data paths)
    if _PROJECT_ROOT_INV_OPT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT_INV_OPT)

    from models.inventory_optimization.core_models import (
        calculate_eoq,
        calculate_safety_stock,
        calculate_reorder_point,
        get_z_score,
        run_inventory_simulation
    )
    _CORE_MODELS_LOADED = True
except ImportError as e_import: # pragma: no cover
    _CORE_MODELS_LOADED = False
    st.error(
        f"Error importing core inventory models: {e_import}. "
        "Ensure 'src/models/inventory_optimization/core_models.py' exists, and that "
        "relevant directories (like 'src' and 'src/models') have '__init__.py' files."
    )
    # Define dummy functions if critical imports failed
    def calculate_eoq(*args, **kwargs): st.error("EOQ function unavailable."); return None
    def calculate_safety_stock(*args, **kwargs): st.error("Safety Stock function unavailable."); return None
    def calculate_reorder_point(*args, **kwargs): st.error("ROP function unavailable."); return None
    def get_z_score(*args, **kwargs): st.error("Z-score function unavailable."); return None
    def run_inventory_simulation(*args, **kwargs): st.error("Inventory simulation unavailable."); return pd.DataFrame()
except Exception as e_path: # pragma: no cover
    _CORE_MODELS_LOADED = False
    st.error(f"Unexpected error during sys.path setup or core model imports: {e_path}")


# --- Configuration Paths (using module-level _PROJECT_ROOT_INV_OPT) ---
EXPERIMENT_LOGS_PATH = os.path.join(_PROJECT_ROOT_INV_OPT, 'reports', 'experiment_logs')
PROCESSED_DATA_PATH = os.path.join(_PROJECT_ROOT_INV_OPT, 'data', 'processed')
FEATURED_DATA_FILENAME = 'walmart_data_featured.parquet'


@st.cache_data
def _load_experiment_logs_for_inventory(log_path: str = EXPERIMENT_LOGS_PATH) -> pd.DataFrame | None:
    """
    Loads and combines experiment log CSV files relevant for inventory calculations.

    This function is adapted to fetch logs that can provide demand characteristics
    (like RMSE for std. dev. of demand) from forecasting experiments.

    Args:
        log_path: Path to the directory containing experiment log CSV files.

    Returns:
        pd.DataFrame: A combined DataFrame of logs, or None if loading fails.
    """
    log_files_info = {
        'classical_ts': 'classical_timeseries_experiments.csv',
        'classical_reg': 'classical_regression_experiments.csv',
        'deep_learning': 'deep_learning_experiments.csv'
    }
    all_logs_list = []

    def _safe_literal_eval_dict_inv(val_str):
        """Safely evaluate a string representation of a Python dictionary."""
        if isinstance(val_str, dict): return val_str
        if isinstance(val_str, str):
            try: return ast.literal_eval(val_str)
            except (ValueError, SyntaxError, TypeError): pass
        return {}

    for log_type, file_name in log_files_info.items():
        full_path = os.path.join(log_path, file_name)
        if os.path.exists(full_path):
            try:
                df_log = pd.read_csv(full_path)
                essential_cols = ['Store', 'Dept', 'Model', 'RMSE', 'Test_Period', 'Parameters']
                missing_essentials = [col for col in essential_cols if col not in df_log.columns]
                if missing_essentials:
                    st.info(f"Note: Some expected columns ({missing_essentials}) missing in log file '{file_name}'.")
                df_log['Log_Type_Source'] = log_type
                all_logs_list.append(df_log)
            except Exception as e:
                st.warning(f"Could not load or process log file '{file_name}': {e}")
        else:
            st.info(f"Optional log file not found: '{full_path}'.")

    if not all_logs_list:
        st.error("No experiment logs were found or loaded. Auto-population of demand inputs from logs will be unavailable.")
        return None

    df_combined = pd.concat(all_logs_list, ignore_index=True)
    if 'Parameters' in df_combined.columns:
        df_combined['Parameters_Dict'] = df_combined['Parameters'].apply(_safe_literal_eval_dict_inv)
    else:
        df_combined['Parameters_Dict'] = pd.Series([{} for _ in range(len(df_combined))])

    for col in ['Store', 'Dept']:
        if col in df_combined.columns:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce').fillna(-1).astype(int)
    if 'RMSE' in df_combined.columns: # Primary metric needed from logs
        df_combined['RMSE'] = pd.to_numeric(df_combined['RMSE'], errors='coerce')
    return df_combined

@st.cache_data
def _load_historical_sales_for_inventory(data_path: str = PROCESSED_DATA_PATH, filename: str = FEATURED_DATA_FILENAME) -> pd.DataFrame | None:
    """
    Loads historical sales data, selecting only essential columns for inventory analysis.

    Args:
        data_path: Directory path containing the data file.
        filename: Name of the data file.

    Returns:
        pd.DataFrame: DataFrame with 'Date', 'Store', 'Dept', 'Weekly_Sales', or None if loading fails.
    """
    full_path = os.path.join(data_path, filename)
    if not os.path.exists(full_path):
        st.error(f"Historical sales data file not found: {full_path}")
        return None
    try:
        if full_path.endswith('.parquet'):
            df = pd.read_parquet(full_path)
        elif full_path.endswith('.csv'):
            df = pd.read_csv(full_path, parse_dates=['Date'])
        else:
            st.error(f"Unsupported data file format: {full_path}. Only .parquet or .csv supported.")
            return None

        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Select only necessary columns for this page's context
        required_cols = ['Date', 'Store', 'Dept', 'Weekly_Sales']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Historical sales data is missing required columns: {missing_cols}.")
            return None
        return df[required_cols]
    except Exception as e:
        st.error(f"Error loading historical sales data from '{full_path}': {e}")
        return None

# --- Main Page Rendering Function ---
def render_inventory_optimization_page():
    """
    Renders the Inventory Strategy & Optimization page.

    This includes sections for:
    - Selecting a forecast source to auto-populate demand characteristics.
    - An interactive calculator for EOQ, Safety Stock, and Reorder Point.
    - An inventory simulation module.
    - A sensitivity analysis tool for inventory parameters.
    """
    if not _CORE_MODELS_LOADED: # pragma: no cover
        st.error("Core inventory calculation models could not be loaded. This page cannot function correctly. Please check the import errors above.")
        st.stop()

    sns.set_theme(style="whitegrid", palette="pastel")
    st.header("🧮 Inventory Strategy & Optimization")
    st.markdown(
        "This module demonstrates how demand forecasts and statistical methods can be used to "
        "determine optimal inventory levels and policies."
    )

    df_logs = _load_experiment_logs_for_inventory()
    df_historical_sales = _load_historical_sales_for_inventory()

    if df_logs is None or df_logs.empty:
        st.info("Experiment logs not available. Demand characteristics (Avg. Demand, Std. Dev.) will need to be entered manually below.")
    if df_historical_sales is None or df_historical_sales.empty:
        st.info("Historical sales data not available. Simulation using historical demand patterns will be disabled.")

    # --- Forecast Source Selection for Auto-populating Demand ---
    st.subheader("Forecast Source for Demand Inputs")
    st.caption(
        "Optionally, select a Store, Department, and a previously trained forecasting model "
        "to auto-populate Average Weekly Demand and Standard Deviation of Weekly Demand "
        "fields in the calculator below. Otherwise, enter these values manually."
    )

    sel_col1, sel_col2, sel_col3 = st.columns(3)
    selected_store_inv, selected_dept_inv, selected_model_info_inv = None, None, None
    depts_for_store_inv = [] # Initialize

    unique_stores_inv = []
    if df_logs is not None and 'Store' in df_logs.columns:
        unique_stores_inv = sorted([s for s in df_logs['Store'].unique() if s != -1 and s != 0]) # Exclude global/placeholder

    with sel_col1:
        if not unique_stores_inv:
            st.info("No store-specific forecast logs available for selection.")
        else:
            selected_store_inv = st.selectbox(
                "Store:", unique_stores_inv, index=0, key="inv_store_select",
                help="Select the store relevant to the item for inventory planning."
            )
    with sel_col2:
        if selected_store_inv and df_logs is not None and 'Dept' in df_logs.columns:
            depts_for_store_inv = sorted([
                d for d in df_logs[df_logs['Store'] == selected_store_inv]['Dept'].unique() if d != -1 and d != 0
            ])
            if not depts_for_store_inv:
                st.info(f"No department-specific forecast logs for Store {selected_store_inv}.")
            else:
                selected_dept_inv = st.selectbox(
                    "Department:", depts_for_store_inv, index=0, key="inv_dept_select",
                    help="Select the department relevant to the item."
                )
        elif selected_store_inv is None and unique_stores_inv:
            st.info("Select a Store to view available Departments.")

    with sel_col3:
        if selected_store_inv and selected_dept_inv and df_logs is not None:
            relevant_models_df = df_logs[
                (df_logs['Store'] == selected_store_inv) &
                (df_logs['Dept'] == selected_dept_inv) &
                (df_logs['RMSE'].notna()) # Only models with RMSE can provide Std.Dev.
            ].copy()
            if not relevant_models_df.empty:
                relevant_models_df['display_name'] = (
                    relevant_models_df['Model'] + " (RMSE: " +
                    relevant_models_df['RMSE'].round(2).astype(str) + ")"
                )
                model_display_names = relevant_models_df['display_name'].tolist()
                selected_model_display_inv = st.selectbox(
                    "Forecasting Model:", model_display_names, index=0, key="inv_model_select",
                    help="Select a model. Its RMSE will be used for Std.Dev of Demand, and its test period actuals for Avg. Demand."
                )
                if selected_model_display_inv:
                    selected_model_info_inv = relevant_models_df[
                        relevant_models_df['display_name'] == selected_model_display_inv
                    ].iloc[0]
            else:
                st.info(f"No models with logged RMSE found for S{selected_store_inv}-D{selected_dept_inv}.")
        elif selected_store_inv and not selected_dept_inv and depts_for_store_inv:
            st.info("Select a Department to view available models.")
        elif selected_store_inv is None and unique_stores_inv:
             st.info("Select Store & Department to view models.")

    # --- Auto-populate Demand Parameters if Model Selected ---
    default_avg_weekly_demand, default_std_dev_weekly_demand = 100.0, 20.0 # Default fallbacks
    if selected_model_info_inv is not None:
        logged_rmse = selected_model_info_inv.get('RMSE')
        if pd.notna(logged_rmse):
            default_std_dev_weekly_demand = round(logged_rmse, 2)
            st.caption(f"Std. Dev. of Demand auto-populated from selected model's RMSE: {default_std_dev_weekly_demand}")

        test_period_str = selected_model_info_inv.get('Test_Period')
        params_dict = selected_model_info_inv.get('Parameters_Dict', {})
        # Try to get test period from parameters if not directly in 'Test_Period' column
        if pd.isna(test_period_str) and isinstance(params_dict, dict):
             test_period_str = params_dict.get('test_period_in_params', params_dict.get('test_period'))

        if isinstance(test_period_str, str) and df_historical_sales is not None:
            cleaned_test_period_str = re.sub(r'^[^\w\s\d:-]+|[^\w\s\d:-]+$', '', test_period_str.strip())
            if ' to ' in cleaned_test_period_str:
                try:
                    start_str, end_str = cleaned_test_period_str.split(' to ')
                    start_date = pd.to_datetime(start_str.strip())
                    end_date = pd.to_datetime(end_str.strip())
                    actuals_in_test_period = df_historical_sales[
                        (df_historical_sales['Store'] == selected_store_inv) &
                        (df_historical_sales['Dept'] == selected_dept_inv) &
                        (df_historical_sales['Date'] >= start_date) &
                        (df_historical_sales['Date'] <= end_date)
                    ]['Weekly_Sales']
                    if not actuals_in_test_period.empty:
                        default_avg_weekly_demand = round(actuals_in_test_period.mean(), 2)
                        st.caption(f"Avg. Weekly Demand auto-populated from actual sales during selected model's test period: {default_avg_weekly_demand}")
                    else:
                        st.caption(f"No historical sales data found for S{selected_store_inv}-D{selected_dept_inv} during the logged test period ({test_period_str}). Avg. Demand not auto-populated from historicals.")
                except Exception as e_date_parse:
                    st.caption(f"Could not parse test period '{test_period_str}' to auto-fill Avg. Demand: {e_date_parse}.")
        elif selected_model_info_inv is not None and df_historical_sales is None:
            st.caption("Historical sales data not loaded; Avg. Demand cannot be auto-populated from model's test period.")
        elif selected_model_info_inv is not None:
             st.caption("Test period not defined for selected model; Avg. Demand cannot be auto-populated from model's test period.")

    # --- Interactive Inventory Calculator ---
    st.markdown("---")
    st.subheader("🧮 Interactive Inventory Calculator")
    st.markdown("#### Input Parameters")
    if 'baseline_params_inv' not in st.session_state: # Use page-specific key
        st.session_state.baseline_params_inv = {}

    input_col1, input_col2 = st.columns(2)
    with input_col1:
        st.markdown("**Demand Characteristics (Weekly)**")
        avg_weekly_demand = st.number_input(
            "Average Weekly Demand (Units):", min_value=0.0,
            value=st.session_state.baseline_params_inv.get('avg_weekly_demand', default_avg_weekly_demand),
            step=10.0, format="%.2f", key="inv_avg_demand_input"
        )
        std_dev_weekly_demand = st.number_input(
            "Std. Dev. of Weekly Demand (Units):", min_value=0.0,
            value=st.session_state.baseline_params_inv.get('std_dev_weekly_demand', default_std_dev_weekly_demand),
            step=1.0, format="%.2f", key="inv_std_demand_input",
            help="This can be auto-populated from a selected forecasting model's RMSE."
        )
        st.markdown("**Cost Parameters**")
        item_cost_per_unit = st.number_input(
            "Cost per Unit ($):", min_value=0.01,
            value=st.session_state.baseline_params_inv.get('item_cost_per_unit', 10.0),
            step=0.5, format="%.2f", key="inv_item_cost_input"
        )
        ordering_cost_per_order = st.number_input(
            "Cost per Order ($):", min_value=0.0,
            value=st.session_state.baseline_params_inv.get('ordering_cost_per_order', 50.0),
            step=5.0, format="%.2f", key="inv_order_cost_input"
        )
    with input_col2:
        annual_holding_rate_percent = st.slider( # Moved here for better layout balance
            "Annual Holding Cost Rate (% of Item Cost):", min_value=0, max_value=50,
            value=st.session_state.baseline_params_inv.get('annual_holding_rate_percent', 20),
            step=1, key="inv_holding_rate_input"
        )
        st.markdown("**Lead Time Characteristics (Weekly)**")
        avg_lead_time_weeks = st.number_input(
            "Supplier Lead Time (Weeks):", min_value=0.0,
            value=st.session_state.baseline_params_inv.get('avg_lead_time_weeks', 4.0),
            step=0.5, format="%.1f", key="inv_avg_lt_input"
        )
        std_dev_lead_time_weeks = st.number_input(
            "Std. Dev. of Lead Time (Weeks):", min_value=0.0,
            value=st.session_state.baseline_params_inv.get('std_dev_lead_time_weeks', 1.0),
            step=0.1, format="%.2f", key="inv_std_lt_input"
        )
        st.markdown("**Service Level**")
        target_service_level_percent = st.slider(
            "Target Service Level (%):", min_value=50, max_value=99,
            value=st.session_state.baseline_params_inv.get('target_service_level_percent', 95),
            step=1, key="inv_service_level_input"
        )

    # Update session state with current inputs
    st.session_state.baseline_params_inv = {
        'avg_weekly_demand': avg_weekly_demand, 'std_dev_weekly_demand': std_dev_weekly_demand,
        'item_cost_per_unit': item_cost_per_unit, 'ordering_cost_per_order': ordering_cost_per_order,
        'annual_holding_rate_percent': annual_holding_rate_percent,
        'avg_lead_time_weeks': avg_lead_time_weeks, 'std_dev_lead_time_weeks': std_dev_lead_time_weeks,
        'target_service_level_percent': target_service_level_percent
    }
    if 'inv_calc_results' not in st.session_state: # Initialize if not present
        st.session_state.inv_calc_results = {}

    if st.button("Calculate Optimal Inventory Parameters", key="calc_inventory_btn"):
        st.session_state.inv_calc_results = {} # Reset previous results
        params = st.session_state.baseline_params_inv
        annual_demand = params['avg_weekly_demand'] * 52
        annual_holding_cost_pu = params['item_cost_per_unit'] * (params['annual_holding_rate_percent'] / 100.0)

        eoq_val = None
        if annual_demand > 0 and params['ordering_cost_per_order'] >= 0 and annual_holding_cost_pu > 0:
            eoq_val = calculate_eoq(annual_demand, params['ordering_cost_per_order'], annual_holding_cost_pu)
        else:
            st.warning(
                "EOQ cannot be calculated. Ensure: Avg. Weekly Demand > 0, "
                "Ordering Cost >= 0, and Holding Cost Rate > 0 (or Item Cost > 0)."
            )
        ss_val = calculate_safety_stock(
            params['avg_weekly_demand'], params['std_dev_weekly_demand'],
            params['avg_lead_time_weeks'], params['std_dev_lead_time_weeks'],
            float(params['target_service_level_percent'])
        )
        rop_val = None
        if ss_val is not None: # ROP depends on SS
            rop_val = calculate_reorder_point(
                params['avg_weekly_demand'], params['avg_lead_time_weeks'], ss_val
            )
        # Store all results and inputs used for calculation
        st.session_state.inv_calc_results = {
            'eoq': eoq_val, 'safety_stock': ss_val, 'rop': rop_val,
            'inputs_used': params.copy() # Store a copy of the inputs that led to these results
        }

    # Display calculated results
    if st.session_state.inv_calc_results and any(st.session_state.inv_calc_results.values()):
        results = st.session_state.inv_calc_results
        inputs_used = results.get('inputs_used', {}) # Get the inputs that were used for this calculation
        st.markdown("---"); st.subheader("📈 Calculated Inventory Parameters:")
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            eoq_display = f"{results.get('eoq', 0):.0f} units" if results.get('eoq') is not None else "N/A"
            st.metric(label="Economic Order Quantity (EOQ)", value=eoq_display)
        with res_col2:
            ss_display = f"{results.get('safety_stock', 0):.0f} units" if results.get('safety_stock') is not None else "N/A"
            st.metric(label="Safety Stock (SS)", value=ss_display)
            if results.get('safety_stock') is not None:
                z_val = get_z_score(float(inputs_used.get('target_service_level_percent', 0)))
                if z_val is not None:
                    st.caption(f"For {inputs_used.get('target_service_level_percent',0)}% SL (Z ≈ {z_val:.3f})")
        with res_col3:
            rop_display = f"{results.get('rop', 0):.0f} units" if results.get('rop') is not None else "N/A"
            st.metric(label="Reorder Point (ROP)", value=rop_display)
            if results.get('rop') is not None:
                avg_demand_during_lt = inputs_used.get('avg_weekly_demand',0) * inputs_used.get('avg_lead_time_weeks',0)
                st.caption(f"Avg. Demand during LT: {avg_demand_during_lt:.0f} units")

        # Illustrative Cost Breakdown
        if all(results.get(k) is not None for k in ['eoq', 'safety_stock']) and \
           all(inputs_used.get(k) is not None for k in ['item_cost_per_unit', 'ordering_cost_per_order', 'annual_holding_rate_percent']):
            
            annual_demand_calc = inputs_used['avg_weekly_demand'] * 52
            item_cost_calc = inputs_used['item_cost_per_unit']
            ordering_cost_calc = inputs_used['ordering_cost_per_order']
            holding_rate_calc = inputs_used['annual_holding_rate_percent'] / 100.0
            annual_holding_cost_pu_calc = item_cost_calc * holding_rate_calc
            eoq_val_calc = results['eoq']
            ss_val_calc = results['safety_stock']

            if eoq_val_calc > 0 and annual_holding_cost_pu_calc > 0:
                num_orders = annual_demand_calc / eoq_val_calc
                annual_ordering_cost = num_orders * ordering_cost_calc
                avg_cycle_stock = eoq_val_calc / 2
                annual_holding_cycle = avg_cycle_stock * annual_holding_cost_pu_calc
                annual_holding_safety = ss_val_calc * annual_holding_cost_pu_calc
                total_holding = annual_holding_cycle + annual_holding_safety
                total_relevant_cost = annual_ordering_cost + total_holding

                st.markdown("#### Cost Breakdown (Annualized - Illustrative)")
                cost_data = {
                    "Cost Component": [
                        "Annual Ordering Cost", "Annual Holding Cost (Cycle Stock)",
                        "Annual Holding Cost (Safety Stock)", "Total Annual Holding Cost",
                        "Total Relevant Inventory Cost (Ordering + Holding)"
                    ],
                    "Value ($)": [
                        f"{annual_ordering_cost:.2f}", f"{annual_holding_cycle:.2f}",
                        f"{annual_holding_safety:.2f}", f"{total_holding:.2f}",
                        f"{total_relevant_cost:.2f}"
                    ]
                }
                st.table(pd.DataFrame(cost_data))
        st.markdown("---")


    # --- Inventory Simulation Section ---
    st.subheader("🧪 Inventory Simulation")
    st.markdown(
        "Simulate inventory levels over time based on the calculated parameters (or custom inputs if re-calculated) "
        "and a chosen demand pattern."
    )
    sim_inputs_col1, sim_inputs_col2 = st.columns(2)
    with sim_inputs_col1:
        simulation_weeks = st.number_input(
            "Simulation Duration (Weeks):", min_value=4, max_value=208, value=52, step=4, key="sim_weeks"
        )
        # Default initial inventory to ROP or a fallback if ROP not calculated
        default_sim_initial_inv = int(st.session_state.inv_calc_results.get('rop', 150)
                                     if st.session_state.inv_calc_results and st.session_state.inv_calc_results.get('rop') is not None
                                     else 150)
        initial_inventory = st.number_input(
            "Initial Inventory Level (Units):", min_value=0,
            value=default_sim_initial_inv, step=10, key="sim_initial_inv"
        )
    with sim_inputs_col2:
        demand_pattern_options = ["Use Constant Average Weekly Demand (from inputs)"]
        hist_demand_label = "Disabled (Historical Data/Selections Missing)"
        if df_historical_sales is not None and selected_store_inv is not None and selected_dept_inv is not None:
            hist_demand_label = f"Use Actual Historical Demand (S{selected_store_inv}-D{selected_dept_inv})"
            demand_pattern_options.append(hist_demand_label)
        selected_demand_pattern = st.selectbox(
            "Demand Pattern for Simulation:", options=demand_pattern_options, key="sim_demand_pattern"
        )

    if st.button("Run Inventory Simulation", key="run_sim_btn"):
        calc_results = st.session_state.get('inv_calc_results', {})
        baseline_p = st.session_state.get('baseline_params_inv', {})

        if not calc_results or calc_results.get('eoq') is None or calc_results.get('rop') is None:
            st.warning("Please calculate optimal inventory parameters (EOQ, ROP) first using the calculator above before running the simulation.")
        else:
            eoq_for_sim = float(calc_results['eoq'])
            rop_for_sim = float(calc_results['rop'])
            # Use integer lead time for simulation periods
            lead_time_for_sim = int(round(baseline_p.get('avg_lead_time_weeks', 1)))

            demand_series_for_sim = []
            if selected_demand_pattern.startswith("Use Constant Average"):
                demand_series_for_sim = [baseline_p.get('avg_weekly_demand', 100)] * simulation_weeks
            elif selected_demand_pattern.startswith("Use Actual Historical Demand") and df_historical_sales is not None:
                hist_sales_query = df_historical_sales[
                    (df_historical_sales['Store'] == selected_store_inv) &
                    (df_historical_sales['Dept'] == selected_dept_inv)
                ]['Weekly_Sales'].sort_index().values # Ensure sorted by date if not already
                
                if len(hist_sales_query) >= simulation_weeks:
                    demand_series_for_sim = hist_sales_query[:simulation_weeks]
                elif len(hist_sales_query) > 0: # Tile if historical data is shorter than simulation period
                    demand_series_for_sim = np.tile(
                        hist_sales_query, (simulation_weeks // len(hist_sales_query)) + 1
                    )[:simulation_weeks]
                else:
                    st.warning(
                        f"No historical sales data found for S{selected_store_inv}-D{selected_dept_inv}. "
                        "Simulation will use constant average demand instead."
                    )
                    demand_series_for_sim = [baseline_p.get('avg_weekly_demand', 100)] * simulation_weeks
            
            if not (isinstance(demand_series_for_sim, (list, np.ndarray)) and len(demand_series_for_sim) == simulation_weeks and simulation_weeks > 0):
                st.error(
                    f"Could not prepare a valid demand series for simulation. "
                    f"Expected {simulation_weeks} weeks of demand data, but preparation failed or yielded an incorrect length."
                )
            else:
                with st.spinner("Running inventory simulation... This may take a moment."):
                    sim_results_df = run_inventory_simulation(
                        demand_series=np.array(demand_series_for_sim, dtype=float),
                        initial_inventory=float(initial_inventory),
                        order_quantity_eoq=float(eoq_for_sim),
                        reorder_point_rop=float(rop_for_sim),
                        lead_time_periods=lead_time_for_sim,
                        simulation_periods=simulation_weeks
                    )

                if sim_results_df is not None and not sim_results_df.empty:
                    st.success("Simulation complete!")
                    st.markdown("#### Simulation Results Summary")
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    total_demand_sim = sim_results_df['Demand'].sum()
                    total_sold_sim = sim_results_df['Units_Sold'].sum()
                    total_stockouts_sim = sim_results_df['Stockout_Units'].sum()
                    num_orders_sim = len(sim_results_df[sim_results_df['Order_Placed_Qty'] > 0])

                    summary_col1.metric("Total Demand Simulated", f"{total_demand_sim:.0f} units")
                    summary_col1.metric("Total Units Sold", f"{total_sold_sim:.0f} units")
                    summary_col2.metric("Total Stockout Units", f"{total_stockouts_sim:.0f} units",
                                        delta_color="inverse" if total_stockouts_sim > 0 else "normal")
                    if total_demand_sim > 0:
                        fill_rate = (total_sold_sim / total_demand_sim) * 100
                        summary_col2.metric("Service Level (Fill Rate)", f"{fill_rate:.2f}%")
                    summary_col3.metric("Number of Orders Placed", f"{num_orders_sim}")

                    st.markdown("#### Simulated Inventory Level Over Time")
                    plt.style.use('seaborn-v0_8-darkgrid')
                    fig_sim, ax_sim = plt.subplots(figsize=(16, 7))
                    ax_sim.plot(sim_results_df['Period'], sim_results_df['Inventory_End_Period'],
                                label='Inventory Level', color='dodgerblue', marker='o', markersize=3,
                                linestyle='-', linewidth=1.5, zorder=5)
                    ax_sim.axhline(y=rop_for_sim, color='darkorange', linestyle='--', linewidth=2,
                                   label=f'Reorder Point (ROP ≈ {rop_for_sim:.0f})', zorder=3)
                    if calc_results.get('safety_stock') is not None:
                         ss_val_plot = calc_results['safety_stock']
                         ax_sim.axhline(y=ss_val_plot, color='forestgreen', linestyle=':', linewidth=2,
                                        label=f'Safety Stock (SS ≈ {ss_val_plot:.0f})', zorder=3)

                    stockout_periods_df = sim_results_df[sim_results_df['Stockout_Units'] > 0]
                    if not stockout_periods_df.empty:
                        ax_sim.scatter(stockout_periods_df['Period'],
                                       sim_results_df.loc[stockout_periods_df.index, 'Inventory_End_Period'], # Ensure correct y-values for scatter
                                       color='red', marker='X', s=80, label='Stockout Occurred', zorder=10)

                    ax_sim.set_xlabel("Simulation Week", fontsize=12)
                    ax_sim.set_ylabel("Units in Inventory", fontsize=12)
                    ax_sim.set_title("Simulated Inventory Level Over Time", fontsize=16, fontweight='bold')
                    ax_sim.legend(fontsize=10)
                    ax_sim.grid(True, linestyle='--', alpha=0.7)
                    ax_sim.tick_params(axis='both', which='major', labelsize=10)
                    plt.fill_between(sim_results_df['Period'], 0, sim_results_df['Inventory_End_Period'],
                                     alpha=0.1, color='dodgerblue')
                    plt.tight_layout()
                    st.pyplot(fig_sim)
                    plt.close(fig_sim) # Close plot

                    if st.checkbox("Show Detailed Simulation Results Table", False, key="show_sim_table"):
                        st.dataframe(sim_results_df.style.format("{:.0f}", subset=['Demand', 'Inventory_Start_Period', 'Inventory_End_Period', 'Units_Sold', 'Order_Received_Qty', 'Order_Placed_Qty', 'Stockout_Units']))
                else:
                    st.error("Simulation did not produce any results. Please check inputs and console for errors.")

    # --- Sensitivity Analysis Section ---
    st.markdown("---")
    st.subheader("🔬 Sensitivity Analysis")
    st.markdown(
        "Explore how changes in key input parameters (one at a time) affect the optimal "
        "inventory levels (EOQ, Safety Stock, ROP), holding other parameters at their current 'calculator input' values."
    )
    baseline_params_sens = st.session_state.get('baseline_params_inv', {}).copy()
    if not baseline_params_sens:
        st.info("Please enter baseline parameters in the calculator above to enable sensitivity analysis.")
        return # Stop if no baseline params

    param_to_vary_options = {
        "Annual Holding Cost Rate (%)": "annual_holding_rate_percent",
        "Ordering Cost per Order ($)": "ordering_cost_per_order",
        "Target Service Level (%)": "target_service_level_percent",
        "Average Weekly Demand (Units)": "avg_weekly_demand",
        "Std. Dev. of Weekly Demand (Units)": "std_dev_weekly_demand",
        "Supplier Lead Time (Weeks)": "avg_lead_time_weeks",
        "Std. Dev. of Lead Time (Weeks)": "std_dev_lead_time_weeks"
    }
    selected_param_display_name = st.selectbox(
        "Select Parameter to Vary:", options=list(param_to_vary_options.keys()), key="sens_param_select"
    )
    selected_param_key = param_to_vary_options[selected_param_display_name]

    # Define sensible default ranges and step for each parameter
    default_val = baseline_params_sens.get(selected_param_key, 0)
    default_range_config = {
        "annual_holding_rate_percent": (max(0.1, default_val * 0.5), default_val * 1.5, 5.0, "%.1f"),
        "ordering_cost_per_order": (max(0, default_val * 0.5), default_val * 1.5, 10.0, "%.2f"),
        "target_service_level_percent": (max(50.0,default_val - 15), min(99.9, default_val + 15), 2.0, "%.1f"),
        "avg_weekly_demand": (max(1, default_val * 0.5), default_val * 1.5, max(1.0, float(default_val*0.1)), "%.2f"),
        "std_dev_weekly_demand": (max(0, default_val * 0.5), default_val * 1.5, max(1.0, float(default_val*0.1)), "%.2f"),
        "avg_lead_time_weeks": (max(0.1, default_val * 0.5), default_val * 1.5, 0.5, "%.1f"), # Lead time > 0
        "std_dev_lead_time_weeks": (max(0, default_val * 0.5), default_val * 1.5, 0.2, "%.2f")
    }
    current_range = default_range_config.get(selected_param_key, (0, 100, 10, "%.2f"))

    sens_range_col1, sens_range_col2, sens_range_col3 = st.columns(3)
    with sens_range_col1:
        vary_min = st.number_input("Min Value for Variation:", value=float(current_range[0]), format=current_range[3], key="sens_min")
    with sens_range_col2:
        vary_max = st.number_input("Max Value for Variation:", value=float(current_range[1]), format=current_range[3], key="sens_max")
    with sens_range_col3:
        num_steps = st.number_input("Number of Steps for Variation:", min_value=2, max_value=30, value=10, step=1, key="sens_steps")

    if st.button("Run Sensitivity Analysis", key="run_sens_analysis_btn"):
        if vary_min >= vary_max:
            st.warning("Min value for variation must be strictly less than Max value.")
        else:
            param_values_to_test = np.linspace(vary_min, vary_max, int(num_steps))
            sensitivity_results_list = []
            with st.spinner(f"Running sensitivity analysis for '{selected_param_display_name}'..."):
                for val in param_values_to_test:
                    current_params_iter = baseline_params_sens.copy()
                    current_params_iter[selected_param_key] = val # Set the varying parameter

                    annual_demand_iter = current_params_iter['avg_weekly_demand'] * 52
                    annual_holding_cost_pu_iter = current_params_iter['item_cost_per_unit'] * \
                                                  (current_params_iter['annual_holding_rate_percent'] / 100.0)
                    eoq_sens_iter = None
                    if annual_demand_iter > 0 and current_params_iter['ordering_cost_per_order'] >= 0 and annual_holding_cost_pu_iter > 0:
                        eoq_sens_iter = calculate_eoq(
                            annual_demand_iter, current_params_iter['ordering_cost_per_order'], annual_holding_cost_pu_iter
                        )
                    ss_sens_iter = calculate_safety_stock(
                        current_params_iter['avg_weekly_demand'], current_params_iter['std_dev_weekly_demand'],
                        current_params_iter['avg_lead_time_weeks'], current_params_iter['std_dev_lead_time_weeks'],
                        float(current_params_iter['target_service_level_percent'])
                    )
                    rop_sens_iter = None
                    if ss_sens_iter is not None:
                        rop_sens_iter = calculate_reorder_point(
                            current_params_iter['avg_weekly_demand'], current_params_iter['avg_lead_time_weeks'], ss_sens_iter
                        )
                    sensitivity_results_list.append({
                        selected_param_display_name: val,
                        'EOQ': eoq_sens_iter,
                        'Safety_Stock': ss_sens_iter,
                        'ROP': rop_sens_iter
                    })
            results_df_sens = pd.DataFrame(sensitivity_results_list)

            st.markdown("#### Sensitivity Analysis Results Table")
            st.dataframe(results_df_sens.style.format("{:.2f}", na_rep="N/A"))

            st.markdown("#### Sensitivity Plots")
            fig_sens, axes_sens = plt.subplots(3, 1, figsize=(12, 18), sharex=True) # Increased figure size
            plot_vars = ['EOQ', 'Safety_Stock', 'ROP']
            colors = ['crimson', 'dodgerblue', 'forestgreen']
            titles = ["EOQ Sensitivity", "Safety Stock Sensitivity", "Reorder Point Sensitivity"]

            for i, var_to_plot in enumerate(plot_vars):
                sns.lineplot(data=results_df_sens, x=selected_param_display_name, y=var_to_plot,
                             ax=axes_sens[i], marker='o', color=colors[i], legend=False) # hue=None removed
                axes_sens[i].set_ylabel(f"{var_to_plot.replace('_',' ')} (Units)", fontsize=11)
                axes_sens[i].set_title(titles[i], fontsize=14)
                axes_sens[i].grid(True, linestyle=':', alpha=0.6)
                axes_sens[i].tick_params(axis='both', which='major', labelsize=10)
            axes_sens[-1].set_xlabel(f"Varied Parameter: {selected_param_display_name}", fontsize=12)
            plt.tight_layout(pad=2.0) # Add padding
            st.pyplot(fig_sens)
            plt.close(fig_sens)

    st.markdown("---")
    st.caption(
        "Note: For safety stock and reorder point calculations, demand and lead time inputs "
        "(average and standard deviation) must be in the same time unit (e.g., weekly)."
    )

if __name__ == "__main__": # pragma: no cover
    # This block facilitates standalone testing of this page.
    # Ensure necessary data files are accessible and core_models can be imported.
    # st.set_page_config(layout="wide", page_title="Inventory Optimization")
    if not _CORE_MODELS_LOADED:
        st.error("Standalone run: Core inventory models failed to load. Page functionality is severely limited.")
    render_inventory_optimization_page()