"""
Renders the Model Performance Comparison page for the Streamlit application.

This page loads, processes, and visualizes experiment logs from various
forecasting models. It allows users to compare model performance through
summary tables and plots, focusing on metrics like MAE, RMSE, and MAPE.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast # For safely evaluating string representations of dicts/lists
import sys # For sys.path modifications

# --- Project Path Setup ---
_PROJECT_ROOT_MODEL_PERF = None # Module-level variable

def _get_project_root_for_model_perf_page() -> str:
    """
    Determines the project root directory for this model performance page.

    Assumes this script is located within: `PROJECT_ROOT/app/page_content/`.
    Navigates up two directories from this file's location. This is primarily
    for locating the 'reports/experiment_logs' directory, especially useful
    if running this page script standalone.

    Returns:
        str: The absolute path to the project root directory.
    """
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # app/page_content -> app -> PROJECT_ROOT
        project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
        return project_root
    except NameError: # pragma: no cover
        # This fallback is for environments where __file__ might not be defined
        # (e.g., some interactive shells or specific execution contexts).
        st.warning(
            "Could not automatically determine project root using `__file__`. "
            "Falling back to current working directory. Log paths might be incorrect "
            "if not run from the project's root directory or via the main application."
        )
        return os.getcwd()

_PROJECT_ROOT_MODEL_PERF = _get_project_root_for_model_perf_page()

# Ensure project root is in sys.path for potential utility imports if this page evolves
if _PROJECT_ROOT_MODEL_PERF not in sys.path: # pragma: no cover
    sys.path.insert(0, _PROJECT_ROOT_MODEL_PERF)

EXPERIMENT_LOGS_PATH = os.path.join(_PROJECT_ROOT_MODEL_PERF, 'reports', 'experiment_logs')

@st.cache_data
def _load_and_process_all_logs(log_path: str = EXPERIMENT_LOGS_PATH) -> pd.DataFrame | None:
    """
    Loads, combines, and processes all experiment log CSV files.

    This function searches for predefined log files, concatenates them,
    parses parameter strings into dictionaries, ensures numeric metrics,
    and categorizes models into families.

    Args:
        log_path: Path to the directory containing experiment log CSV files.

    Returns:
        pd.DataFrame: A combined and processed DataFrame of all logs,
                      or None if no logs are loaded or critical errors occur.
    """
    log_files_info = {
        'classical_ts': 'classical_timeseries_experiments.csv',
        'classical_reg': 'classical_regression_experiments.csv',
        'deep_learning': 'deep_learning_experiments.csv'
    }
    all_logs_list = []

    def _safe_literal_eval_perf(val_str: str | dict) -> dict:
        """
        Safely evaluate a string representation of a Python dictionary.
        If already a dict, returns it. If evaluation fails, returns an empty dict.
        """
        if isinstance(val_str, dict):
            return val_str
        if isinstance(val_str, str):
            try:
                return ast.literal_eval(val_str)
            except (ValueError, SyntaxError, TypeError):
                return {} # Fallback for malformed strings
        return {} # Fallback for other types

    def _get_model_family(model_name_str: str | None) -> str:
        """
        Categorizes a model name into a broader model family.
        """
        if not isinstance(model_name_str, str) or not model_name_str:
            return 'Unknown'
        model_name_lower = model_name_str.lower()
        if 'naive' in model_name_lower: return 'Baseline Naive'
        if 'ets' in model_name_lower: return 'ETS'
        if 'sarima' in model_name_lower: return 'SARIMA'
        if 'prophet' in model_name_lower: return 'Prophet'
        if 'randomforest' in model_name_lower or 'random_forest' in model_name_lower: return 'Random Forest'
        if 'xgboost' in model_name_lower: return 'XGBoost'
        if 'lightgbm' in model_name_lower or 'lgbm' in model_name_lower: return 'LightGBM'
        if 'lstm' in model_name_lower: return 'LSTM'
        if 'dummy' in model_name_lower: return 'Dummy/Test' # For any dummy/test models
        return 'Other Classical' # Default for unrecognized classical models

    for log_type, file_name in log_files_info.items():
        full_path = os.path.join(log_path, file_name)
        if os.path.exists(full_path):
            try:
                df_log = pd.read_csv(full_path)
                essential_cols = ['Model', 'Parameters', 'RMSE', 'MAE', 'MAPE']
                missing_cols = [col for col in essential_cols if col not in df_log.columns]
                if missing_cols:
                    st.info(f"Note: Some expected columns ({missing_cols}) missing in log file '{file_name}'. Results may be partial.")
                df_log['Log_Type_Source'] = log_type
                all_logs_list.append(df_log)
            except Exception as e:
                st.warning(f"Could not load or process log file '{file_name}': {e}")
        else:
            st.info(f"Optional log file not found: '{full_path}'. This may be expected if not all model types were run.")

    if not all_logs_list:
        st.error("No experiment logs were found or successfully loaded. Cannot display model performance.")
        return None

    df_combined = pd.concat(all_logs_list, ignore_index=True)

    # Process 'Parameters' into a dictionary column
    if 'Parameters' in df_combined.columns:
        df_combined['Parameters_Dict'] = df_combined['Parameters'].apply(_safe_literal_eval_perf)
    else:
        df_combined['Parameters_Dict'] = pd.Series([{} for _ in range(len(df_combined))])

    # Ensure metric columns are numeric, adding them with NaNs if entirely missing
    metric_cols = ['MAE', 'RMSE', 'MAPE']
    for col in metric_cols:
        if col in df_combined.columns:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        else:
            df_combined[col] = np.nan # Add column with NaNs if it doesn't exist

    # Create Model_Family column for easier grouping and visualization
    if 'Model' in df_combined.columns:
        df_combined['Model_Family'] = df_combined['Model'].apply(_get_model_family)
    else:
        st.warning("'Model' column not found in combined logs. Cannot create 'Model_Family'. Assigning 'Unknown'.")
        df_combined['Model_Family'] = 'Unknown'

    # Standardize Store and Dept columns if they exist, or add placeholders
    for col_sd in ['Store', 'Dept']:
        if col_sd in df_combined.columns:
            df_combined[col_sd] = pd.to_numeric(df_combined[col_sd], errors='coerce').fillna(-1).astype(int)
        else:
            df_combined[col_sd] = -1 # Placeholder for logs without Store/Dept (e.g. global models)
    return df_combined

def render_model_performance_page():
    """
    Renders the Model Performance Comparison page.

    Displays aggregated results from model training experiments, including
    summary tables and visualizations comparing performance across different
    model families and specific model configurations.
    """
    sns.set_theme(style="whitegrid", palette="muted") # Apply a consistent theme
    st.header("🏆 Model Performance Insights & Comparison")
    st.markdown("""
    This section aggregates results from all trained forecasting models, allowing for
    a comparative analysis of their performance based on metrics like MAE, RMSE, and MAPE.
    Lower values for these metrics generally indicate better model performance.
    """)

    with st.spinner("Loading and processing experiment logs... This might take a moment."):
        df_logs = _load_and_process_all_logs()

    if df_logs is None or df_logs.empty:
        return # Stop if no logs are available

    st.success(f"Loaded a total of {len(df_logs)} experiment runs for performance analysis.")

    if st.checkbox("Show Combined Experiment Logs Dataframe", False, key="show_combined_logs_perf"):
        st.dataframe(df_logs)
        st.caption(f"Available columns in the combined log: {df_logs.columns.tolist()}")

    # --- Summary Tables ---
    st.subheader("1. Performance Summary Tables")

    st.markdown("#### Average Metrics by Model Family")
    st.caption("Average performance metrics (MAE, RMSE, MAPE) across all logged runs for each model family.")
    if 'Model_Family' in df_logs.columns and not df_logs[['MAE', 'RMSE', 'MAPE']].isnull().all().all():
        avg_metrics_family = df_logs.groupby('Model_Family')[['MAE', 'RMSE', 'MAPE']].mean().sort_values(by='RMSE')
        st.dataframe(avg_metrics_family.style.format("{:.3f}")) # Increased precision
    else:
        st.info("Not enough data or relevant metric columns (MAE, RMSE, MAPE) to calculate average metrics by model family.")

    st.markdown("#### Average Metrics by Specific Model Name")
    st.caption("Average performance metrics for each unique model name/configuration logged.")
    if 'Model' in df_logs.columns and not df_logs[['MAE', 'RMSE', 'MAPE']].isnull().all().all():
        avg_metrics_model = df_logs.groupby('Model')[['MAE', 'RMSE', 'MAPE']].mean().sort_values(by='RMSE')
        st.dataframe(avg_metrics_model.style.format("{:.3f}"))
    else:
        st.info("Not enough data or metric columns to calculate average metrics by specific model name.")

    st.markdown("#### Top Performing Model per Store-Department (based on RMSE)")
    st.caption("Identifies the model with the lowest RMSE for each unique Store-Department combination (excluding global models).")
    # Ensure necessary columns exist and filter out placeholder/global model entries for this specific ranking
    if all(col in df_logs.columns for col in ['Store', 'Dept', 'RMSE', 'Model']):
        df_to_rank = df_logs.dropna(subset=['Store', 'Dept', 'RMSE', 'Model'])
        # Exclude global models (Store/Dept are 0 or -1) from this store-department specific ranking
        df_specific_store_dept_models = df_to_rank[(df_to_rank['Store'] > 0) & (df_to_rank['Dept'] > 0)]

        if not df_specific_store_dept_models.empty:
            try:
                # idxmin finds the index of the minimum RMSE for each group
                best_models_idx = df_specific_store_dept_models.groupby(['Store', 'Dept'])['RMSE'].idxmin()
                best_models_summary = df_specific_store_dept_models.loc[best_models_idx]

                cols_to_display = ['Store', 'Dept', 'Model', 'Model_Family', 'RMSE', 'MAE', 'MAPE']
                # Filter to columns that actually exist in the summary
                cols_to_display = [col for col in cols_to_display if col in best_models_summary.columns]

                if not best_models_summary.empty:
                    st.dataframe(
                        best_models_summary[cols_to_display].sort_values(by=['Store', 'Dept'])
                        .style.format({'RMSE': "{:.2f}", 'MAE': "{:.2f}", 'MAPE': "{:.2f}%"})
                    )
                    if 'Model_Family' in best_models_summary.columns:
                        st.markdown("##### Count of Top Performing Model Families (Store-Dept Specific)")
                        st.dataframe(best_models_summary['Model_Family'].value_counts())
                else:
                    st.info("No best performing models found after filtering for store-specific results.")
            except Exception as e_rank:
                st.error(f"An error occurred during model ranking: {e_rank}")
        else:
            st.info("Not enough data (excluding global models) with Store, Dept, and RMSE to rank models per Store-Department.")
    else:
        st.info("Required columns for ranking (Store, Dept, RMSE, Model) are not all present in the logs.")


    # --- Visualizations ---
    st.subheader("2. Performance Visualizations")
    plt.style.use('seaborn-v0_8-whitegrid') # Consistent style for plots

    metric_plot_config = [
        {'metric': 'RMSE', 'palette': 'viridis', 'title_suffix': '(Lower is Better)'},
        {'metric': 'MAPE', 'palette': 'magma', 'title_suffix': '(Lower is Better, %)'}
    ]

    for config in metric_plot_config:
        metric = config['metric']
        if 'Model_Family' in df_logs.columns and not df_logs[metric].isnull().all():
            st.markdown(f"#### Average {metric} by Model Family")
            avg_metric_family_data = df_logs.groupby('Model_Family')[metric].mean().dropna().sort_values()
            if not avg_metric_family_data.empty:
                fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                avg_metric_family_data.plot(kind='bar', ax=ax_bar, color=sns.color_palette(config['palette'], len(avg_metric_family_data)))
                ax_bar.set_title(f'Average {metric} by Model Family {config["title_suffix"]}', fontsize=14)
                ax_bar.set_ylabel(f'Average {metric}', fontsize=12)
                ax_bar.set_xlabel('Model Family', fontsize=12)
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.yticks(fontsize=10)
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar) # Close plot
            else:
                st.info(f"No data to plot for Average {metric} by Model Family.")

            st.markdown(f"#### Distribution of {metric} by Model Family")
            # Filter data for boxplot: remove NaNs in metric and Model_Family
            metric_boxplot_data = df_logs.dropna(subset=[metric, 'Model_Family'])
            if metric == 'MAPE': # Special handling for MAPE outliers
                mape_upper_cap = metric_boxplot_data[metric].quantile(0.98) if not metric_boxplot_data.empty else 200
                plot_data_boxplot = metric_boxplot_data[metric_boxplot_data[metric] <= mape_upper_cap]
                plot_title = f'Distribution of {metric} by Model Family (Values capped at {mape_upper_cap:.0f}%)'
            else:
                plot_data_boxplot = metric_boxplot_data
                plot_title = f'Distribution of {metric} by Model Family'

            if not plot_data_boxplot.empty:
                order_boxplot = plot_data_boxplot.groupby('Model_Family')[metric].median().sort_values().index
                fig_box, ax_box = plt.subplots(figsize=(12, 7))
                sns.boxplot(data=plot_data_boxplot, x='Model_Family', y=metric, ax=ax_box,
                            order=order_boxplot, hue='Model_Family', legend=False, # hue can be redundant if x is Model_Family
                            palette=config['palette'], linewidth=1.5)
                ax_box.set_title(plot_title, fontsize=14)
                ax_box.set_ylabel(metric, fontsize=12)
                ax_box.set_xlabel('Model Family', fontsize=12)
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.yticks(fontsize=10)
                plt.tight_layout()
                st.pyplot(fig_box)
                plt.close(fig_box) # Close plot
            else:
                st.info(f"Not enough valid data to generate {metric} distribution boxplot.")
        else:
            st.warning(f"Cannot generate {metric} plots: 'Model_Family' column or '{metric}' data missing or all NaN.")


if __name__ == "__main__": # pragma: no cover
    # This block is for standalone testing of this page.
    # It changes the CWD if run from `page_content` to ensure relative paths to `reports` work.
    if os.path.basename(os.getcwd()) == 'page_content':
        os.chdir(os.path.join("..", "..")) # Navigate to project root

    # Redefine PROJECT_ROOT and EXPERIMENT_LOGS_PATH for standalone context
    _PROJECT_ROOT_MODEL_PERF = os.getcwd()
    EXPERIMENT_LOGS_PATH = os.path.join(_PROJECT_ROOT_MODEL_PERF, 'reports', 'experiment_logs')

    # Page configuration would typically be in main_app.py
    # st.set_page_config(layout="wide", page_title="Model Performance Insights")
    render_model_performance_page() # Corrected function name