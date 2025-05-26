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
        st.warning(
            "Could not automatically determine project root using `__file__`. "
            "Falling back to current working directory. Log paths might be incorrect "
            "if not run from the project's root directory or via the main application."
        )
        return os.getcwd()

_PROJECT_ROOT_MODEL_PERF = _get_project_root_for_model_perf_page()

if _PROJECT_ROOT_MODEL_PERF not in sys.path: # pragma: no cover
    sys.path.insert(0, _PROJECT_ROOT_MODEL_PERF)

EXPERIMENT_LOGS_PATH = os.path.join(_PROJECT_ROOT_MODEL_PERF, 'reports', 'experiment_logs')

@st.cache_data
def _load_and_process_all_logs(log_path: str = EXPERIMENT_LOGS_PATH) -> pd.DataFrame | None:
    """
    Loads, combines, and processes all experiment log CSV files from the specified path.

    This function searches for predefined log files based on `log_files_info`,
    concatenates them if found, parses parameter strings into dictionaries,
    ensures metrics columns (MAE, RMSE, MAPE) are numeric, and categorizes
    models into broader families for easier analysis.

    Args:
        log_path (str): The file system path to the directory containing
                        experiment log CSV files. Defaults to EXPERIMENT_LOGS_PATH.

    Returns:
        pd.DataFrame | None: A combined and processed Pandas DataFrame of all
                              loaded logs. Returns None if no logs are found
                              or if a critical error occurs during loading or
                              initial processing.
    """
    log_files_info = {
        'classical_ts': 'classical_timeseries_experiments.csv',
        'classical_reg': 'classical_regression_experiments.csv',
        'deep_learning': 'deep_learning_experiments.csv'
    }
    all_logs_list = []

    def _safe_literal_eval_perf(val_str: str | dict) -> dict:
        """
        Safely evaluates a string representation of a Python dictionary.

        If the input is already a dictionary, it's returned directly.
        If the input is a string, `ast.literal_eval` is used.
        Handles potential errors by returning an empty dictionary.

        Args:
            val_str (str | dict): The input string suspected to be a dictionary
                                  representation, or an actual dictionary.

        Returns:
            dict: The evaluated dictionary. Returns an empty dictionary if
                  evaluation fails or if the input is not a string or dict.
        """
        if isinstance(val_str, dict):
            return val_str
        if isinstance(val_str, str):
            try:
                return ast.literal_eval(val_str)
            except (ValueError, SyntaxError, TypeError): # Catch specific errors
                return {} # Fallback for malformed strings
        return {} # Fallback for other unexpected types

    def _get_model_family(model_name_str: str | None) -> str:
        """
        Categorizes a given model name string into a predefined model family.

        This helps in grouping similar models for aggregated performance analysis.
        The categorization is based on common keywords found in model names.

        Args:
            model_name_str (str | None): The name of the model as a string.
                                         Can be None or empty.

        Returns:
            str: The determined model family (e.g., 'SARIMA', 'Random Forest',
                 'LSTM'). Returns 'Unknown' if the model name is not recognized
                 or if the input is invalid (None or empty).
        """
        if not isinstance(model_name_str, str) or not model_name_str.strip():
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
        if 'dummy' in model_name_lower: return 'Dummy/Test'
        return 'Other Classical' # Default for unrecognized classical models

    for log_type, file_name in log_files_info.items():
        full_path = os.path.join(log_path, file_name)
        if os.path.exists(full_path):
            try:
                df_log = pd.read_csv(full_path)
                # Check for essential columns before appending
                essential_cols_check = ['Model', 'Parameters', 'RMSE', 'MAE', 'MAPE']
                if not all(col in df_log.columns for col in essential_cols_check):
                    st.info(
                        f"Note: Log file '{file_name}' is missing one or more essential columns "
                        f"({', '.join(essential_cols_check)}). It will be processed, but some analyses might be affected."
                    )
                df_log['Log_Type_Source'] = log_type
                all_logs_list.append(df_log)
            except Exception as e:
                st.warning(f"Could not load or process log file '{file_name}': {e}")
        else:
            st.info(f"Note: Log file not found: '{full_path}'. This might be expected if not all model types were run.")

    if not all_logs_list:
        st.error("CRITICAL: No experiment logs were found or successfully loaded. Cannot display model performance.")
        return None

    df_combined = pd.concat(all_logs_list, ignore_index=True)

    # Process 'Parameters' into a dictionary column
    if 'Parameters' in df_combined.columns:
        df_combined['Parameters_Dict'] = df_combined['Parameters'].apply(_safe_literal_eval_perf)
    else:
        # Ensure Parameters_Dict column exists even if 'Parameters' column is missing from all logs
        df_combined['Parameters_Dict'] = pd.Series([{} for _ in range(len(df_combined))])


    # Ensure metric columns are numeric, adding them with NaNs if entirely missing
    metric_cols = ['MAE', 'RMSE', 'MAPE']
    for col in metric_cols:
        if col in df_combined.columns:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        else:
            df_combined[col] = np.nan # Add column with NaNs if it doesn't exist from any log file

    # Create Model_Family column for easier grouping and visualization
    if 'Model' in df_combined.columns:
        df_combined['Model_Family'] = df_combined['Model'].apply(_get_model_family)
    else:
        # Ensure Model_Family column exists even if 'Model' column is missing
        df_combined['Model_Family'] = 'Unknown'
        st.warning("Warning: 'Model' column not found in combined logs. 'Model_Family' set to 'Unknown'.")


    # Standardize Store and Dept columns if they exist, or add placeholders
    for col_sd in ['Store', 'Dept']:
        if col_sd in df_combined.columns:
            df_combined[col_sd] = pd.to_numeric(df_combined[col_sd], errors='coerce').fillna(-1).astype(int)
        else:
            df_combined[col_sd] = -1 # Placeholder for logs without Store/Dept
    return df_combined

def render_model_performance_page():
    """
    Renders the Model Performance Comparison page with detailed explanations
    and articulated insights for each component.

    This page aggregates results from all trained forecasting models, enabling
    a comprehensive comparative analysis of their performance based on key
    metrics such as MAE, RMSE, and MAPE. It provides insights into which
    model families and specific configurations perform best, both overall
    and for specific store-department combinations. The visualisations
    further aid in understanding performance distributions and consistencies.
    """
    sns.set_theme(style="whitegrid", palette="muted") # Consistent theme for plots
    st.header("🏆 Model Performance & Comparative Insights Engine")
    st.markdown("""
    Welcome to the **Model Performance Engine**. Selecting the optimal forecasting model is a critical step in leveraging 
    data science for business value. This page provides a comprehensive and interactive platform to analyse and compare 
    the performance of various models trained on the Walmart sales data.

    **Understanding Key Performance Metrics:**
    We primarily focus on three standard regression metrics to evaluate forecast accuracy:
    - **MAE (Mean Absolute Error):** Represents the average magnitude of errors in a set of predictions, without considering their direction. It's an average of the absolute differences between prediction and actual observation where all individual differences have equal weight. *Lower is better.*
    - **RMSE (Root Mean Squared Error):** This is the square root of the average of squared differences between prediction and actual observation. RMSE gives a relatively high weight to large errors, meaning it's more sensitive to outliers. *Lower is better.*
    - **MAPE (Mean Absolute Percentage Error):** Expresses accuracy as a percentage of the error. Because it's a percentage, it can be easier to interpret but can be skewed by small denominators or zero actual values. *Lower is better.*

    The insights derived here guide model selection, hyperparameter tuning, and ultimately, the deployment of the most effective forecasting solutions.
    """)

    with st.spinner("Loading and processing all experiment logs... This may take a moment for large datasets."):
        df_logs = _load_and_process_all_logs()

    if df_logs is None or df_logs.empty:
        # Error message handled by _load_and_process_all_logs if it returns None
        return

    st.success(f"Successfully loaded and processed **{len(df_logs)}** individual experiment runs for comparative performance analysis.")

    if st.checkbox("Show Combined and Processed Experiment Logs Dataframe", False, key="show_combined_logs_perf_detailed"):
        st.markdown("""
        This dataframe contains all loaded experiment runs, processed with standardised column names, 
        parsed parameters, and an assigned 'Model_Family'. It serves as the foundation for all subsequent analyses on this page. 
        Exploring this raw log can be useful for detailed ad-hoc investigations or debugging.
        """)
        st.dataframe(df_logs)
        st.caption(f"Available columns in the combined log include: {df_logs.columns.tolist()}")

    st.markdown("---")
    # --- Summary Tables ---
    st.subheader("1. Aggregated Performance Summary Tables")
    st.markdown("""
    Summary tables provide a quick, quantitative overview of model performance, aggregated at different levels. 
    This allows for rapid identification of generally strong or weak performing model categories and specific configurations.
    """)

    st.markdown("#### 1.1. Average Performance Metrics by Model Family")
    st.markdown("""
    This table presents the average MAE, RMSE, and MAPE for each broad model family (e.g., SARIMA, Prophet, Random Forest, LSTM). 
    **Value:** It offers a high-level understanding of which general modelling approaches (time series, regression, deep learning) 
    tend to perform better on average for this specific forecasting problem. This can guide initial model selection in future, similar tasks.
    Models are sorted by RMSE by default.
    """)
    if 'Model_Family' in df_logs.columns and not df_logs[['MAE', 'RMSE', 'MAPE']].isnull().all().all(): # Check if all metric columns are entirely NaN
        avg_metrics_family = df_logs.groupby('Model_Family')[['MAE', 'RMSE', 'MAPE']].mean().sort_values(by='RMSE')
        st.dataframe(avg_metrics_family.style.format("{:.3f}").highlight_min(axis=0, color='lightgreen'))
    else:
        st.info("Insufficient data or missing MAE, RMSE, MAPE columns to calculate average metrics by model family.")

    st.markdown("#### 1.2. Average Performance Metrics by Specific Model Configuration")
    st.markdown("""
    Here, performance metrics are averaged for each unique model name, which often represents a specific configuration or set of hyperparameters.
    **Value:** This allows for a more granular comparison, pinpointing the exact model configurations that yielded superior or inferior results. 
    It's instrumental for understanding the impact of tuning within a model family. Sorted by RMSE.
    """)
    if 'Model' in df_logs.columns and not df_logs[['MAE', 'RMSE', 'MAPE']].isnull().all().all():
        avg_metrics_model = df_logs.groupby('Model')[['MAE', 'RMSE', 'MAPE']].mean().sort_values(by='RMSE')
        st.dataframe(avg_metrics_model.style.format("{:.3f}").highlight_min(axis=0, color='lightgreen'))
    else:
        st.info("Insufficient data or missing metric columns to calculate average metrics by specific model name.")

    st.markdown("#### 1.3. Top Performing Model for Each Store-Department (Based on Lowest RMSE)")
    st.markdown("""
    Recognising that a single global model may not be optimal for all individual sales series (due to unique local patterns), 
    this analysis identifies the specific model configuration that achieved the lowest RMSE for each unique Store-Department combination. 
    Only models trained on specific Store-Department series are included here (global models, where Store/Dept might be -1 or 0, are excluded).
    **Value:** This insight is critical for designing a deployment strategy that might involve a portfolio of specialised models rather 
    than a one-size-fits-all approach, potentially yielding significantly better aggregate forecast accuracy in a real-world scenario.
    RMSE is chosen as the primary ranking metric here due to its sensitivity to large, costly errors.
    """)
    if all(col in df_logs.columns for col in ['Store', 'Dept', 'RMSE', 'Model']):
        df_to_rank = df_logs.dropna(subset=['Store', 'Dept', 'RMSE', 'Model'])
        # Exclude global models (Store/Dept are placeholders like -1) from this store-department specific ranking
        df_specific_store_dept_models = df_to_rank[(df_to_rank['Store'] > 0) & (df_to_rank['Dept'] > 0)]

        if not df_specific_store_dept_models.empty:
            try:
                # idxmin finds the index of the minimum RMSE for each group
                best_models_idx = df_specific_store_dept_models.groupby(['Store', 'Dept'])['RMSE'].idxmin()
                best_models_summary = df_specific_store_dept_models.loc[best_models_idx].copy() # Use .copy() to avoid SettingWithCopyWarning

                cols_to_display = ['Store', 'Dept', 'Model', 'Model_Family', 'RMSE', 'MAE', 'MAPE']
                # Filter to columns that actually exist in the summary
                cols_to_display = [col for col in cols_to_display if col in best_models_summary.columns]

                if not best_models_summary.empty:
                    # Ensure metrics are numeric before applying style format
                    for metric_col_style in ['RMSE', 'MAE', 'MAPE']:
                        if metric_col_style in best_models_summary.columns:
                            best_models_summary[metric_col_style] = pd.to_numeric(best_models_summary[metric_col_style], errors='coerce')
                    
                    st.dataframe(
                        best_models_summary[cols_to_display].sort_values(by=['Store', 'Dept'])
                        .style.format({'RMSE': "{:.2f}", 'MAE': "{:.2f}", 'MAPE': "{:.2f}%"})
                    )
                    if 'Model_Family' in best_models_summary.columns:
                        st.markdown("##### Dominance of Model Families (Store-Department Specific Wins)")
                        st.markdown("""
                        This count summarises which model families most frequently produce the best results (lowest RMSE) 
                        at the granular Store-Department level, indicating the overall robustness or suitability of certain 
                        algorithmic approaches for specific sales series characteristics within this dataset.
                        """)
                        st.dataframe(best_models_summary['Model_Family'].value_counts())
                else:
                    st.info("No top-performing models found after filtering for store-specific results.")
            except KeyError as e_key: 
                st.error(f"A KeyError occurred during model ranking (likely a missing column after an operation): {e_key}. Please check log processing and available columns.")
            except Exception as e_rank:
                st.error(f"An unexpected error occurred during model ranking: {e_rank}")
        else:
            st.info("Insufficient data (excluding global models) with Store, Dept, and RMSE to rank models per Store-Department.")
    else:
        st.info("Required columns for ranking (Store, Dept, RMSE, Model) are not all present in the logs.")

    st.markdown("---")
    # --- Visualizations ---
    st.subheader("2. Visual Performance Comparisons")
    st.markdown("""
    Visualisations offer an intuitive way to compare model performance, highlighting not just average scores but also the
    distribution and consistency of their performance across different runs or series.
    """)
    plt.style.use('seaborn-v0_8-whitegrid') # Consistent style for plots

    metric_plot_config = [
        {'metric': 'RMSE', 'palette': 'viridis', 'title_suffix': '(Lower is Better)'},
        {'metric': 'MAPE', 'palette': 'plasma', 'title_suffix': '(Lower is Better, Expressed as %)'}
    ]

    for config in metric_plot_config:
        metric = config['metric']
        if 'Model_Family' in df_logs.columns and not df_logs[metric].isnull().all(): # Check if metric column is not all NaN
            st.markdown(f"#### 2.1. Bar Chart: Average {metric} by Model Family")
            st.markdown(f"""
            This bar chart visually contrasts the average **{metric}** across different model families. 
            **Interpretation:** Shorter bars indicate better average performance for that metric. This allows for a quick visual grasp of 
            which model categories are, on average, leading in terms of predictive accuracy for this dataset. 
            The suffix `{config["title_suffix"]}` reminds us of the desired direction for this metric.
            """)
            avg_metric_family_data = df_logs.groupby('Model_Family')[metric].mean().dropna().sort_values()
            if not avg_metric_family_data.empty:
                fig_bar, ax_bar = plt.subplots(figsize=(12, 7)) 
                avg_metric_family_data.plot(kind='bar', ax=ax_bar, color=sns.color_palette(config['palette'], len(avg_metric_family_data)))
                ax_bar.set_title(f'Average {metric} by Model Family {config["title_suffix"]}', fontsize=15)
                ax_bar.set_ylabel(f'Average {metric}', fontsize=13)
                ax_bar.set_xlabel('Model Family', fontsize=13)
                plt.xticks(rotation=45, ha='right', fontsize=11)
                plt.yticks(fontsize=11)
                ax_bar.grid(axis='y', linestyle='--', alpha=0.7) 
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar)
            else:
                st.info(f"No data to plot for Average {metric} by Model Family (all values might be NaN or empty after grouping).")

            st.markdown(f"#### 2.2. Box Plot: Distribution of {metric} by Model Family")
            st.markdown(f"""
            Beyond simple averages, these box plots reveal the **spread, central tendency, and consistency** of **{metric}** for each model family 
            across all individual experiment runs (e.g., for different stores/departments or hyperparameter sets).
            **Interpretation:**
            - The **line inside the box** represents the median {metric}.
            - The **box itself** spans the interquartile range (IQR: 25th to 75th percentile), indicating where the middle 50% of the metric values lie.
            - The **whiskers** extend to show the range of the data, excluding outliers (which are plotted as individual points).
            A compact box with a low median is generally desirable. This visualisation helps assess not only which models perform well on average, 
            but also which are more consistently reliable across diverse conditions.
            """)
            metric_boxplot_data = df_logs.dropna(subset=[metric, 'Model_Family'])
            plot_title_box = f'Distribution of {metric} by Model Family {config["title_suffix"]}'
            
            # Initialise plot_data_boxplot to ensure it's defined
            plot_data_boxplot = pd.DataFrame()

            if metric == 'MAPE': 
                if not metric_boxplot_data.empty:
                    mape_values = metric_boxplot_data[metric].dropna()
                    if not mape_values.empty:
                        mape_upper_cap = mape_values.quantile(0.98) 
                        mape_upper_cap = max(mape_upper_cap, 100) if pd.notnull(mape_upper_cap) and mape_upper_cap > 0 else 100 # Ensure cap is reasonable
                        
                        plot_data_boxplot = metric_boxplot_data[metric_boxplot_data[metric] <= mape_upper_cap].copy()
                        plot_title_box = f'Distribution of {metric} by Model Family (MAPE values capped at {mape_upper_cap:.0f}%)'
                    else: # All MAPE values were NaN
                        plot_data_boxplot = pd.DataFrame() # Assign empty dataframe
                # If metric_boxplot_data was empty initially, plot_data_boxplot remains empty
            else: # For RMSE or other metrics
                plot_data_boxplot = metric_boxplot_data.copy()


            if not plot_data_boxplot.empty:
                order_boxplot = plot_data_boxplot.groupby('Model_Family')[metric].median().sort_values().index
                fig_box, ax_box = plt.subplots(figsize=(14, 8)) 
                sns.boxplot(data=plot_data_boxplot, x='Model_Family', y=metric, ax=ax_box,
                            order=order_boxplot, hue='Model_Family', legend=False, 
                            palette=config['palette'], linewidth=1.5, fliersize=3) 
                ax_box.set_title(plot_title_box, fontsize=15)
                ax_box.set_ylabel(metric, fontsize=13)
                ax_box.set_xlabel('Model Family', fontsize=13)
                plt.xticks(rotation=45, ha='right', fontsize=11)
                plt.yticks(fontsize=11)
                ax_box.grid(axis='y', linestyle='--', alpha=0.7) 
                plt.tight_layout()
                st.pyplot(fig_box)
                plt.close(fig_box)
            else:
                st.info(f"Not enough valid data to generate the {metric} distribution box plot by Model Family (data might be empty or all NaN after processing).")
        else:
            st.warning(f"Cannot generate {metric} plots: 'Model_Family' column or '{metric}' data is missing or contains all NaN values.")

    st.markdown("---")
    st.info(
        "**Further Exploration:** Consider using the 'Forecast Explorer' page to delve into individual model predictions "
        "for specific Store-Department combinations, especially for models identified as top performers here. This allows for qualitative assessment alongside these quantitative metrics.",
        icon="💡"
    )

if __name__ == "__main__": # pragma: no cover
    # This block is for standalone testing of this page.
    # Adjust CWD if running directly from 'page_content' to find 'reports'
    if os.path.basename(os.getcwd()) == 'page_content':
        os.chdir(os.path.join("..", "..")) 

    # Redefine PROJECT_ROOT and EXPERIMENT_LOGS_PATH for standalone context
    _PROJECT_ROOT_MODEL_PERF = os.getcwd()
    EXPERIMENT_LOGS_PATH = os.path.join(_PROJECT_ROOT_MODEL_PERF, 'reports', 'experiment_logs')
    
    # Page configuration would typically be in main_app.py
    st.set_page_config(layout="wide", page_title="SupplyChainAI - Model Performance")
    render_model_performance_page()