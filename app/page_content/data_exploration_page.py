"""
Renders the Data Exploration page for the Streamlit application.

This page provides an interactive interface for Exploratory Data Analysis (EDA)
on the Walmart sales dataset. It includes data loading, initial overviews,
missing value analysis, and various univariate, bivariate, and time series
visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
# sys and io are not directly used in the final version of this file.
# If load_and_merge_walmart_data or other utilities were to require them
# in this context, they would be re-added.

# Attempt to import the data loading function from the 'src' directory.
# It's assumed that main_app.py has configured sys.path correctly.
try:
    from data_processing.load_walmart_data import load_and_merge_walmart_data
except ImportError:
    st.error(
        "Critical Error: Could not import 'load_and_merge_walmart_data'. "
        "Ensure 'src' directory is correctly added to sys.path by the main application "
        "and the file 'src/data_processing/load_walmart_data.py' exists and is valid."
    )
    def load_and_merge_walmart_data(*args, **kwargs) -> None:
        """
        Dummy function for `load_and_merge_walmart_data`.

        This function serves as a placeholder if the actual data loading utility
        cannot be imported. It prevents the Streamlit page from crashing immediately
        and displays an error message.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None: Indicates data loading failure.
        """
        st.error(
            "Placeholder: `load_and_merge_walmart_data` function is not available due to import failure. "
            "Data exploration cannot proceed."
        )
        return None

def _get_project_root_for_data_page() -> str:
    """
    Determines the project root directory, primarily for locating data assets.

    Assumes this script (`data_exploration_page.py`) is located within:
    `PROJECT_ROOT/app/page_content/`.
    It navigates up two directories from this file's location.

    Returns:
        str: The absolute path to the project root.

    Raises:
        NameError: If `__file__` is not defined (e.g., in some non-standard
                   execution environments), which would prevent path determination.
                   A Streamlit warning is also issued in such cases.
    """
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # app/page_content -> app -> PROJECT_ROOT
        project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
        return project_root
    except NameError:
        st.warning(
            "Could not determine project root using `__file__` (it was undefined). "
            "Falling back to current working directory. Data asset paths may be incorrect "
            "if this page is run standalone without proper context."
        )
        return os.getcwd() # Fallback, less reliable for module context

def _handle_markdown_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in 'MarkDown' columns with 0.

    This is a common preprocessing step for these specific columns where NaN
    often implies no markdown was applied.

    Args:
        df: Pandas DataFrame potentially containing MarkDown columns.

    Returns:
        pd.DataFrame: DataFrame with missing MarkDown values filled.
    """
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    df_copy = df.copy()
    for col in markdown_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(0)
    return df_copy

def _handle_cpi_unemployment_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing CPI and Unemployment values using forward/backward fill
    grouped by Store, then by overall median if NaNs persist.

    Args:
        df: Pandas DataFrame with 'CPI' and 'Unemployment' columns.
            Requires 'Store' and 'Date' columns for sorting and grouping.

    Returns:
        pd.DataFrame: DataFrame with missing CPI/Unemployment values filled.
    """
    df_copy = df.sort_values(by=['Store', 'Date']).copy() # Ensure correct order for ffill/bfill
    for col_name in ['CPI', 'Unemployment']:
        if col_name in df_copy.columns:
            # Group by store and apply ffill then bfill
            df_copy[col_name] = df_copy.groupby('Store', group_keys=False)[col_name].apply(lambda x: x.ffill().bfill())
            # If any NaNs remain (e.g., a store has all NaNs), fill with overall median
            if df_copy[col_name].isnull().any():
                overall_median = df_copy[col_name].median()
                df_copy[col_name] = df_copy[col_name].fillna(overall_median)
    return df_copy

@st.cache_data
def get_cleaned_data_for_eda(project_root_path: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Loads, merges, and cleans Walmart sales data for EDA.

    The raw data is loaded from a 'data/raw' subdirectory within the
    provided project_root_path. Cleaning involves handling missing values
    in MarkDown, CPI, and Unemployment columns.

    Args:
        project_root_path: The absolute path to the project root directory.

    Returns:
        tuple[pd.DataFrame | None, pd.DataFrame | None]:
            A tuple containing:
            - df_raw_merged: The initially merged DataFrame (or None if loading fails).
            - df_cleaned_for_eda: The DataFrame after EDA-specific cleaning (or None if processing fails).
    """
    raw_data_dir = os.path.join(project_root_path, 'data', 'raw')
    df_raw_merged = load_and_merge_walmart_data(data_path=raw_data_dir)

    if df_raw_merged is None:
        st.error("Failed to load and merge raw data via `load_and_merge_walmart_data`.")
        return None, None

    df_temp_markdown_filled = _handle_markdown_missing_values(df_raw_merged)
    df_cleaned_for_eda = _handle_cpi_unemployment_missing_values(df_temp_markdown_filled)

    return df_raw_merged, df_cleaned_for_eda


def render_data_exploration_page():
    """
    Renders the main content for the Data Exploration and Insights page.

    This function orchestrates the display of data overviews, missing value
    analyses, and various plots for EDA.
    """
    st.header("📊 Data Insights & Exploratory Analysis")
    st.markdown("""
    This section presents an overview of the Walmart sales dataset, including its structure,
    key statistics, missing value analysis, and visualizations of important features and their
    relationship with weekly sales.
    """)

    project_root = _get_project_root_for_data_page()

    st.subheader("1. Data Loading and Initial Overview")
    df_raw, df_cleaned = None, None
    with st.spinner("Loading, merging, and cleaning Walmart datasets for EDA..."):
        df_raw, df_cleaned = get_cleaned_data_for_eda(project_root_path=project_root)

    if df_cleaned is None:
        st.error("Failed to load or clean data. Cannot proceed with exploration. "
                 "Check data paths and the `load_and_merge_walmart_data` function.")
        return # Stop execution if data isn't available

    st.success("Data loaded, merged, and initially cleaned successfully!")

    if st.checkbox("Show raw merged data sample (first 100 rows)", False, key="show_raw_sample_eda"):
        if df_raw is not None:
            st.dataframe(df_raw.head(100))
        else:
            # This case should ideally not be reached if df_cleaned is not None,
            # but included for robustness.
            st.warning("Raw merged data is unavailable for display.")

    st.subheader("Cleaned DataFrame Overview (Used for EDA Plots Below)")
    st.write(f"**Shape of cleaned data:** {df_cleaned.shape} (Rows, Columns)")

    st.markdown("#### DataFrame Info (Column Types & Non-Null Counts)")
    # Creating a summary similar to df.info()
    info_list = []
    for col in df_cleaned.columns:
        info_list.append({
            "Column": col,
            "Non-Null Count": df_cleaned[col].count(),
            "Dtype": str(df_cleaned[col].dtype)
        })
    df_info_summary = pd.DataFrame(info_list)
    df_info_summary["Null Count"] = len(df_cleaned) - df_info_summary["Non-Null Count"]
    df_info_summary["Null Percentage (%)"] = \
        (df_info_summary["Null Count"] / len(df_cleaned) * 100).round(2)
    # Reorder columns for better readability
    df_info_summary = df_info_summary[
        ["Column", "Non-Null Count", "Null Count", "Null Percentage (%)", "Dtype"]
    ]
    st.dataframe(df_info_summary, use_container_width=True)
    memory_usage_mb = df_cleaned.memory_usage(deep=True).sum() / (1024 * 1024)
    st.caption(f"Total memory usage of cleaned DataFrame: {memory_usage_mb:.2f} MB")

    st.subheader("Descriptive Statistics (Cleaned Data)")
    st.markdown("##### Numerical Features")
    try:
        st.dataframe(df_cleaned.describe(include=np.number).T)
    except Exception as e:
        st.warning(f"Could not display numerical descriptive statistics: {e}")

    st.markdown("##### Categorical & Datetime Features")
    try:
        st.dataframe(df_cleaned.describe(include=['object', 'datetime64[ns]']).T)
    except Exception as e:
        st.warning(f"Could not display categorical/datetime descriptive statistics: {e}")

    st.subheader("2. Final Missing Value Check (Tabular)")
    final_missing_summary = df_info_summary[df_info_summary["Null Count"] > 0][
        ["Column", "Null Count", "Null Percentage (%)"]
    ]
    if not final_missing_summary.empty:
        st.write("Remaining missing values after initial cleaning:")
        st.dataframe(final_missing_summary)
    else:
        st.success("No critical missing values remain in the cleaned dataset used for EDA plots!")

    # --- Target Variable Analysis ---
    st.subheader("3. Target Variable Analysis: `Weekly_Sales`")
    fig_sales_dist, ax_sales_dist = plt.subplots(figsize=(10, 5))
    sns.histplot(df_cleaned['Weekly_Sales'], kde=True, bins=100, ax=ax_sales_dist)
    ax_sales_dist.set_title('Distribution of Weekly Sales')
    # ax_sales_dist.legend() # Legend not typically needed for a single histogram
    st.pyplot(fig_sales_dist)
    plt.close(fig_sales_dist) # Close plot to free memory
    st.caption(
        f"Skewness: {df_cleaned['Weekly_Sales'].skew():.2f}, "
        f"Kurtosis: {df_cleaned['Weekly_Sales'].kurtosis():.2f}"
    )
    neg_sales_count = len(df_cleaned[df_cleaned['Weekly_Sales'] < 0])
    if neg_sales_count > 0:
        st.write(
            f"Instances with negative Weekly_Sales (e.g., returns): {neg_sales_count} "
            f"({(neg_sales_count/len(df_cleaned)*100):.2f}%)"
        )
    else:
        st.write("No instances of negative Weekly_Sales found.")


    # --- Univariate Analysis ---
    st.subheader("4. Univariate Feature Analysis")
    numerical_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
    # Filter to features actually present in the DataFrame
    valid_numerical_features = [f for f in numerical_features if f in df_cleaned.columns]

    if valid_numerical_features:
        num_feat_select = st.selectbox(
            "Select Numerical Feature for Univariate Analysis:",
            valid_numerical_features,
            key="eda_num_uni_select"
        )
        if num_feat_select: # Ensure a selection is made
            fig_num, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(15, 5))
            sns.histplot(df_cleaned[num_feat_select].dropna(), kde=True, ax=ax_h)
            ax_h.set_title(f'Distribution of {num_feat_select}')
            sns.boxplot(x=df_cleaned[num_feat_select].dropna(), ax=ax_b)
            ax_b.set_title(f'Box Plot of {num_feat_select}')
            st.pyplot(fig_num)
            plt.close(fig_num)
    else:
        st.info("No standard numerical features found for univariate analysis.")

    categorical_features = ['Type', 'IsHoliday']
    valid_categorical_features = [f for f in categorical_features if f in df_cleaned.columns]

    if valid_categorical_features:
        cat_feat_select = st.selectbox(
            "Select Categorical Feature for Univariate Analysis:",
            valid_categorical_features,
            key="eda_cat_uni_select"
        )
        if cat_feat_select:
            fig_cat, ax_cat = plt.subplots(figsize=(8, 5))
            sns.countplot(
                data=df_cleaned,
                x=cat_feat_select,
                # hue=cat_feat_select, # Hue is redundant for countplot of a single var
                ax=ax_cat,
                palette='viridis',
                legend=False, # No legend needed for simple countplot
                order=df_cleaned[cat_feat_select].value_counts().index
            )
            ax_cat.set_title(f'Distribution of {cat_feat_select}')
            st.pyplot(fig_cat)
            plt.close(fig_cat)
    else:
        st.info("No standard categorical features found for univariate analysis.")


    # --- Bivariate Analysis ---
    st.subheader("5. Bivariate Analysis: Features vs. `Weekly_Sales`")
    if valid_numerical_features:
        num_feat_bivar_select = st.selectbox(
            "Select Numerical Feature for Bivariate Analysis vs. Sales:",
            valid_numerical_features,
            key="eda_num_bivar_select"
        )
        if num_feat_bivar_select:
            fig_bnum, ax_bnum = plt.subplots(figsize=(10, 6))
            # Sample data for regplot if dataset is large to improve performance
            sample_df = df_cleaned.sample(n=min(5000, len(df_cleaned)), random_state=42) \
                if len(df_cleaned) > 5000 else df_cleaned
            sns.regplot(
                data=sample_df,
                x=num_feat_bivar_select,
                y='Weekly_Sales',
                ax=ax_bnum,
                scatter_kws={'alpha': 0.3, 's': 10},
                line_kws={'color': 'red'}
            )
            ax_bnum.set_title(f'{num_feat_bivar_select} vs. Weekly_Sales')
            st.pyplot(fig_bnum)
            plt.close(fig_bnum)
            if pd.api.types.is_numeric_dtype(df_cleaned[num_feat_bivar_select]) and \
               pd.api.types.is_numeric_dtype(df_cleaned['Weekly_Sales']):
                correlation = df_cleaned[num_feat_bivar_select].corr(df_cleaned['Weekly_Sales'])
                st.write(f"Correlation between {num_feat_bivar_select} and Weekly_Sales: {correlation:.3f}")
    else:
        st.info("No standard numerical features found for bivariate analysis vs. Sales.")


    if valid_categorical_features:
        cat_feat_bivar_select = st.selectbox(
            "Select Categorical Feature for Bivariate Analysis vs. Sales:",
            valid_categorical_features,
            key="eda_cat_bivar_select"
        )
        if cat_feat_bivar_select:
            fig_bcat, ax_bcat = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                data=df_cleaned,
                x=cat_feat_bivar_select,
                y='Weekly_Sales',
                # hue=cat_feat_bivar_select, # Hue often redundant for boxplot by category
                ax=ax_bcat,
                palette='viridis',
                legend=False
            )
            ax_bcat.set_title(f'{cat_feat_bivar_select} vs. Weekly_Sales')
            st.pyplot(fig_bcat)
            plt.close(fig_bcat)
    else:
        st.info("No standard categorical features found for bivariate analysis vs. Sales.")


    # --- Time Series Specific EDA ---
    st.subheader("6. Time Series Analysis of `Weekly_Sales`")
    st.markdown("##### Overall Weekly Sales Trend (Aggregated Across All Stores/Depts)")
    if 'Date' in df_cleaned.columns and 'Weekly_Sales' in df_cleaned.columns:
        overall_sales_ts = df_cleaned.groupby('Date')['Weekly_Sales'].sum()
        if not overall_sales_ts.empty:
            fig_ts_overall, ax_ts_overall = plt.subplots(figsize=(12, 6))
            ax_ts_overall.plot(overall_sales_ts.index, overall_sales_ts.values)
            ax_ts_overall.set_title('Overall Weekly Sales Trend')
            ax_ts_overall.set_xlabel('Date')
            ax_ts_overall.set_ylabel('Total Weekly Sales')
            st.pyplot(fig_ts_overall)
            plt.close(fig_ts_overall)

            st.markdown("##### Seasonal Decomposition of Overall Sales")
            # Ensure data is a Series with DatetimeIndex and regular frequency
            overall_sales_series_for_decomp = overall_sales_ts.asfreq('W-FRI') # Assuming weekly data ending on Friday
            # Interpolate missing values if any after asfreq
            overall_sales_series_for_decomp = overall_sales_series_for_decomp.interpolate(method='time')

            if not overall_sales_series_for_decomp.empty and \
               len(overall_sales_series_for_decomp.dropna()) >= 2 * 52: # Need at least 2 full periods for decomposition
                try:
                    decomposition = seasonal_decompose(
                        overall_sales_series_for_decomp.dropna(),
                        model='additive',
                        period=52 # Assuming yearly seasonality for weekly data
                    )
                    fig_decomp = decomposition.plot()
                    fig_decomp.set_size_inches((14, 10))
                    plt.tight_layout()
                    st.pyplot(fig_decomp)
                    plt.close(fig_decomp)
                except Exception as e:
                    st.warning(f"Could not perform seasonal decomposition on overall sales: {e}")
            else:
                st.warning("Overall sales series is too short or has too many NaNs for seasonal decomposition (requires at least 2 years of data).")
        else:
            st.info("No data available for overall weekly sales trend.")
    else:
        st.warning("'Date' or 'Weekly_Sales' column not found for time series analysis.")


    st.markdown("##### Sales Trend for Specific Store & Department")
    col1, col2 = st.columns(2)
    unique_stores_eda = sorted(df_cleaned['Store'].unique()) if 'Store' in df_cleaned.columns else []

    if not unique_stores_eda:
        st.info("No 'Store' data available for specific trend analysis.")
    else:
        selected_store_eda = col1.selectbox(
            "Select Store:",
            unique_stores_eda,
            index=0,
            key="eda_store_select_ts"
        )
        selected_dept_eda = None
        if selected_store_eda and 'Dept' in df_cleaned.columns:
            available_depts_eda = sorted(
                df_cleaned[df_cleaned['Store'] == selected_store_eda]['Dept'].unique()
            )
            if available_depts_eda:
                selected_dept_eda = col2.selectbox(
                    "Select Department:",
                    available_depts_eda,
                    index=0,
                    key="eda_dept_select_ts"
                )
            else:
                col2.info(f"No departments found for Store {selected_store_eda}.")

        if selected_store_eda and selected_dept_eda:
            store_dept_data = df_cleaned[
                (df_cleaned['Store'] == selected_store_eda) &
                (df_cleaned['Dept'] == selected_dept_eda)
            ]
            if not store_dept_data.empty and 'Date' in store_dept_data.columns:
                store_dept_ts = store_dept_data.set_index('Date')['Weekly_Sales'].sort_index()
                fig_sd, ax_sd = plt.subplots(figsize=(12, 6))
                ax_sd.plot(store_dept_ts.index, store_dept_ts.values)
                ax_sd.set_title(f'Weekly Sales for Store {selected_store_eda} - Department {selected_dept_eda}')
                ax_sd.set_xlabel('Date')
                ax_sd.set_ylabel('Weekly Sales')
                st.pyplot(fig_sd)
                plt.close(fig_sd)

                cleaned_ts_sd = store_dept_ts.dropna()
                # For ACF/PACF, ensure sufficient data points and valid lags
                if len(cleaned_ts_sd) > 20 : # Arbitrary minimum length for meaningful ACF/PACF
                    # Lags should not exceed n_obs // 2 - 1
                    max_lags = min(60, len(cleaned_ts_sd) // 2 - 1)
                    if max_lags > 0:
                        fig_acf_pacf, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
                        plot_acf(cleaned_ts_sd, lags=max_lags, ax=ax1)
                        ax1.set_title('Autocorrelation Function (ACF)')
                        plot_pacf(cleaned_ts_sd, lags=max_lags, method='ywm', ax=ax2) # yule-walker method
                        ax2.set_title('Partial Autocorrelation Function (PACF)')
                        st.pyplot(fig_acf_pacf)
                        plt.close(fig_acf_pacf)
                    else:
                        st.caption(f"Time series for S{selected_store_eda}-D{selected_dept_eda} is too short for ACF/PACF plots after NaNs.")
                else:
                    st.caption(f"Time series for S{selected_store_eda}-D{selected_dept_eda} is too short for ACF/PACF plots.")
            elif store_dept_data.empty:
                st.warning(f"No sales data found for Store {selected_store_eda}, Department {selected_dept_eda}.")
            else:
                st.warning(f"Data for S{selected_store_eda}-D{selected_dept_eda} is missing the 'Date' column.")
        elif selected_store_eda and not selected_dept_eda and available_depts_eda: # Only store selected, but depts were available
             col2.info("Please select a department to view specific time series.")


if __name__ == "__main__":
    # This block allows for standalone testing of this page module.
    # For assets and data to load correctly, ensure this script is run
    # from a context where relevant paths are correctly resolved by
    # _get_project_root_for_data_page() or ensure data is accessible.
    # Page configuration (title, layout) is typically handled by the main app.
    # st.set_page_config(layout="wide")
    render_data_exploration_page()