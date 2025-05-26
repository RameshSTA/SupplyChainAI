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

# Assuming load_and_merge_walmart_data is in the path set by the main app
try:
    from data_processing.load_walmart_data import load_and_merge_walmart_data
except ImportError:
    st.error(
        "Critical Error: Could not import 'load_and_merge_walmart_data'. "
        "Ensure 'src' directory is correctly added to sys.path by the main application."
    )
    # Define a dummy function if import fails, to allow the app to partially load and show the error.
    def load_and_merge_walmart_data(*args, **kwargs):
        st.error("Placeholder: `load_and_merge_walmart_data` is unavailable. Data exploration cannot proceed.")
        return None

def _get_project_root_for_data_page() -> str:
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
        return project_root
    except NameError:
        st.warning("Could not determine project root using `__file__`. Falling back to CWD.")
        return os.getcwd()

def _handle_markdown_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    df_copy = df.copy()
    for col in markdown_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(0)
    return df_copy

def _handle_cpi_unemployment_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.sort_values(by=['Store', 'Date']).copy()
    for col_name in ['CPI', 'Unemployment']:
        if col_name in df_copy.columns:
            df_copy[col_name] = df_copy.groupby('Store', group_keys=False)[col_name].apply(lambda x: x.ffill().bfill())
            if df_copy[col_name].isnull().any():
                overall_median = df_copy[col_name].median()
                df_copy[col_name] = df_copy[col_name].fillna(overall_median)
    return df_copy

@st.cache_data # Cache the output of this function for efficiency
def get_cleaned_data_for_eda(project_root_path: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    raw_data_dir = os.path.join(project_root_path, 'data', 'raw')
    df_raw_merged = load_and_merge_walmart_data(data_path=raw_data_dir)

    if df_raw_merged is None:
        # Error already displayed by dummy or actual function if it fails
        return None, None

    df_temp_markdown_filled = _handle_markdown_missing_values(df_raw_merged)
    df_cleaned_for_eda = _handle_cpi_unemployment_missing_values(df_temp_markdown_filled)
    
    # Convert 'Date' column to datetime if it's not already, crucial for time series
    if 'Date' in df_cleaned_for_eda.columns:
        df_cleaned_for_eda['Date'] = pd.to_datetime(df_cleaned_for_eda['Date'])

    return df_raw_merged, df_cleaned_for_eda


def render_data_exploration_page():
    """
    Renders the main content for the Data Exploration and Insights page
    with enhanced professional explanations and articulated details.
    """
    st.header("📊 Data Insights & Exploratory Analysis Engine")
    st.markdown("""
    Welcome to the **Data Insights Engine**. Exploratory Data Analysis (EDA) is a cornerstone of any data science project. 
    It involves critically investigating data to discover patterns, spot anomalies, test hypotheses, and check assumptions 
    with the help of summary statistics and graphical representations. 
    
    This interactive page provides a comprehensive EDA of the Walmart sales dataset, revealing its underlying structure, 
    key statistical properties, data quality assessments, and insightful visualisations. Understanding these aspects is
    paramount for effective feature engineering and robust model development.
    """)

    project_root = _get_project_root_for_data_page()

    st.subheader("1. Initial Data Acquisition & Integrity Check")
    st.markdown("""
    The first step involves loading and merging the raw Walmart datasets (sales, stores, features). 
    Initial cleaning is performed to handle obvious inconsistencies and prepare the data for in-depth exploration.
    """)
    df_raw, df_cleaned = None, None
    with st.spinner("Loading, merging, and performing initial cleaning on Walmart datasets..."):
        df_raw, df_cleaned = get_cleaned_data_for_eda(project_root_path=project_root)

    if df_cleaned is None:
        st.error("Critical: Failed to load or clean data. Data exploration cannot proceed. "
                 "Please check data paths and the integrity of the `load_and_merge_walmart_data` function.")
        return

    st.success("Walmart datasets loaded, merged, and initially cleaned successfully!")

    if st.checkbox("Show raw merged data sample (first 100 rows) for initial inspection", False, key="show_raw_sample_eda_detailed"):
        if df_raw is not None:
            st.markdown("Displaying a sample of the raw merged data *before* EDA-specific cleaning:")
            st.dataframe(df_raw.head(100))
        else:
            st.warning("Raw merged data is unavailable for display.")

    st.markdown("---")
    st.subheader("2. Overview of Cleaned Dataset for Analysis")
    st.markdown(f"""
    The following analyses and visualisations are based on the cleaned dataset.
    **Shape of the cleaned data:** `{df_cleaned.shape[0]}` rows and `{df_cleaned.shape[1]}` columns. 
    This provides a snapshot of the dataset's scale.
    """)

    st.markdown("#### 2.1. DataFrame Structure: Column Datatypes & Non-Null Values")
    st.markdown("""
    Understanding the data type of each column (e.g., numerical, categorical, datetime) and the count of non-null values 
    is fundamental. It helps identify potential data quality issues and informs preprocessing strategies.
    """)
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
    df_info_summary = df_info_summary[
        ["Column", "Non-Null Count", "Null Count", "Null Percentage (%)", "Dtype"]
    ]
    st.dataframe(df_info_summary, use_container_width=True)
    memory_usage_mb = df_cleaned.memory_usage(deep=True).sum() / (1024 * 1024)
    st.caption(f"Total memory footprint of the cleaned DataFrame: {memory_usage_mb:.2f} MB. This is relevant for performance considerations in larger datasets.")

    st.markdown("#### 2.2. Descriptive Statistics: Summarising Feature Distributions")
    st.markdown("""
    Descriptive statistics provide a quantitative summary of the dataset's features. For numerical features, this includes 
    measures like mean, median, standard deviation, and quartiles, offering insights into central tendency, spread, and potential outliers. 
    For categorical features, it includes counts and frequencies of unique values.
    """)
    st.markdown("##### Numerical Features Summary")
    try:
        st.dataframe(df_cleaned.describe(include=np.number).T)
    except Exception as e:
        st.warning(f"Could not display numerical descriptive statistics: {e}")

    st.markdown("##### Categorical & Datetime Features Summary")
    try:
        # Ensure 'Date' is treated as datetime for describe if it exists
        include_types = ['object']
        if 'Date' in df_cleaned.columns and pd.api.types.is_datetime64_any_dtype(df_cleaned['Date']):
            include_types.append('datetime64[ns]')
        st.dataframe(df_cleaned.describe(include=include_types).T)
    except Exception as e:
        st.warning(f"Could not display categorical/datetime descriptive statistics: {e}")
    
    st.markdown("---")
    st.subheader("3. Detailed Missing Value Assessment")
    st.markdown("""
    After initial automated cleaning (like filling MarkDown NaNs with 0 and addressing CPI/Unemployment), 
    we perform a final check for any residual missing values. This ensures the data used for subsequent 
    visualisations and modelling is robust.
    """)
    final_missing_summary = df_info_summary[df_info_summary["Null Count"] > 0][
        ["Column", "Null Count", "Null Percentage (%)"]
    ]
    if not final_missing_summary.empty:
        st.warning("Remaining missing values identified after initial cleaning stage:")
        st.dataframe(final_missing_summary)
        st.markdown("""
        *Presence of missing values can significantly bias analyses and model performance. 
        Further imputation or feature engineering strategies might be required depending on the nature 
        and extent of these missing data points for specific modelling tasks.*
        """)
    else:
        st.success("No critical missing values remain in the cleaned dataset used for the EDA plots below. This enhances the reliability of subsequent insights.")

    st.markdown("---")
    # --- Target Variable Analysis ---
    st.subheader("4. In-Depth Analysis of Target Variable: `Weekly_Sales`")
    st.markdown("""
    Understanding the distribution and characteristics of the target variable (`Weekly_Sales`) is crucial. 
    It forms the basis for predictive modelling and helps in identifying potential data transformations 
    or specialized modelling approaches that might be necessary.
    """)
    fig_sales_dist, ax_sales_dist = plt.subplots(figsize=(10, 6))
    sns.histplot(df_cleaned['Weekly_Sales'], kde=True, bins=100, ax=ax_sales_dist)
    ax_sales_dist.set_title('Distribution of Weekly Sales', fontsize=15)
    ax_sales_dist.set_xlabel('Weekly Sales Amount', fontsize=12)
    ax_sales_dist.set_ylabel('Frequency', fontsize=12)
    st.pyplot(fig_sales_dist)
    plt.close(fig_sales_dist)
    
    skewness = df_cleaned['Weekly_Sales'].skew()
    kurtosis = df_cleaned['Weekly_Sales'].kurtosis()
    st.markdown(f"""
    **Statistical Insights:**
    - **Skewness:** `{skewness:.2f}`. A positive skewness indicates a distribution with a longer tail on the right side, meaning a higher frequency of lower sales values and some instances of very high sales (potential outliers or peak periods).
    - **Kurtosis:** `{kurtosis:.2f}`. This value indicates the "tailedness" of the distribution. A high kurtosis suggests more extreme outliers than a normal distribution.
    
    *These characteristics might influence model selection; for instance, some models are sensitive to skewed data or outliers.*
    """)
    neg_sales_count = len(df_cleaned[df_cleaned['Weekly_Sales'] < 0])
    if neg_sales_count > 0:
        st.markdown(
            f"**Note on Negative Sales:** Observed `{neg_sales_count}` instances of negative `Weekly_Sales` "
            f"({(neg_sales_count/len(df_cleaned)*100):.2f}% of total). These typically represent customer returns or corrections "
            "and require careful consideration during data preprocessing for forecasting tasks."
        )
    else:
        st.markdown("**Data Quality Note:** No instances of negative `Weekly_Sales` were found in the cleaned dataset.")

    st.markdown("---")
    # --- Univariate Analysis ---
    st.subheader("5. Univariate Analysis: Understanding Individual Feature Characteristics")
    st.markdown("""
    Univariate analysis focuses on understanding the distribution, central tendency, and spread of individual features. 
    This helps in identifying outliers, data patterns, and informs feature transformation or selection processes.
    """)
    numerical_features_options = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
    valid_numerical_features = [f for f in numerical_features_options if f in df_cleaned.columns]

    if valid_numerical_features:
        st.markdown("#### 5.1. Numerical Features")
        num_feat_select = st.selectbox(
            "Select a Numerical Feature to Analyse its Distribution:",
            valid_numerical_features,
            key="eda_num_uni_select_detailed"
        )
        if num_feat_select:
            fig_num, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(16, 6))
            sns.histplot(df_cleaned[num_feat_select].dropna(), kde=True, ax=ax_h, bins=50)
            ax_h.set_title(f'Distribution of {num_feat_select}', fontsize=14)
            ax_h.set_xlabel(num_feat_select, fontsize=12)
            ax_h.set_ylabel('Frequency', fontsize=12)
            
            sns.boxplot(x=df_cleaned[num_feat_select].dropna(), ax=ax_b)
            ax_b.set_title(f'Box Plot of {num_feat_select}', fontsize=14)
            ax_b.set_xlabel(num_feat_select, fontsize=12)
            st.pyplot(fig_num)
            plt.close(fig_num)
            st.markdown(f"""
            **Interpreting {num_feat_select}:**
            - The **histogram and KDE plot** illustrate the shape of the data's distribution (e.g., normal, skewed, bimodal).
            - The **box plot** visually summarises key statistics: median (central line), interquartile range (IQR - the box), and potential outliers (points beyond the whiskers).
            - Understanding these characteristics for `{num_feat_select}` (e.g., typical range, presence of outliers) is vital for assessing its potential impact on sales and for appropriate feature scaling or transformation during model building.
            """)
    else:
        st.info("No standard numerical features (Temperature, Fuel_Price, etc.) found for univariate analysis.")

    categorical_features_options = ['Type', 'IsHoliday']
    valid_categorical_features = [f for f in categorical_features_options if f in df_cleaned.columns]

    if valid_categorical_features:
        st.markdown("#### 5.2. Categorical Features")
        cat_feat_select = st.selectbox(
            "Select a Categorical Feature to Analyse its Distribution:",
            valid_categorical_features,
            key="eda_cat_uni_select_detailed"
        )
        if cat_feat_select:
            fig_cat, ax_cat = plt.subplots(figsize=(8, 6))
            sns.countplot(
                data=df_cleaned,
                x=cat_feat_select,
                ax=ax_cat,
                palette='viridis',
                order=df_cleaned[cat_feat_select].value_counts().index
            )
            ax_cat.set_title(f'Frequency Distribution of {cat_feat_select}', fontsize=14)
            ax_cat.set_xlabel(cat_feat_select, fontsize=12)
            ax_cat.set_ylabel('Count', fontsize=12)
            # Add percentage labels on bars
            total = len(df_cleaned[cat_feat_select])
            for p in ax_cat.patches:
                percentage = f'{100 * p.get_height() / total:.1f}%'
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                ax_cat.annotate(percentage, (x, y), ha='center', va='bottom')
            st.pyplot(fig_cat)
            plt.close(fig_cat)
            st.markdown(f"""
            **Interpreting {cat_feat_select}:**
            - The **count plot** shows the frequency of each category within the `{cat_feat_select}` feature.
            - This helps understand the balance or imbalance between categories, which can be important for model training (e.g., handling imbalanced classes) and interpreting results. For instance, understanding the proportion of holiday weeks versus non-holiday weeks is crucial for assessing the impact of `IsHoliday`.
            """)
    else:
        st.info("No standard categorical features (Type, IsHoliday) found for univariate analysis.")

    st.markdown("---")
    # --- Bivariate Analysis ---
    st.subheader("6. Bivariate Analysis: Exploring Relationships with `Weekly_Sales`")
    st.markdown("""
    Bivariate analysis examines the relationship between two variables. Here, we primarily focus on how different features 
    correlate with or influence `Weekly_Sales`. Identifying strong relationships can highlight important predictors for our sales forecasting models.
    """)
    if valid_numerical_features:
        st.markdown("#### 6.1. Numerical Features vs. `Weekly_Sales`")
        num_feat_bivar_select = st.selectbox(
            "Select Numerical Feature for Bivariate Analysis against Weekly_Sales:",
            valid_numerical_features,
            key="eda_num_bivar_select_detailed"
        )
        if num_feat_bivar_select:
            fig_bnum, ax_bnum = plt.subplots(figsize=(10, 6))
            sample_df = df_cleaned.sample(n=min(5000, len(df_cleaned)), random_state=42) if len(df_cleaned) > 5000 else df_cleaned
            sns.regplot(
                data=sample_df, x=num_feat_bivar_select, y='Weekly_Sales', ax=ax_bnum,
                scatter_kws={'alpha':0.2, 's':15}, line_kws={'color':'red', 'lw':2}
            )
            ax_bnum.set_title(f'Relationship between {num_feat_bivar_select} and Weekly_Sales', fontsize=14)
            ax_bnum.set_xlabel(num_feat_bivar_select, fontsize=12)
            ax_bnum.set_ylabel('Weekly Sales', fontsize=12)
            st.pyplot(fig_bnum)
            plt.close(fig_bnum)
            
            if pd.api.types.is_numeric_dtype(df_cleaned[num_feat_bivar_select]) and pd.api.types.is_numeric_dtype(df_cleaned['Weekly_Sales']):
                correlation = df_cleaned[num_feat_bivar_select].corr(df_cleaned['Weekly_Sales'])
                st.markdown(f"""
                **Insight for {num_feat_bivar_select} vs. Weekly_Sales:**
                - The scatter plot with a regression line provides a visual indication of the relationship's direction (positive/negative) and strength.
                - **Pearson Correlation Coefficient:** `{correlation:.3f}`. 
                - A coefficient close to +1 or -1 indicates a strong linear relationship, while a value close to 0 suggests a weak or no linear relationship. This feature's correlation is `{abs(correlation):.3f}`, which can be interpreted as {'strong' if abs(correlation) > 0.5 else 'moderate' if abs(correlation) > 0.2 else 'weak'}. 
                Features with stronger correlations are often more directly influential in predictive linear models.
                """)
    else:
        st.info("No standard numerical features available for bivariate analysis vs. Sales.")

    if valid_categorical_features:
        st.markdown("#### 6.2. Categorical Features vs. `Weekly_Sales`")
        cat_feat_bivar_select = st.selectbox(
            "Select Categorical Feature for Bivariate Analysis against Weekly_Sales:",
            valid_categorical_features,
            key="eda_cat_bivar_select_detailed"
        )
        if cat_feat_bivar_select:
            fig_bcat, ax_bcat = plt.subplots(figsize=(10, 7))
            sns.boxplot(data=df_cleaned, x=cat_feat_bivar_select, y='Weekly_Sales', ax=ax_bcat, palette='viridis')
            ax_bcat.set_title(f'Distribution of Weekly_Sales across {cat_feat_bivar_select} Categories', fontsize=14)
            ax_bcat.set_xlabel(cat_feat_bivar_select, fontsize=12)
            ax_bcat.set_ylabel('Weekly Sales', fontsize=12)
            st.pyplot(fig_bcat)
            plt.close(fig_bcat)
            st.markdown(f"""
            **Insight for {cat_feat_bivar_select} vs. Weekly_Sales:**
            - The box plots display the distribution of `Weekly_Sales` for each category of `{cat_feat_bivar_select}`.
            - Comparing the medians (lines within boxes), interquartile ranges (height of boxes), and whisker lengths across categories can reveal if certain categories are associated with significantly different sales levels or variability. 
            - For example, this can show if sales differ significantly by store 'Type' or during 'IsHoliday' periods, making these important features for modelling.
            """)
    else:
        st.info("No standard categorical features available for bivariate analysis vs. Sales.")

    st.markdown("---")
    # --- Time Series Specific EDA ---
    st.subheader("7. Time Series Analysis: Unveiling Temporal Sales Patterns")
    st.markdown("""
    Time series analysis is crucial for sales data as it often exhibits trends, seasonality, and other time-dependent patterns. 
    Understanding these dynamics is key to building accurate forecasting models.
    """)
    
    if 'Date' in df_cleaned.columns and 'Weekly_Sales' in df_cleaned.columns:
        st.markdown("#### 7.1. Overall Weekly Sales Trend (Aggregated)")
        st.markdown("""
        This plot shows the total `Weekly_Sales` aggregated across all stores and departments over time. 
        It provides a high-level view of overall business performance, long-term growth or decline, and significant market-wide seasonal peaks.
        """)
        overall_sales_ts = df_cleaned.groupby('Date')['Weekly_Sales'].sum()
        if not overall_sales_ts.empty:
            fig_ts_overall, ax_ts_overall = plt.subplots(figsize=(14, 7))
            ax_ts_overall.plot(overall_sales_ts.index, overall_sales_ts.values, linewidth=2)
            ax_ts_overall.set_title('Overall Aggregated Weekly Sales Trend', fontsize=15)
            ax_ts_overall.set_xlabel('Date', fontsize=12)
            ax_ts_overall.set_ylabel('Total Weekly Sales', fontsize=12)
            ax_ts_overall.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig_ts_overall)
            plt.close(fig_ts_overall)

            st.markdown("#### 7.2. Seasonal Decomposition of Overall Sales")
            st.markdown("""
            Decomposing the aggregated time series allows us to isolate and understand distinct underlying patterns:
            - **Trend:** The long-term progression of sales (e.g., increasing, decreasing, or stable).
            - **Seasonality:** Regular, predictable fluctuations that occur within a one-year period (e.g., holiday peaks, summer lulls).
            - **Residuals:** The random, irregular component remaining after trend and seasonality are removed.
            This decomposition is invaluable for selecting and configuring appropriate time series forecasting models.
            """)
            # Ensure data is a Series with DatetimeIndex and regular frequency
            overall_sales_series_for_decomp = overall_sales_ts.asfreq('W-FRI', method='pad') # Pad to handle potential missing weeks for decomposition
            overall_sales_series_for_decomp = overall_sales_series_for_decomp.interpolate(method='time') # Interpolate any remaining NaNs

            if not overall_sales_series_for_decomp.empty and len(overall_sales_series_for_decomp.dropna()) >= 2 * 52: # Need at least 2 full periods (years)
                try:
                    decomposition = seasonal_decompose(overall_sales_series_for_decomp.dropna(), model='additive', period=52) # Yearly seasonality
                    fig_decomp = decomposition.plot()
                    fig_decomp.set_size_inches((14, 10))
                    fig_decomp.suptitle('Seasonal Decomposition of Overall Weekly Sales', y=1.02, fontsize=16)
                    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
                    st.pyplot(fig_decomp)
                    plt.close(fig_decomp)
                except Exception as e:
                    st.warning(f"Could not perform seasonal decomposition on overall sales. Error: {e}. This may be due to insufficient data points or irregular frequency after processing.")
            else:
                st.warning("Overall sales series is too short or has too many NaNs for seasonal decomposition (requires at least 2 years of consistent weekly data).")
        else:
            st.info("No data available for overall weekly sales trend analysis.")
    else:
        st.warning("'Date' or 'Weekly_Sales' column not found, cannot perform primary time series analysis.")

    st.markdown("#### 7.3. Granular Sales Trend: Specific Store & Department")
    st.markdown("""
    Analysing sales at a more granular level (specific Store-Department combinations) can reveal micro-market trends, 
    departmental performance nuances, and distinct seasonality patterns that might be obscured in aggregated data. 
    This level of detail is often critical for operational forecasting and inventory planning.
    """)
    col_store, col_dept = st.columns(2)
    unique_stores_eda = sorted(df_cleaned['Store'].unique()) if 'Store' in df_cleaned.columns else []

    if not unique_stores_eda:
        st.info("No 'Store' data available for specific trend analysis.")
    else:
        selected_store_eda = col_store.selectbox(
            "Select Store for Granular Time Series Analysis:", unique_stores_eda, index=0, key="eda_store_select_ts_detailed"
        )
        selected_dept_eda = None
        if selected_store_eda and 'Dept' in df_cleaned.columns:
            available_depts_eda = sorted(df_cleaned[df_cleaned['Store'] == selected_store_eda]['Dept'].unique())
            if available_depts_eda:
                selected_dept_eda = col_dept.selectbox(
                    "Select Department for Granular Time Series Analysis:", available_depts_eda, index=0, key="eda_dept_select_ts_detailed"
                )
            else:
                col_dept.info(f"No departments found for Store {selected_store_eda}.")

        if selected_store_eda and selected_dept_eda:
            store_dept_data = df_cleaned[
                (df_cleaned['Store'] == selected_store_eda) & (df_cleaned['Dept'] == selected_dept_eda)
            ]
            if not store_dept_data.empty and 'Date' in store_dept_data.columns:
                store_dept_ts = store_dept_data.set_index('Date')['Weekly_Sales'].sort_index().asfreq('W-FRI').interpolate(method='time')
                
                fig_sd, ax_sd = plt.subplots(figsize=(14, 7))
                ax_sd.plot(store_dept_ts.index, store_dept_ts.values, linewidth=2)
                ax_sd.set_title(f'Weekly Sales: Store {selected_store_eda} - Dept {selected_dept_eda}', fontsize=15)
                ax_sd.set_xlabel('Date', fontsize=12)
                ax_sd.set_ylabel('Weekly Sales', fontsize=12)
                ax_sd.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig_sd)
                plt.close(fig_sd)

                st.markdown("""
                **Autocorrelation (ACF) and Partial Autocorrelation (PACF) Plots:**
                These plots are diagnostic tools essential for identifying the underlying structure of a time series and determining appropriate parameters (p, d, q) for ARIMA-family models.
                - The **ACF plot** shows the correlation of the time series with its own lagged values.
                - The **PACF plot** shows the correlation between the series and its lag after removing the effects of any shorter lags.
                Significant spikes in these plots suggest potential autoregressive (AR) or moving average (MA) components.
                """)
                cleaned_ts_sd = store_dept_ts.dropna()
                if len(cleaned_ts_sd) > 30 : # Adjusted minimum length for more reliable ACF/PACF
                    max_lags = min(40, len(cleaned_ts_sd) // 2 - 1) # Common practice for lags
                    if max_lags > 0:
                        fig_acf_pacf, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                        plot_acf(cleaned_ts_sd, lags=max_lags, ax=ax1, zero=False) # zero=False to exclude lag 0
                        ax1.set_title('Autocorrelation Function (ACF)', fontsize=13)
                        plot_pacf(cleaned_ts_sd, lags=max_lags, method='ywm', ax=ax2, zero=False) # yule-walker method
                        ax2.set_title('Partial Autocorrelation Function (PACF)', fontsize=13)
                        plt.tight_layout()
                        st.pyplot(fig_acf_pacf)
                        plt.close(fig_acf_pacf)
                    else:
                        st.caption(f"Time series for Store {selected_store_eda}-Dept {selected_dept_eda} is too short ({len(cleaned_ts_sd)} points) for meaningful ACF/PACF plots (max lags: {max_lags}).")
                else:
                    st.caption(f"Time series for Store {selected_store_eda}-Dept {selected_dept_eda} is too short ({len(cleaned_ts_sd)} points) for ACF/PACF analysis after removing NaNs.")
            elif store_dept_data.empty:
                st.warning(f"No sales data found for the combination: Store {selected_store_eda}, Department {selected_dept_eda}.")
            else: # Should not happen if 'Date' is checked earlier, but as a fallback.
                st.warning(f"Data for Store {selected_store_eda}-Dept {selected_dept_eda} is missing the 'Date' column or is unsuitable for time series plotting.")
        elif selected_store_eda and not selected_dept_eda and (not available_depts_eda or len(available_depts_eda) > 0) : # Store selected, but no dept selected or no depts available
             if available_depts_eda: # Only prompt if depts were available
                col_dept.info("Please select a department to view its specific time series and ACF/PACF plots.")


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="SupplyChainAI - Data Exploration")
    render_data_exploration_page()