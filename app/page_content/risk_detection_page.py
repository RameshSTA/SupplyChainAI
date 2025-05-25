"""
Renders the Proactive Supply Chain Risk Detection & Alerting page.

This page demonstrates a conceptual approach to risk detection by processing
sample news data. It applies sentiment analysis, predicts risk categories using
a pre-trained model (if available), and calculates a dynamic AI-driven risk score.
Users can filter and sort these risk signals for review.
"""
import streamlit as st
import pandas as pd
import os
import sys
import ast  # For safely evaluating string representations if used (not directly in current load logic)
import re
import joblib # For loading the pre-trained risk classification model

# --- Project Path Setup & Custom Module Imports ---
_PROJECT_ROOT_RISK_PAGE = None  # Module-level variable for project root
_NLP_UTILS_LOADED = False       # Flag to track if NLP utilities are loaded

def _get_project_root_for_risk_page() -> str:
    """
    Determines the project root directory for this risk detection page.

    Assumes this script is located within: `PROJECT_ROOT/app/page_content/`.
    Navigates up two directories from this file's current location.
    This is primarily for locating data files and NLP utility modules.

    Returns:
        str: The absolute path to the project root directory.
    """
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # app/page_content -> app -> PROJECT_ROOT
        project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
        return project_root
    except NameError:  # pragma: no cover
        st.warning(
            "Could not automatically determine project root using `__file__`. "
            "Falling back to current working directory. Paths to data or utilities might be incorrect."
        )
        return os.getcwd()

_PROJECT_ROOT_RISK_PAGE = _get_project_root_for_risk_page()

# Attempt to import NLP utility functions
try:
    SRC_PATH_RISK = os.path.join(_PROJECT_ROOT_RISK_PAGE, 'src')
    if SRC_PATH_RISK not in sys.path:
        sys.path.insert(0, SRC_PATH_RISK)
    if _PROJECT_ROOT_RISK_PAGE not in sys.path: # Ensure project root itself is available
        sys.path.insert(0, _PROJECT_ROOT_RISK_PAGE)

    from utils.nlp_utils import get_sentiment, preprocess_text
    _NLP_UTILS_LOADED = True
except ImportError as e_import:  # pragma: no cover
    st.error(
        f"Error importing NLP utilities for Risk Detection page: {e_import}. "
        "Ensure 'src/utils/nlp_utils.py' exists and necessary dependencies (like VADER) are installed. "
        "Sentiment analysis and AI category prediction might be affected."
    )
    def get_sentiment(text: str) -> tuple[str, float]:
        """Dummy sentiment function if nlp_utils fails to load."""
        st.error("Sentiment analysis function (get_sentiment) is unavailable due to import error."); return "Error", 0.0
    def preprocess_text(text: str) -> str:
        """Dummy text preprocessing function if nlp_utils fails to load."""
        st.error("Text preprocessing function (preprocess_text) is unavailable due to import error."); return text
except Exception as e_path:  # pragma: no cover
    st.error(f"Unexpected error during sys.path setup or NLP utility imports: {e_path}")
    def get_sentiment(text: str) -> tuple[str, float]:
        """Dummy sentiment function due to path error."""
        st.error("Sentiment analysis function (get_sentiment) is unavailable due to path error."); return "Error", 0.0
    def preprocess_text(text: str) -> str:
        """Dummy text preprocessing function due to path error."""
        st.error("Text preprocessing function (preprocess_text) is unavailable due to path error."); return text


# --- Configuration Paths ---
SAMPLE_NEWS_DATA_PATH = os.path.join(_PROJECT_ROOT_RISK_PAGE, 'data', 'raw', 'sample_risk_news.csv')
MODEL_STORE_PATH_RISK_PAGE = os.path.join(_PROJECT_ROOT_RISK_PAGE, 'models_store', 'risk_detection')

@st.cache_data
def _load_and_analyze_sample_news(file_path: str = SAMPLE_NEWS_DATA_PATH) -> pd.DataFrame | None:
    """
    Loads sample news data from a CSV file and performs initial analysis.

    This includes converting dates, reading manual risk scores, and applying
    VADER sentiment analysis to news summaries or headlines if NLP utilities are loaded.

    Args:
        file_path: The path to the sample news CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the news data with added sentiment
                      and manual risk score columns, or None if loading fails.
    """
    if not os.path.exists(file_path):
        st.error(f"Sample news data file not found at expected location: {file_path}")
        st.info(f"Please ensure 'sample_risk_news.csv' is in the '{os.path.join(_PROJECT_ROOT_RISK_PAGE, 'data', 'raw')}' directory.")
        return None
    try:
        df = pd.read_csv(file_path)
        if 'date_published' in df.columns:
            df['date_published'] = pd.to_datetime(df['date_published'], errors='coerce')

        # Standardize manual risk score column
        df['manual_risk_score'] = pd.to_numeric(df.get('risk_score'), errors='coerce').fillna(0).astype(int)

        sentiments_col, scores_col = [], []
        if _NLP_UTILS_LOADED and callable(get_sentiment):
            # Apply sentiment analysis row-wise to combined headline/summary
            for text_content in df.apply(lambda row: str(row.get('summary', str(row.get('headline', '')))).strip(), axis=1):
                if text_content: # Only process non-empty text
                    label, score = get_sentiment(text_content)
                    sentiments_col.append(label)
                    scores_col.append(score)
                else:
                    sentiments_col.append("Neutral") # Default for empty text
                    scores_col.append(0.0)
            df['predicted_sentiment_vader'] = sentiments_col
            df['sentiment_score_vader'] = scores_col
        else:
            st.info("NLP utilities (get_sentiment) not loaded. VADER sentiment analysis will be skipped.")
            df['predicted_sentiment_vader'] = "N/A (NLP Util Error)"
            df['sentiment_score_vader'] = 0.0
        return df
    except Exception as e:
        st.error(f"Error loading or analyzing sample news data from '{file_path}': {e}")
        return None

@st.cache_resource
def _load_risk_classification_pipeline(model_dir: str = MODEL_STORE_PATH_RISK_PAGE) -> object | None:
    """
    Loads a saved risk classification pipeline (e.g., TF-IDF + Classifier).

    It looks for model files starting with "risk_category_classifier_" and
    ending with ".joblib" in the specified directory, loading the most
    recently named one (lexicographically).

    Args:
        model_dir: Directory where the risk classification model is stored.

    Returns:
        The loaded scikit-learn pipeline object, or None if no model is found or loading fails.
    """
    if not os.path.exists(model_dir):
        st.info(f"Model directory not found: '{model_dir}'. AI category prediction will use fallback logic.")
        return None
    try:
        # Find model files, assuming a naming convention like 'risk_category_classifier_YYYYMMDD_HHMMSS.joblib'
        model_files = [f for f in os.listdir(model_dir)
                       if f.startswith("risk_category_classifier_") and f.endswith(".joblib")]
        if not model_files:
            st.info(f"No risk classification model file (e.g., 'risk_category_classifier_*.joblib') found in '{model_dir}'. AI category prediction will use fallback.")
            return None

        latest_model_file = sorted(model_files, reverse=True)[0] # Get the "latest" by name
        model_path = os.path.join(model_dir, latest_model_file)
        pipeline = joblib.load(model_path)
        st.success(f"Risk classification model pipeline loaded successfully from: {latest_model_file}")
        return pipeline
    except Exception as e:
        st.error(f"Error loading risk classification model from '{model_dir}': {e}")
        return None

def _calculate_ai_risk_score(predicted_category: str | None, vader_sentiment_label: str | None) -> int:
    """
    Calculates a dynamic AI-driven risk score based on predicted category and sentiment.

    The score is primarily determined by the risk category, with adjustments
    based on VADER sentiment analysis.

    Args:
        predicted_category: The AI-predicted risk category for a news item.
        vader_sentiment_label: The sentiment label ('Positive', 'Negative', 'Neutral')
                               from VADER analysis.

    Returns:
        int: A risk score between 0 and 10 (inclusive).
    """
    base_score = 0
    # Predefined scores for various risk categories
    category_scores = {
        "Natural Disaster": 8, "Pandemic": 9, "Supplier Financial": 8,
        "Industrial Accident": 7, "Geopolitical": 7, "Cybersecurity": 7,
        "Logistics": 6, "Labor Dispute": 6, "Material Shortage": 6,
        "Supplier Quality": 6, "Regulatory": 5, "Market": 4,
        "Labor Dispute (Resolved)": 2, "Supplier Positive": 1, # Lower scores for resolved/positive
        "Neutral": 2, "Error Predicting": 3, "N/A (No text/Error)": 0, "Model N/A": 0
    }
    # Assign base score from category, with a fallback for unknown categories
    base_score = category_scores.get(predicted_category, 3 if predicted_category else 0)

    # Adjust score based on sentiment
    if vader_sentiment_label == "Negative":
        base_score = min(10, base_score + 2) # Increase risk for negative sentiment, cap at 10
    elif vader_sentiment_label == "Positive":
        # Only reduce score if it's not an inherently positive or neutral category
        if predicted_category not in ["Supplier Positive", "Labor Dispute (Resolved)", "Neutral"]:
            base_score = max(0, base_score - 1) # Decrease risk for positive sentiment, floor at 0

    return max(0, min(10, base_score)) # Ensure score is within 0-10 range

def _highlight_text_html(text_to_highlight: str, search_query: str, color: str = "rgba(255, 223, 186, 0.7)") -> str:
    """
    Highlights occurrences of a search query within a text string using HTML mark tags.

    Args:
        text_to_highlight: The text in which to search and highlight.
        search_query: The term to search for (case-insensitive).
        color: The background color for the highlight (CSS format).

    Returns:
        str: The text with matching terms highlighted, or the original text if
             no query is provided or text is not a string.
    """
    if not search_query or not isinstance(text_to_highlight, str):
        return text_to_highlight
    try:
        # Use re.sub for case-insensitive replacement and HTML marking
        return re.sub(
            f"({re.escape(search_query)})",
            rf"<mark style='background-color:{color}; padding: 0.1em 0.2em; border-radius: 3px; font-weight: 500;'>\1</mark>",
            text_to_highlight,
            flags=re.IGNORECASE
        )
    except Exception: # pragma: no cover
        return text_to_highlight # Fallback in case of regex error

def _get_sentiment_emoji(sentiment_label: str | None) -> str:
    """
    Returns an emoji corresponding to a sentiment label.

    Args:
        sentiment_label: The sentiment label ('Positive', 'Negative', 'Neutral', 'Error').

    Returns:
        str: An emoji representing the sentiment, or a question mark for unknown labels.
    """
    if sentiment_label == "Positive": return "🟢"    # Green circle for Positive
    elif sentiment_label == "Negative": return "🔴"  # Red circle for Negative
    elif sentiment_label == "Neutral": return "⚪"   # White circle for Neutral
    elif sentiment_label == "Error": return "⚠️"   # Warning sign for Error
    return "❔"  # Question mark for N/A or other

def render_risk_detection_page():
    """
    Renders the Proactive Supply Chain Risk Detection & Alerting page.

    This involves loading sample news data, applying NLP for sentiment and
    category prediction, calculating AI risk scores, and providing UI elements
    for filtering, sorting, and displaying these risk signals.
    """
    st.header("🚨 Proactive Supply Chain Risk Detection & Alerting")
    st.markdown(
        "This module demonstrates processing news-like data to identify potential supply chain risks, "
        "categorize them using AI (if a model is available), assess sentiment, "
        "and generate a dynamic, AI-enhanced risk score."
    )

    with st.expander("Business Impact & Value Proposition", expanded=False): # Default to collapsed
        st.markdown("""
        In today's volatile global landscape, supply chains are more vulnerable than ever.
        Proactively identifying and assessing potential risks allows businesses to:
        * **Minimize Disruptions & Ensure Continuity:** Get early warnings of events like port congestion, supplier issues, or natural disasters.
        * **Reduce Operational Costs:** Avoid expensive last-minute fixes, expedited shipping, or production halts.
        * **Protect Revenue & Market Share:** Prevent stockouts and maintain customer trust by ensuring product availability.
        * **Enhance Resilience & Agility:** Develop contingency plans and adapt more quickly to changing conditions.
        * **Improve Strategic Decision-Making:** Leverage data-driven risk insights for sourcing, inventory, and network design.
        * **Maintain Brand Reputation:** Consistently meet customer expectations and contractual obligations.

        This demonstration utilizes a sample news feed. A production-grade system would integrate diverse,
        real-time data sources (e.g., news APIs, social media, weather alerts, shipping data) and
        employ more sophisticated AI/NLP models for comprehensive, automated risk assessment and alerting.
        """)
    st.markdown("---")

    news_df_analyzed = _load_and_analyze_sample_news()
    risk_classifier_pipeline = _load_risk_classification_pipeline()

    if news_df_analyzed is None or news_df_analyzed.empty:
        st.warning("Could not load or process sample news data. Risk detection demonstration cannot proceed fully.")
        return

    # --- Predict Risk Category and Calculate AI Risk Score ---
    ai_predicted_categories = []
    ai_risk_scores = []

    for index, row in news_df_analyzed.iterrows():
        current_ai_category = "Model N/A" # Default if no model or prediction error
        text_for_prediction = str(row.get('summary', str(row.get('headline', '')))).strip()

        if risk_classifier_pipeline is not None and _NLP_UTILS_LOADED and callable(preprocess_text) and text_for_prediction:
            processed_text = preprocess_text(text_for_prediction)
            if processed_text: # Ensure text remains after preprocessing
                try:
                    current_ai_category = risk_classifier_pipeline.predict([processed_text])[0]
                except Exception: # pragma: no cover
                    current_ai_category = "Error Predicting"
            else:
                current_ai_category = "N/A (No text after preprocess)"
        elif not text_for_prediction :
             current_ai_category = "N/A (No text input)"


        ai_predicted_categories.append(current_ai_category)
        vader_sentiment_label = row.get('predicted_sentiment_vader', "Neutral")
        current_ai_score = _calculate_ai_risk_score(current_ai_category, vader_sentiment_label)
        ai_risk_scores.append(current_ai_score)

    news_df_analyzed['ai_predicted_category'] = ai_predicted_categories
    news_df_analyzed['ai_risk_score'] = ai_risk_scores


    st.subheader("Monitored Risk Signals")
    st.markdown("#### Filter Risk Signals")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 1.5, 1.5, 1.5]) # Adjusted column ratios

    with filter_col1:
        search_term = st.text_input(
            "Search Text (Headline/Summary/Keywords):", key="risk_search_term",
            placeholder="e.g., 'port congestion', 'fire', 'delay'"
        )
    with filter_col2:
        # Use AI predicted categories if available, otherwise manual, or combine if sensible
        category_options_col = 'ai_predicted_category' if 'ai_predicted_category' in news_df_analyzed.columns else 'risk_category'
        unique_categories = ["All"] + sorted(news_df_analyzed[category_options_col].dropna().unique().tolist())
        category_filter_label = f"{'AI Predicted' if category_options_col == 'ai_predicted_category' else 'Manual'} Category:"
        selected_category_filter = st.selectbox(category_filter_label, unique_categories, key="risk_category_filter")

    with filter_col3:
        sentiment_filter_options = ["All", "Positive", "Neutral", "Negative", "Error", "N/A (NLP Util Error)"]
        selected_sentiment_filter = st.selectbox(
            "VADER Sentiment:", sentiment_filter_options, key="risk_sentiment_filter"
        )
    with filter_col4:
        min_score_val, max_score_val = 0, 10 # Default range for risk scores
        score_column_for_filter = 'ai_risk_score' # Prioritize AI risk score
        if score_column_for_filter in news_df_analyzed.columns and not news_df_analyzed[score_column_for_filter].empty:
            min_score_val_data = int(news_df_analyzed[score_column_for_filter].min())
            max_score_val_data = int(news_df_analyzed[score_column_for_filter].max())
            # Ensure slider range covers data range but still within 0-10 if data is sparse
            min_score_val = min(min_score_val_data, min_score_val)
            max_score_val = max(max_score_val_data, max_score_val)

        selected_risk_score_range = st.slider(
            "Filter by AI Risk Score:",
            min_value=0, max_value=10, # Keep slider fixed to 0-10 for consistency
            value=(min_score_val, max_score_val), # Default value range from data
            key="risk_score_range_filter"
        )
    st.caption("Sentiment by VADER. AI Category by trained model (if loaded). AI Score combines AI Category & Sentiment.")

    # --- Apply Filters ---
    filtered_df = news_df_analyzed.copy()
    if search_term:
        search_term_lower = search_term.lower()
        filtered_df = filtered_df[filtered_df.apply(
            lambda r: search_term_lower in str(r.get('headline','')).lower() or \
                      search_term_lower in str(r.get('summary','')).lower() or \
                      search_term_lower in str(r.get('keywords','')).lower(), axis=1
        )]
    if selected_category_filter != "All":
        filtered_df = filtered_df[filtered_df[category_options_col] == selected_category_filter]
    if selected_sentiment_filter != "All" and 'predicted_sentiment_vader' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['predicted_sentiment_vader'] == selected_sentiment_filter]
    if 'ai_risk_score' in filtered_df.columns: # Always filter by AI risk score if available
        filtered_df = filtered_df[
            (filtered_df['ai_risk_score'] >= selected_risk_score_range[0]) &
            (filtered_df['ai_risk_score'] <= selected_risk_score_range[1])
        ]

    if filtered_df.empty:
        st.info("No news items match your current filter criteria.")
    else:
        st.write(f"Displaying **{len(filtered_df)}** matching risk signals:")
        # Define sort options, ensuring columns exist before offering them
        sort_options_dict = {"Date & AI Risk Score (Desc)": (['date_published', 'ai_risk_score'], [False, False])}
        if 'ai_risk_score' in filtered_df.columns:
             sort_options_dict["AI Risk Score (Desc)"] = (['ai_risk_score'], [False])
        if 'manual_risk_score' in filtered_df.columns:
             sort_options_dict["Date & Manual Risk Score (Desc)"] = (['date_published', 'manual_risk_score'], [False, False])
             sort_options_dict["Manual Risk Score (Desc)"] = (['manual_risk_score'], [False])
        if 'date_published' in filtered_df.columns:
             sort_options_dict["Date (Desc)"] = (['date_published'], [False])
        if 'ai_predicted_category' in filtered_df.columns:
             sort_options_dict["AI Predicted Category"] = (['ai_predicted_category'], [True])
        if 'predicted_sentiment_vader' in filtered_df.columns:
             sort_options_dict["VADER Sentiment"] = (['predicted_sentiment_vader'], [True])

        if sort_options_dict: # Proceed only if there's at least one valid sort option
            selected_sort_key = st.selectbox(
                "Sort by:", list(sort_options_dict.keys()), key="risk_sort_option"
            )
            if selected_sort_key and selected_sort_key in sort_options_dict:
                sort_cols, ascending_orders = sort_options_dict[selected_sort_key]
                # Ensure all columns for sorting actually exist in the DataFrame
                if all(col in filtered_df.columns for col in sort_cols):
                    try:
                        filtered_df = filtered_df.sort_values(by=sort_cols, ascending=ascending_orders)
                    except Exception as e_sort: # pragma: no cover
                        st.warning(f"Could not sort by '{selected_sort_key}': {e_sort}. Displaying unsorted.")
                else: # pragma: no cover
                     st.warning(f"One or more sort columns for '{selected_sort_key}' not found. Displaying unsorted.")
        else: # pragma: no cover
            st.info("No sort options available due to missing columns in data.")


        # Display filtered and sorted news items
        for index, row in filtered_df.iterrows():
            st.markdown("---")
            headline_str = str(row.get('headline', 'No Headline Available'))
            summary_str = str(row.get('summary', 'No summary available.'))
            keywords_str = str(row.get('keywords', 'N/A'))

            # Apply highlighting if search term is active
            display_headline = _highlight_text_html(headline_str, search_term)
            display_summary = _highlight_text_html(summary_str, search_term)
            display_keywords = _highlight_text_html(keywords_str, search_term) if keywords_str != 'N/A' else 'N/A'

            sentiment_label = row.get('predicted_sentiment_vader', "N/A")
            sentiment_score_val = row.get('sentiment_score_vader')
            sentiment_emoji_char = _get_sentiment_emoji(sentiment_label)
            sentiment_display = f"{sentiment_emoji_char} Sentiment: **{sentiment_label}**"
            if isinstance(sentiment_score_val, float):
                sentiment_display += f" (Score: {sentiment_score_val:.2f})"

            st.markdown(f"#### {display_headline}", unsafe_allow_html=True)
            display_col_left, display_col_right = st.columns([3, 1]) # Content on left, metric on right

            with display_col_left:
                st.markdown(f"**Summary:** {display_summary}", unsafe_allow_html=True)
                date_published_str = pd.to_datetime(row.get('date_published')).strftime('%Y-%m-%d') \
                    if pd.notna(row.get('date_published')) else 'N/A'
                st.caption(
                    f"Source: {row.get('source', 'N/A')} | "
                    f"Published: {date_published_str} | "
                    f"Risk ID: {row.get('id', 'N/A')}"
                )
                st.markdown(sentiment_display)
                st.markdown(
                    f"AI Predicted Category: **{row.get('ai_predicted_category', 'N/A')}** "
                    f"(Manually Labeled: *{row.get('risk_category', 'N/A')}*)"
                )
            with display_col_right:
                ai_score = row.get('ai_risk_score', 0)
                delta_txt, delta_clr = "", "normal"
                if ai_score >= 7: delta_txt, delta_clr = "High AI Risk", "inverse"
                elif ai_score >= 4: delta_txt, delta_clr = "Medium AI Risk", "off"
                else: delta_txt = "Low AI Risk"
                st.metric(label="AI Risk Score", value=str(ai_score),
                          delta=delta_txt, delta_color=delta_clr)
                st.caption(f"Manual Score: {row.get('manual_risk_score', 0)}")

            if keywords_str != 'N/A': # Only show if keywords exist
                st.markdown(f"**Keywords:** `{display_keywords}`", unsafe_allow_html=True)
            st.write("") # Adds a little vertical space

    st.markdown("---")
    st.caption(
        "Future enhancements could include: real-time news feed integration, more advanced NLP models for nuanced "
        "risk scoring & categorization, geographical map visualizations of risk hotspots, and alert notifications."
    )

if __name__ == "__main__":  # pragma: no cover
    # This block is for standalone testing.
    # It attempts to set the CWD to project root if run from `page_content` for relative paths.
    if os.path.basename(os.getcwd()) == 'page_content':
        os.chdir(os.path.join("..", "..")) # Navigate to project root

    _PROJECT_ROOT_RISK_PAGE = os.getcwd() # Re-affirm for standalone context
    SAMPLE_NEWS_DATA_PATH = os.path.join(_PROJECT_ROOT_RISK_PAGE, 'data', 'raw', 'sample_risk_news.csv')
    MODEL_STORE_PATH_RISK_PAGE = os.path.join(_PROJECT_ROOT_RISK_PAGE, 'models_store', 'risk_detection')

    # st.set_page_config(layout="wide", page_title="Risk Detection") # Usually in main_app.py
    if not _NLP_UTILS_LOADED:
        st.warning("Standalone run: NLP utilities failed to load. Page functionality will be limited.")
    render_risk_detection_page()