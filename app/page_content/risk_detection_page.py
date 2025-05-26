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
import ast
import re
import joblib # For loading the pre-trained risk classification model

# --- Project Path Setup & Custom Module Imports ---
_PROJECT_ROOT_RISK_PAGE = None
_NLP_UTILS_LOADED = False

def _get_project_root_for_risk_page() -> str:
    """
    Determines the project root directory for this risk detection page.

    Assumes this script is located within `PROJECT_ROOT/app/page_content/`.
    Navigates up two directories to establish the project root, crucial for
    locating data files, model artifacts, and NLP utility modules consistently.

    Returns:
        str: The absolute path to the project root directory. Falls back to
             the current working directory if `__file__` is undefined.
    """
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    except NameError:  # pragma: no cover
        st.warning(
            "Could not automatically determine project root using `__file__` (undefined in this context). "
            "Falling back to current working directory. Paths to data, models, or utilities might be incorrect "
            "if this page is run standalone without the project root as CWD."
        )
        return os.getcwd()

_PROJECT_ROOT_RISK_PAGE = _get_project_root_for_risk_page()

# Attempt to import NLP utility functions
try:
    SRC_PATH_RISK = os.path.join(_PROJECT_ROOT_RISK_PAGE, 'src')
    if SRC_PATH_RISK not in sys.path:
        sys.path.insert(0, SRC_PATH_RISK)
    if _PROJECT_ROOT_RISK_PAGE not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT_RISK_PAGE)

    from utils.nlp_utils import get_sentiment, preprocess_text
    _NLP_UTILS_LOADED = True
    print("Risk Detection Page: NLP utilities (get_sentiment, preprocess_text) loaded successfully.")
except ImportError as e_import:  # pragma: no cover
    _NLP_UTILS_LOADED = False # Ensure flag is correctly set on import error
    st.error(
        f"CRITICAL IMPORT ERROR loading NLP utilities for Risk Detection page: {e_import}. "
        "This page relies on 'src/utils/nlp_utils.py' for sentiment analysis and text preprocessing. "
        "Please ensure the file exists, all dependencies (like VADER_Lexicon for NLTK) are correctly installed, "
        "and relevant directories have '__init__.py' files. AI-driven sentiment and category prediction "
        "functionality will be impaired or use fallbacks."
    )
    # Define dummy functions if critical imports failed
    def get_sentiment(text: str) -> tuple[str, float]:
        st.error("NLP Utility Error: Sentiment analysis function (get_sentiment) is unavailable."); return "Error", 0.0
    def preprocess_text(text: str) -> str:
        st.error("NLP Utility Error: Text preprocessing function (preprocess_text) is unavailable."); return text
except Exception as e_path:  # pragma: no cover
    _NLP_UTILS_LOADED = False # Ensure flag is correctly set on other path errors
    st.error(f"Unexpected error during sys.path setup or NLP utility imports for Risk Page: {e_path}")
    def get_sentiment(text: str) -> tuple[str, float]:
        st.error("Path Setup Error: Sentiment analysis function (get_sentiment) is unavailable."); return "Error", 0.0
    def preprocess_text(text: str) -> str:
        st.error("Path Setup Error: Text preprocessing function (preprocess_text) is unavailable."); return text


# --- Configuration Paths ---
SAMPLE_NEWS_DATA_PATH = os.path.join(_PROJECT_ROOT_RISK_PAGE, 'data', 'raw', 'sample_risk_news.csv')
MODEL_STORE_PATH_RISK_PAGE = os.path.join(_PROJECT_ROOT_RISK_PAGE, 'models_store', 'risk_detection')

@st.cache_data(show_spinner="Loading and analysing sample news data...")
def _load_and_analyze_sample_news(file_path: str = SAMPLE_NEWS_DATA_PATH) -> pd.DataFrame | None:
    """
    Loads sample news data from a CSV, performs initial cleaning, and applies
    VADER sentiment analysis if NLP utilities are available.

    The sample data typically includes columns like 'headline', 'summary',
    'date_published', 'source', 'keywords', and optionally a manual 'risk_score'
    or 'risk_category'. This function standardizes these and adds sentiment scores.

    Args:
        file_path: Path to the sample news CSV file.

    Returns:
        A Pandas DataFrame with loaded and initially analyzed news data,
        or None if loading or essential processing fails.
    """
    if not os.path.exists(file_path):
        st.error(f"CRITICAL: Sample news data file ('sample_risk_news.csv') not found at the expected location: '{file_path}'. "
                 "This file is essential for the risk detection demonstration.")
        st.info(f"Please ensure the file is placed in: '{os.path.join(_PROJECT_ROOT_RISK_PAGE, 'data', 'raw')}'")
        return None
    try:
        df = pd.read_csv(file_path)
        if 'date_published' in df.columns:
            df['date_published'] = pd.to_datetime(df['date_published'], errors='coerce')
            if df['date_published'].isnull().any():
                st.warning("Note: Some 'date_published' values in the sample news data could not be parsed to datetime.")

        # Standardize manual risk score column if it exists
        if 'risk_score' in df.columns:
            df['manual_risk_score'] = pd.to_numeric(df.get('risk_score'), errors='coerce').fillna(0).astype(int)
        else:
            df['manual_risk_score'] = 0 # Default if column is missing

        # Prepare for sentiment analysis
        sentiments_col, scores_col = [], []
        # Use 'summary' if available, fallback to 'headline', then ensure it's a string
        texts_to_analyze = df.apply(lambda row: str(row.get('summary', str(row.get('headline', '')))).strip(), axis=1)

        if _NLP_UTILS_LOADED and callable(get_sentiment):
            st.caption("Applying VADER sentiment analysis to news content...")
            for text_content in texts_to_analyze:
                if text_content:
                    label, score = get_sentiment(text_content)
                    sentiments_col.append(label)
                    scores_col.append(score)
                else: # Handle empty text content
                    sentiments_col.append("Neutral")
                    scores_col.append(0.0)
            df['predicted_sentiment_vader'] = sentiments_col
            df['sentiment_score_vader'] = scores_col
        else:
            st.info("NLP utilities (specifically `get_sentiment` from `nlp_utils`) were not loaded successfully. VADER sentiment analysis will be skipped, and related fields will show 'N/A'.")
            df['predicted_sentiment_vader'] = "N/A (NLP Util Error)"
            df['sentiment_score_vader'] = 0.0
        return df
    except Exception as e:
        st.error(f"An error occurred while loading or performing initial analysis on sample news data from '{file_path}': {e}")
        return None

@st.cache_resource(show_spinner="Loading pre-trained risk classification model (if available)...", ttl=3600)
def _load_risk_classification_pipeline(model_dir: str = MODEL_STORE_PATH_RISK_PAGE) -> object | None:
    """
    Loads a pre-trained risk classification pipeline (e.g., TF-IDF Vectorizer + Classifier)
    saved as a .joblib file from the specified model directory.

    It identifies model files following the convention "risk_category_classifier_*.joblib"
    and loads the lexicographically latest one. This model is used to predict a
    risk category (e.g., 'Logistics', 'Natural Disaster') from news text.

    Args:
        model_dir: Directory where the risk classification model artifact is stored.

    Returns:
        The loaded scikit-learn pipeline object if successful, otherwise None.
    """
    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        st.info(f"Risk classification model directory not found: '{model_dir}'. AI-driven category prediction will use a fallback (or 'Model N/A').")
        return None
    try:
        model_files = [f for f in os.listdir(model_dir)
                       if f.startswith("risk_category_classifier_") and f.endswith(".joblib")]
        if not model_files:
            st.info(f"No pre-trained risk classification model file (e.g., 'risk_category_classifier_*.joblib') found in '{model_dir}'. AI category prediction will use fallback.")
            return None

        latest_model_file = sorted(model_files, reverse=True)[0] # Lexicographically latest
        model_path = os.path.join(model_dir, latest_model_file)
        
        if not os.path.exists(model_path): # Should not happen if listdir worked, but defensive
            st.error(f"Model file '{latest_model_file}' listed but not found at path '{model_path}'.")
            return None
            
        pipeline = joblib.load(model_path)
        st.success(f"AI Risk Classification Model Pipeline ('{latest_model_file}') loaded successfully.")
        return pipeline
    except Exception as e:
        st.error(f"Error loading risk classification model from '{model_dir}': {e}. AI category prediction may default.")
        return None

def _calculate_ai_risk_score(predicted_category: str | None, vader_sentiment_label: str | None) -> int:
    """
    Calculates a dynamic, AI-enhanced risk score based on the predicted risk category
    and the sentiment of the news item.

    The scoring logic prioritizes the severity implied by the risk category, then
    adjusts this score based on whether the VADER sentiment analysis indicates
    positive, negative, or neutral sentiment. This provides a nuanced risk level.

    Args:
        predicted_category: The AI-predicted risk category (e.g., "Logistics", "Natural Disaster").
        vader_sentiment_label: The sentiment label ('Positive', 'Negative', 'Neutral') from VADER.

    Returns:
        int: An integer risk score, typically between 0 (lowest risk) and 10 (highest risk).
    """
    base_score = 0
    # Define base scores for risk categories; higher implies more severe intrinsic risk
    category_scores = {
        "Natural Disaster": 8, "Pandemic": 9, "Supplier Financial Instability": 8, "Cybersecurity Attack": 7,
        "Industrial Accident": 7, "Geopolitical Conflict": 7, "Port Congestion": 6, "Logistics Disruption": 6,
        "Labor Dispute": 6, "Material Shortage": 6, "Supplier Quality Issue": 6, "Regulatory Change": 5,
        "Market Volatility": 4, "Labor Dispute (Resolved)": 2, "Positive Supplier News": 1,
        "General News/Neutral": 1, "Error Predicting Category": 3, "N/A (No text/Error)": 0, "Model N/A": 0, # Fallbacks
        # Add more specific categories as trained by your model
    }
    # Assign base score, default to a low-mid score if category is novel or unmapped
    default_score_for_unknown_category = 3
    base_score = category_scores.get(predicted_category, default_score_for_unknown_category if predicted_category else 0)

    # Modulate score based on sentiment
    if vader_sentiment_label == "Negative":
        base_score = min(10, base_score + 2) # Amplify risk for negative sentiment
    elif vader_sentiment_label == "Positive":
        # Reduce risk for positive sentiment, but less so for inherently severe categories
        if base_score > 3 and predicted_category not in ["Positive Supplier News", "Labor Dispute (Resolved)"]: # Don't overly reduce already low/positive risk
            base_score = max(0, base_score - 1)
        elif base_score <=3 and predicted_category not in ["Positive Supplier News", "Labor Dispute (Resolved)"]:
             base_score = max(0, base_score - 0) # minimal reduction for neutral/low base score if not explicitly positive category
        elif predicted_category in ["Positive Supplier News"]:
            base_score = 1 # Ensure explicitly positive categories remain low risk
            
    return max(0, min(10, int(round(base_score)))) # Ensure score is an integer within 0-10

def _highlight_text_html(text_to_highlight: str, search_query: str, color: str = "rgba(255, 223, 186, 0.6)") -> str: # Slightly less opaque
    """Highlights occurrences of a search query within text using HTML <mark> tags for display."""
    if not search_query or not isinstance(text_to_highlight, str) or not text_to_highlight.strip():
        return text_to_highlight # Return original if no query or empty text
    try:
        # Case-insensitive highlighting
        return re.sub(
            f"({re.escape(search_query)})",
            rf"<mark style='background-color:{color}; padding: 0.1em 0.2em; border-radius: 3px; font-weight: bold;'>\1</mark>",
            text_to_highlight,
            flags=re.IGNORECASE
        )
    except Exception: # pragma: no cover (Regex errors are rare with re.escape)
        return text_to_highlight

def _get_sentiment_emoji(sentiment_label: str | None) -> str:
    """Returns a representative emoji for a given sentiment label."""
    if sentiment_label == "Positive": return "🟢 Positive"
    elif sentiment_label == "Negative": return "🔴 Negative"
    elif sentiment_label == "Neutral": return "⚪ Neutral"
    elif sentiment_label == "Error": return "⚠️ Error"
    return "❔ N/A"

def render_risk_detection_page():
    """
    Renders the Proactive Supply Chain Risk Detection & Alerting page.

    This interactive page demonstrates a conceptual framework for identifying and
    assessing potential supply chain risks from textual data (e.g., news articles).
    It involves:
    1. Loading sample news data.
    2. Applying Natural Language Processing (NLP) via VADER for sentiment analysis.
    3. Utilizing a pre-trained machine learning model (if available) to predict
       the risk category of each news item.
    4. Calculating a dynamic, AI-enhanced risk score based on the predicted
       category and sentiment.
    5. Providing a user interface with filtering and sorting capabilities to allow
       users to navigate and prioritise these identified risk signals.

    The primary value is to showcase how AI can augment traditional risk monitoring
    by providing structured, quantified, and categorised risk intelligence from
    unstructured text sources, enabling more proactive and informed decision-making.
    """
    st.header("🚨 Proactive Supply Chain Risk Intelligence Engine")
    st.markdown(
        """
        Welcome to the **Risk Intelligence Engine**. In an increasingly complex and volatile global environment,
        the ability to proactively identify, assess, and mitigate supply chain risks is paramount for business continuity and resilience.
        This demonstration showcases an AI-augmented approach to processing textual information (like news feeds) to
        surface potential disruptions, understand their sentiment, categorize their nature, and assign a dynamic risk score.
        """
    )

    with st.expander("ℹ️ The Strategic Value & Business Impact of Proactive Risk Detection", expanded=False):
        st.markdown("""
        A reactive approach to supply chain disruptions often leads to significant financial losses, operational inefficiencies,
        and damaged customer relationships. Proactively identifying and assessing potential risks allows businesses to transition
        from crisis management to strategic foresight, enabling them to:

        * **Minimize Disruptions & Bolster Continuity:** Gain early warnings of emerging threats (e.g., port congestion, supplier insolvencies, geopolitical tensions, natural disasters), allowing for timely intervention.
        * **Optimize Operational Costs:** Avert costly last-minute corrective actions, such as expedited freight, emergency sourcing, or unplanned production downtime.
        * **Safeguard Revenue & Preserve Market Share:** Prevent stockouts due to unforeseen shortages and maintain customer trust by ensuring consistent product availability and service delivery.
        * **Enhance Organisational Resilience & Agility:** Develop robust contingency plans, diversify sourcing strategies, and adapt more rapidly and effectively to evolving global conditions.
        * **Inform Strategic Decision-Making:** Leverage data-driven risk intelligence for more resilient network design, strategic sourcing, inventory positioning, and capital allocation.
        * **Protect Brand Reputation & Stakeholder Value:** Consistently meet customer commitments and contractual obligations, thereby reinforcing brand image and investor confidence.

        While this page utilizes a sample news dataset for demonstration, a production-grade system would integrate diverse,
        real-time data streams (e.g., global news APIs, social media analytics, weather forecasting services, maritime shipping data, financial market indicators)
        and employ more sophisticated, context-aware AI/NLP models for comprehensive, automated risk assessment, prioritisation, and actionable alerting.
        """)
    st.markdown("---")

    # Load data and model
    st.markdown("#### Data Foundation & AI Model Status")
    st.markdown("This demonstration relies on sample news data and, if available, a pre-trained risk classification model.")
    news_df_analyzed = _load_and_analyze_sample_news()
    risk_classifier_pipeline = _load_risk_classification_pipeline()

    if news_df_analyzed is None or news_df_analyzed.empty:
        st.error("Sample news data could not be loaded. The risk detection demonstration cannot proceed without this core data.")
        return # Stop further execution if no data

    # --- Predict Risk Category and Calculate AI Risk Score ---
    st.markdown("#### AI-Driven Risk Enrichment")
    st.markdown("Applying NLP and machine learning to enrich raw news items with predicted categories and dynamic risk scores.")
    
    with st.spinner("Applying AI analysis (category prediction, risk scoring)..."):
        ai_predicted_categories_list = []
        ai_risk_scores_list = []

        for index, row in news_df_analyzed.iterrows():
            current_ai_category_val = "Model N/A" # Default if no model or error
            text_for_prediction_val = str(row.get('summary', str(row.get('headline', '')))).strip()

            if risk_classifier_pipeline is not None and _NLP_UTILS_LOADED and callable(preprocess_text) and text_for_prediction_val:
                processed_text_val = preprocess_text(text_for_prediction_val)
                if processed_text_val: # Ensure text remains after preprocessing
                    try:
                        current_ai_category_val = risk_classifier_pipeline.predict([processed_text_val])[0]
                    except Exception: # pragma: no cover (Model prediction errors)
                        current_ai_category_val = "Error Predicting Category"
                else: # Text became empty after preprocessing
                    current_ai_category_val = "N/A (No Processable Text)"
            elif not text_for_prediction_val: # Original text was empty
                 current_ai_category_val = "N/A (No Input Text)"
            # If risk_classifier_pipeline is None, current_ai_category_val remains "Model N/A"

            ai_predicted_categories_list.append(current_ai_category_val)
            vader_sentiment_label_val = row.get('predicted_sentiment_vader', "Neutral") # Use VADER sentiment from earlier step
            current_ai_score_val = _calculate_ai_risk_score(current_ai_category_val, vader_sentiment_label_val)
            ai_risk_scores_list.append(current_ai_score_val)

        news_df_analyzed['ai_predicted_category'] = ai_predicted_categories_list
        news_df_analyzed['ai_risk_score'] = ai_risk_scores_list
    st.success("AI analysis complete. News items enriched with predicted categories and AI risk scores.")
    st.caption(
        "**Note on AI Outputs:** Sentiment is determined by VADER. "
        "The 'AI Predicted Category' is generated by the loaded text classification model (if available). "
        "The 'AI Risk Score' is a calculated metric combining the predicted category's inherent severity with the observed sentiment."
    )
    st.markdown("---")

    st.subheader("📡 Monitored Risk Signals Dashboard")
    st.markdown(
        "Interact with the filters below to refine the list of risk signals. "
        "This allows you to focus on specific types of risks, sentiment profiles, or score ranges, "
        "and sort them by relevance or urgency."
    )
    st.markdown("#### Filter & Sort Risk Signals")
    filter_col1_ui, filter_col2_ui, filter_col3_ui, filter_col4_ui = st.columns([2.5, 1.5, 1.5, 1.5]) # Adjusted ratios for better spacing

    with filter_col1_ui:
        search_term_ui = st.text_input(
            "Search Text (Headline/Summary/Keywords):", key="risk_search_term_ui",
            placeholder="e.g., 'port congestion', 'factory fire', 'material delay'",
            help="Enter keywords to search within the text content of news items."
        )
    with filter_col2_ui:
        category_options_col_ui = 'ai_predicted_category' # Prioritize AI-predicted category
        if category_options_col_ui not in news_df_analyzed.columns or news_df_analyzed[category_options_col_ui].nunique() < 2 : # Fallback if AI category is not diverse or missing
            category_options_col_ui = 'risk_category' if 'risk_category' in news_df_analyzed.columns else None

        if category_options_col_ui:
            unique_categories_ui = ["All Categories"] + sorted(news_df_analyzed[category_options_col_ui].dropna().unique().tolist())
            category_filter_label_ui = f"{'AI Predicted' if category_options_col_ui == 'ai_predicted_category' else 'Manually Labeled'} Category:"
            selected_category_filter_ui = st.selectbox(category_filter_label_ui, unique_categories_ui, key="risk_category_filter_ui")
        else:
            selected_category_filter_ui = "All Categories" # Default if no category column
            st.info("No suitable category column found for filtering.")

    with filter_col3_ui:
        sentiment_filter_options_ui = ["All Sentiments", "Positive", "Neutral", "Negative", "Error", "N/A (NLP Util Error)"]
        selected_sentiment_filter_ui = st.selectbox(
            "Filter by VADER Sentiment:", sentiment_filter_options_ui, key="risk_sentiment_filter_ui"
        )
    with filter_col4_ui:
        # Define slider range consistently from 0-10 for AI Risk Score
        min_score_val_filter, max_score_val_filter = 0, 10
        default_slider_val = (min_score_val_filter, max_score_val_filter) # Default to full range

        if 'ai_risk_score' in news_df_analyzed.columns and not news_df_analyzed['ai_risk_score'].empty:
             # Optionally adjust default_slider_val based on actual data range, but keep slider 0-10
             # data_min_score = int(news_df_analyzed['ai_risk_score'].min())
             # data_max_score = int(news_df_analyzed['ai_risk_score'].max())
             # default_slider_val = (data_min_score, data_max_score)
             pass # Keep default 0-10 for slider, user can adjust

        selected_risk_score_range_ui = st.slider(
            "Filter by AI Risk Score (0-10):",
            min_value=0, max_value=10,
            value=default_slider_val, # Default to showing all scores
            key="risk_score_range_filter_ui",
            help="Filter news items based on their calculated AI Risk Score. Higher scores indicate greater potential risk."
        )

    # --- Apply Filters ---
    filtered_df_ui = news_df_analyzed.copy()
    if search_term_ui:
        search_term_lower_ui = search_term_ui.lower()
        # Search in multiple relevant text fields
        text_search_cols = ['headline', 'summary', 'keywords']
        text_search_cols_present = [col for col in text_search_cols if col in filtered_df_ui.columns]
        if text_search_cols_present:
            filtered_df_ui = filtered_df_ui[
                filtered_df_ui[text_search_cols_present].apply(
                    lambda row: any(search_term_lower_ui in str(cell_content).lower() for cell_content in row), axis=1
                )
            ]
    if category_options_col_ui and selected_category_filter_ui != "All Categories":
        filtered_df_ui = filtered_df_ui[filtered_df_ui[category_options_col_ui] == selected_category_filter_ui]
    if selected_sentiment_filter_ui != "All Sentiments" and 'predicted_sentiment_vader' in filtered_df_ui.columns:
        filtered_df_ui = filtered_df_ui[filtered_df_ui['predicted_sentiment_vader'] == selected_sentiment_filter_ui]
    
    # Always filter by AI risk score if column exists
    if 'ai_risk_score' in filtered_df_ui.columns:
        filtered_df_ui = filtered_df_ui[
            (filtered_df_ui['ai_risk_score'] >= selected_risk_score_range_ui[0]) &
            (filtered_df_ui['ai_risk_score'] <= selected_risk_score_range_ui[1])
        ]

    if filtered_df_ui.empty:
        st.info("No news items currently match your specified filter criteria. Try broadening your search.")
    else:
        st.markdown(f"Displaying **{len(filtered_df_ui)}** matching risk signals (news items):")
        
        # Define sort options dynamically based on available columns
        sort_options_dict_ui = {}
        if 'date_published' in filtered_df_ui.columns and 'ai_risk_score' in filtered_df_ui.columns:
            sort_options_dict_ui["Most Recent & Highest AI Risk"] = (['date_published', 'ai_risk_score'], [False, False])
        if 'ai_risk_score' in filtered_df_ui.columns:
             sort_options_dict_ui["Highest AI Risk Score First"] = (['ai_risk_score'], [False])
        if 'date_published' in filtered_df_ui.columns:
             sort_options_dict_ui["Most Recent First"] = (['date_published'], [False])
        if 'manual_risk_score' in filtered_df_ui.columns: # If manual scores are used/relevant
             sort_options_dict_ui["Highest Manual Risk Score First"] = (['manual_risk_score'], [False])
        if 'ai_predicted_category' in filtered_df_ui.columns:
             sort_options_dict_ui["AI Predicted Category (A-Z)"] = (['ai_predicted_category'], [True])
        # Add more sort options as needed (e.g., by VADER sentiment if useful)

        if sort_options_dict_ui:
            selected_sort_key_ui = st.selectbox(
                "Sort Results By:", list(sort_options_dict_ui.keys()), key="risk_sort_option_ui",
                help="Choose the order in which to display the filtered risk signals."
            )
            if selected_sort_key_ui and selected_sort_key_ui in sort_options_dict_ui:
                sort_cols_ui, ascending_orders_ui = sort_options_dict_ui[selected_sort_key_ui]
                if all(col in filtered_df_ui.columns for col in sort_cols_ui): # Ensure sort columns exist
                    try:
                        filtered_df_ui = filtered_df_ui.sort_values(by=sort_cols_ui, ascending=ascending_orders_ui)
                    except Exception as e_sort_ui: # pragma: no cover
                        st.warning(f"Could not apply sorting by '{selected_sort_key_ui}': {e_sort_ui}. Displaying results unsorted or by previous sort.")
                else: # pragma: no cover (should be caught by dict construction)
                     st.warning(f"One or more columns required for sorting by '{selected_sort_key_ui}' are not available. Displaying unsorted.")
        else:
            st.info("No dynamic sort options available due to missing key columns in the filtered data.")

        # Display filtered and sorted news items
        st.markdown("#### Filtered Risk Signals:")
        for index, row_data in filtered_df_ui.iterrows():
            st.markdown("---", unsafe_allow_html=True) # Use markdown for <hr /> for better control
            headline_text = str(row_data.get('headline', 'No Headline Provided'))
            summary_text = str(row_data.get('summary', 'No summary content available.'))
            keywords_text = str(row_data.get('keywords', 'N/A'))

            # Apply HTML highlighting for search term
            display_headline_html = _highlight_text_html(headline_text, search_term_ui)
            display_summary_html = _highlight_text_html(summary_text, search_term_ui)
            display_keywords_html = _highlight_text_html(keywords_text, search_term_ui) if keywords_text != 'N/A' else 'N/A'

            # Sentiment display
            sentiment_label_disp = row_data.get('predicted_sentiment_vader', "N/A")
            sentiment_score_disp = row_data.get('sentiment_score_vader')
            sentiment_emoji_disp = _get_sentiment_emoji(sentiment_label_disp) # Gets emoji + text
            sentiment_full_display = f"{sentiment_emoji_disp}"
            if isinstance(sentiment_score_disp, float): # Add VADER compound score
                sentiment_full_display += f" (Compound: {sentiment_score_disp:.2f})"

            st.markdown(f"##### {display_headline_html}", unsafe_allow_html=True)
            
            # Use columns for better layout of metrics and text
            content_col, metrics_col = st.columns([3, 1]) # Content takes more space

            with content_col:
                st.markdown(f"**Summary:** {display_summary_html}", unsafe_allow_html=True)
                date_published_val = pd.to_datetime(row_data.get('date_published')) if pd.notna(row_data.get('date_published')) else None
                date_display_str = date_published_val.strftime('%B %d, %Y') if date_published_val else 'Date N/A'
                
                st.caption(
                    f"Source: `{row_data.get('source', 'N/A')}` | "
                    f"Published: `{date_display_str}` | "
                    f"Internal ID: `{row_data.get('id', 'N/A')}`"
                )
                st.markdown(f"**Sentiment Assessment (VADER):** {sentiment_full_display}")
                st.markdown(
                    f"**AI Predicted Risk Category:** **`{row_data.get('ai_predicted_category', 'N/A')}`** "
                    f"(Manually Labeled: *`{row_data.get('risk_category', 'N/A')}`*)"
                )
                if keywords_text != 'N/A':
                    st.markdown(f"**Identified Keywords:** `{display_keywords_html}`", unsafe_allow_html=True)

            with metrics_col:
                ai_score_val = row_data.get('ai_risk_score', 0)
                delta_text_val, delta_color_val = "", "normal" # Defaults
                if ai_score_val >= 8: delta_text_val, delta_color_val = "High Risk", "inverse"
                elif ai_score_val >= 5: delta_text_val, delta_color_val = "Medium Risk", "off"
                elif ai_score_val >= 1: delta_text_val = "Low Risk"
                else: delta_text_val = "Minimal/No Risk"

                st.metric(label="AI Risk Score", value=str(int(ai_score_val)), # Ensure int for display
                          delta=delta_text_val, delta_color=delta_color_val,
                          help="Calculated score (0-10) based on AI Category & VADER Sentiment. Higher indicates greater potential risk severity or urgency.")
                st.caption(f"Manual Score (if any): {row_data.get('manual_risk_score', 'N/A')}")
            st.write("") # Small vertical spacer

    st.markdown("---")
    st.subheader("Future Enhancements & Considerations")
    st.markdown(
        """
        This demonstration provides a foundational framework for AI-driven risk detection. For a production-level system,
        several enhancements would be crucial:
        - **Real-time Data Ingestion:** Integration with diverse, continuous data streams (news APIs, social media, financial reports, weather services, shipping data, etc.).
        - **Advanced NLP Models:** Utilisation of more sophisticated NLP techniques, including transformer-based models (e.g., BERT, RoBERTa) for nuanced understanding of context, named entity recognition (identifying specific companies, locations, events), and relationship extraction.
        - **Dynamic & Contextual Risk Scoring:** Developing more complex risk scoring algorithms that consider factors like the source's credibility, the magnitude of potential impact, geographical proximity, and historical data on similar events.
        - **User Feedback & Model Retraining:** Incorporating mechanisms for users to validate or correct AI predictions, with this feedback used to continuously retrain and improve the underlying models (Human-in-the-Loop AI).
        - **Alerting & Workflow Integration:** Customizable alert notifications (email, SMS, dashboard flags) and integration with existing risk management workflows or enterprise systems.
        - **Geospatial Visualisation:** Mapping risk events geographically to identify regional hotspots and potential cascading effects.
        - **Knowledge Graph Integration:** Building a supply chain knowledge graph to better understand interdependencies and assess the ripple effects of specific risks.
        """
    )

if __name__ == "__main__":  # pragma: no cover
    # This block facilitates standalone testing of this page module.
    if os.path.basename(os.getcwd()) == 'page_content':
        os.chdir(os.path.join("..", ".."))

    _PROJECT_ROOT_RISK_PAGE = os.getcwd() # Re-affirm for standalone context
    SRC_PATH_RISK = os.path.join(_PROJECT_ROOT_RISK_PAGE, 'src') # Re-affirm for standalone imports
    if SRC_PATH_RISK not in sys.path: sys.path.insert(0, SRC_PATH_RISK)
    if _PROJECT_ROOT_RISK_PAGE not in sys.path: sys.path.insert(0, _PROJECT_ROOT_RISK_PAGE)
    
    SAMPLE_NEWS_DATA_PATH = os.path.join(_PROJECT_ROOT_RISK_PAGE, 'data', 'raw', 'sample_risk_news.csv')
    MODEL_STORE_PATH_RISK_PAGE = os.path.join(_PROJECT_ROOT_RISK_PAGE, 'models_store', 'risk_detection')

    # For standalone, re-check NLP utils as path might have just been fixed
    _NLP_UTILS_LOADED_STANDALONE = False
    try:
        from utils.nlp_utils import get_sentiment as actual_get_sentiment, preprocess_text as actual_preprocess_text
        get_sentiment = actual_get_sentiment
        preprocess_text = actual_preprocess_text
        _NLP_UTILS_LOADED_STANDALONE = True
        print("Standalone Run: NLP Utilities re-checked and loaded.")
    except ImportError:
        st.error("Standalone Run: Failed to load NLP utilities even after path adjustment. Sentiment/Preprocessing will use dummies.")
        # Dummies are already assigned globally if initial load failed.

    st.set_page_config(layout="wide", page_title="SupplyChainAI - Risk Detection")
    render_risk_detection_page()