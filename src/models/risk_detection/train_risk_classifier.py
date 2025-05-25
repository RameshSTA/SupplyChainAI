"""
Trains a text classification model to predict risk categories from news summaries.

This script performs the following steps:
1.  Ensures necessary NLTK resources ('punkt', 'stopwords') are available.
2.  Sets up project paths for data, model storage, and experiment logs.
3.  Defines functions for logging experiment details.
4.  Includes a text preprocessing function (lowercase, remove punctuation, remove stopwords).
5.  The main training function:
    a.  Loads sample news data.
    b.  Preprocesses the text summaries.
    c.  Splits data into training and testing sets.
    d.  Defines and trains a scikit-learn pipeline (TF-IDF Vectorizer + Multinomial Naive Bayes).
    e.  Evaluates the model using accuracy and classification report.
    f.  Saves the trained model pipeline using joblib.
    g.  Logs experiment parameters and metrics.
"""
import pandas as pd
import numpy as np
import os
import sys
import re
import joblib
from datetime import datetime

# NLP and ML libraries
import nltk
# Specific NLTK components will be imported after resource verification.
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression # Imported but not used in provided script
from sklearn.metrics import accuracy_score, classification_report # confusion_matrix not used
from sklearn.pipeline import Pipeline

# Global cache for NLTK stopwords to avoid repeated loading
_STOP_WORDS_ENGLISH_CACHE = None
_NLTK_WORD_TOKENIZER_CACHE = None

def _ensure_nltk_resources() -> bool:
    """
    Checks for and attempts to download NLTK 'punkt' and 'stopwords' if missing.

    Returns:
        bool: True if all required resources are available and verified, False otherwise.
    """
    global _STOP_WORDS_ENGLISH_CACHE, _NLTK_WORD_TOKENIZER_CACHE # Allow modification of globals
    print("--- Verifying NLTK Resources ('punkt' for tokenizer, 'stopwords') ---")
    resources_ok = True

    # Check and download 'punkt' (for word_tokenize)
    try:
        from nltk.tokenize import word_tokenize
        word_tokenize("Test sentence for punkt.") # Verify it works
        _NLTK_WORD_TOKENIZER_CACHE = word_tokenize
        print("NLTK 'punkt' tokenizer resource is available and working.")
    except LookupError:
        print("NLTK 'punkt' resource not found. Attempting download...")
        try:
            nltk.download('punkt', quiet=False)
            from nltk.tokenize import word_tokenize # Re-attempt import
            word_tokenize("Test sentence post-download for punkt.") # Verify again
            _NLTK_WORD_TOKENIZER_CACHE = word_tokenize
            print("NLTK 'punkt' downloaded and verified successfully.")
        except Exception as e_download_punkt:
            print(f"ERROR: Failed to download or verify 'punkt' after attempt: {e_download_punkt}")
            resources_ok = False
    except Exception as e_generic_punkt: # Catch other potential errors during initial check
        print(f"ERROR: An unexpected error occurred while checking 'punkt': {e_generic_punkt}")
        resources_ok = False

    # Check and download 'stopwords'
    try:
        from nltk.corpus import stopwords
        _STOP_WORDS_ENGLISH_CACHE = set(stopwords.words('english')) # Cache for efficiency
        print("NLTK 'stopwords' resource is available.")
    except LookupError:
        print("NLTK 'stopwords' resource not found. Attempting download...")
        try:
            nltk.download('stopwords', quiet=False)
            from nltk.corpus import stopwords # Re-attempt import
            _STOP_WORDS_ENGLISH_CACHE = set(stopwords.words('english')) # Verify and cache
            print("NLTK 'stopwords' downloaded and verified successfully.")
        except Exception as e_download_stopwords:
            print(f"ERROR: Failed to download or verify 'stopwords' after attempt: {e_download_stopwords}")
            resources_ok = False
    except Exception as e_generic_stopwords:
        print(f"ERROR: An unexpected error occurred while checking 'stopwords': {e_generic_stopwords}")
        resources_ok = False

    if not resources_ok:
        error_msg = (
            "One or more essential NLTK resources could not be loaded or downloaded. "
            "Text preprocessing and risk classification cannot proceed effectively. "
            "Please ensure a stable internet connection for downloads, or manually download "
            "these resources by running in a Python console:\n"
            "import nltk\nnltk.download('punkt')\nnltk.download('stopwords')\n"
            "If issues persist, check NLTK data path configurations (nltk.data.path)."
        )
        print(f"\nFATAL NLTK RESOURCE ERROR: {error_msg}\n")
    return resources_ok

# --- Project Path Setup ---
_PROJECT_ROOT_RISK_TRAIN = None # Module-level variable

def _get_project_root_for_risk_training() -> str:
    """
    Determines the project root directory for this risk model training script.

    Assumes this script is located within a nested structure like:
    `PROJECT_ROOT/src/models/risk_detection/train_risk_classifier.py`.
    It navigates up four levels from this file's location. Includes fallbacks for
    interactive environments or different CWDs.

    Returns:
        str: The absolute path to the project root directory.
    """
    try:
        # Assumes file is at: PROJECT_ROOT/src/models/risk_detection/train_risk_classifier.py
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
        return project_root
    except NameError:  # __file__ is not defined
        cwd = os.getcwd()
        # Heuristic fallbacks for common interactive execution paths
        if os.path.exists(os.path.join(cwd, 'data')) and os.path.exists(os.path.join(cwd, 'src')):
            return cwd # CWD is likely project root
        path_elements = os.path.normpath(cwd).split(os.sep)
        if 'risk_detection' in path_elements and 'models' in path_elements and 'src' in path_elements:
            # e.g., CWD is PROJECT_ROOT/src/models/risk_detection
            return os.path.abspath(os.path.join(cwd, "..", "..", ".."))
        if 'models' in path_elements and 'src' in path_elements:
            # e.g., CWD is PROJECT_ROOT/src/models
            return os.path.abspath(os.path.join(cwd, "..", ".."))
        if 'src' in path_elements:
            # e.g., CWD is PROJECT_ROOT/src
            return os.path.abspath(os.path.join(cwd, ".."))
        # Default to CWD with a warning if unsure.
        print(
            f"Warning: `__file__` not defined. Using CWD '{cwd}' as PROJECT_ROOT. "
            "Ensure this is correct for data and model paths."
        )
        return cwd

_PROJECT_ROOT_RISK_TRAIN = _get_project_root_for_risk_training()

# --- Configuration Paths ---
DATA_RAW_PATH = os.path.join(_PROJECT_ROOT_RISK_TRAIN, 'data', 'raw')
SAMPLE_NEWS_FILENAME = 'sample_risk_news.csv' # Source data for training
MODEL_STORE_PATH_RISK = os.path.join(_PROJECT_ROOT_RISK_TRAIN, 'models_store', 'risk_detection')
REPORTS_PATH_RISK = os.path.join(_PROJECT_ROOT_RISK_TRAIN, 'reports', 'experiment_logs')

# Ensure necessary directories exist
os.makedirs(MODEL_STORE_PATH_RISK, exist_ok=True)
os.makedirs(REPORTS_PATH_RISK, exist_ok=True)

# --- Experiment Logging ---
_RISK_CLASSIFICATION_LOG_FILE = os.path.join(REPORTS_PATH_RISK, 'risk_classification_experiments.csv')
_risk_classification_records = [] # Module-level list

def log_risk_classification_experiment(
    model_name: str, params: dict, metrics: dict, run_timestamp: str | None = None
):
    """
    Logs the details of a risk classification model training experiment.

    Args:
        model_name: Name of the specific model configuration (e.g., "MultinomialNB_TFIDF").
        params: Dictionary of parameters used for the model/pipeline.
                Should include 'model_path' if the model is saved.
        metrics: Dictionary of evaluation metrics, including 'accuracy' and
                 classification report sub-dictionaries like 'macro avg'.
        run_timestamp: Timestamp of the run; defaults to current time if None.
    """
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    macro_avg_metrics = metrics.get('macro avg', {})
    record = {
        'Timestamp': run_timestamp,
        'Model_Type': 'RiskCategoryClassifier', # General type
        'Model_Name': model_name,
        'Parameters': str(params), # String representation of parameters
        'Accuracy': metrics.get('accuracy'),
        'Precision_Macro': macro_avg_metrics.get('precision'),
        'Recall_Macro': macro_avg_metrics.get('recall'),
        'F1_Macro': macro_avg_metrics.get('f1-score'),
        # Add other specific metrics if needed, e.g., from report_dict['label_name']
    }
    _risk_classification_records.append(record)
    print(f"Logged experiment for risk classifier: {model_name}")

def save_risk_classification_log():
    """Saves all accumulated risk classification experiment records to a CSV file."""
    if not _risk_classification_records:
        print("No new risk classification experiments were logged to save.")
        return

    log_df = pd.DataFrame(_risk_classification_records)
    try:
        header_needed = not (
            os.path.exists(_RISK_CLASSIFICATION_LOG_FILE) and
            os.path.getsize(_RISK_CLASSIFICATION_LOG_FILE) > 0
        )
        log_df.to_csv(_RISK_CLASSIFICATION_LOG_FILE, mode='a', header=header_needed, index=False)
        print(f"Risk classification experiment log saved/appended to '{_RISK_CLASSIFICATION_LOG_FILE}'")
        _risk_classification_records.clear()
    except Exception as e:
        print(f"Error saving risk classification experiment log: {e}")

# --- Text Preprocessing ---
def preprocess_text_for_risk(text: str) -> str:
    """
    Cleans and preprocesses text data for risk classification.

    Steps:
    1. Converts text to lowercase.
    2. Removes punctuation and non-alphanumeric characters (keeps spaces).
    3. Tokenizes text using NLTK's word_tokenize.
    4. Removes English stopwords and non-alphabetic tokens.
    5. Joins filtered tokens back into a string.

    Requires NLTK 'punkt' and 'stopwords' resources to be available.
    These are checked by `_ensure_nltk_resources()` at script start.

    Args:
        text: The input string to preprocess.

    Returns:
        A string containing the preprocessed text, or an empty string if
        input is invalid or critical NLTK resources are missing.
    """
    if not isinstance(text, str):
        return ""
    if _NLTK_WORD_TOKENIZER_CACHE is None or _STOP_WORDS_ENGLISH_CACHE is None:
        # This should ideally not be reached if _ensure_nltk_resources() ran successfully and exited on failure.
        print("Error: NLTK tokenizer or stopwords not initialized. Cannot preprocess text.")
        return ""

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation, keep word chars and spaces
    try:
        tokens = _NLTK_WORD_TOKENIZER_CACHE(text)
        # Filter out stopwords and tokens that are not purely alphabetic
        filtered_tokens = [
            word for word in tokens if word.isalpha() and word not in _STOP_WORDS_ENGLISH_CACHE
        ]
        return " ".join(filtered_tokens)
    except Exception as e_preprocess: # Should be rare if NLTK resources are confirmed
        print(f"Error during text tokenization/filtering for text '{text[:50]}...': {e_preprocess}")
        return "" # Return empty string on error during tokenization

# --- Main Training Function ---
def train_risk_classifier():
    """
    Main function to train a text classification model for risk categories.

    Orchestrates data loading, text preprocessing, data splitting,
    model training (TF-IDF + Multinomial Naive Bayes pipeline), evaluation,
    model saving, and experiment logging.
    """
    print("--- Starting Risk Category Classification Model Training ---")
    
    news_data_full_path = os.path.join(DATA_RAW_PATH, SAMPLE_NEWS_FILENAME)
    if not os.path.exists(news_data_full_path):
        print(f"Error: Sample news data file not found at '{news_data_full_path}'. Cannot train model.")
        return

    try:
        df = pd.read_csv(news_data_full_path)
        # Drop rows where essential 'summary' or 'risk_category' are missing
        df.dropna(subset=['summary', 'risk_category'], inplace=True)
        if df.empty:
            print("No data available after dropping rows with missing summary or risk_category. Training cannot proceed.")
            return
        print(f"Loaded {len(df)} news items with summary and risk_category for training.")
        print("Initial Risk Category Distribution:\n", df['risk_category'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
    except Exception as e_load:
        print(f"Error loading or performing initial processing on news data: {e_load}")
        return

    print("Preprocessing text summaries...")
    df['processed_summary'] = df['summary'].apply(preprocess_text_for_risk)

    # Remove rows where processed_summary became empty (e.g., if summary was only stopwords/punctuation)
    df = df[df['processed_summary'].str.strip().astype(bool)]
    if df.empty:
        print("No data remaining after text preprocessing (all summaries might have been empty or just stopwords). Training cannot proceed.")
        return
    print(f"{len(df)} items remaining after preprocessing and filtering empty summaries.")

    X = df['processed_summary']
    y = df['risk_category']

    # Check for sufficient data and classes before splitting
    min_samples_for_split = 10 # Adjusted minimum
    min_classes = 2
    if len(X) < min_samples_for_split or y.nunique() < min_classes:
        print(f"Not enough data samples (found {len(X)}, need >= {min_samples_for_split}) or "
              f"unique classes (found {y.nunique()}, need >= {min_classes}) to train a meaningful model. Aborting.")
        return

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError: # Stratify might fail with very small classes
        print("Warning: Stratification failed during train-test split (likely due to very small class sizes). "
              "Proceeding with a non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

    if X_train.empty or X_test.empty:
        print(f"Train or test set is empty after split. Train size: {len(X_train)}, Test size: {len(X_test)}. Cannot proceed.")
        return
    print(f"Data split into Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Define and train the model pipeline (TF-IDF + Multinomial Naive Bayes)
    # Parameters chosen are common defaults; can be tuned via hyperparameter optimization.
    tfidf_params = {'max_df': 0.95, 'min_df': 2, 'ngram_range': (1, 2), 'sublinear_tf': True} # Adjusted min_df
    clf_params = {'alpha': 0.1} # Smoothing parameter for MNB

    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', MultinomialNB(**clf_params))
    ])
    model_name_for_log = "MultinomialNB_TFIDF_tuned" # More descriptive name
    # Combine all parameters for logging
    pipeline_params_for_log = {f"tfidf_{k}": v for k,v in tfidf_params.items()}
    pipeline_params_for_log.update({f"clf_{k}": v for k,v in clf_params.items()})

    print(f"\nTraining {model_name_for_log} model with parameters: {pipeline_params_for_log}...")
    try:
        model_pipeline.fit(X_train, y_train)
        y_pred_on_test = model_pipeline.predict(X_test)

        accuracy_val = accuracy_score(y_test, y_pred_on_test)
        # Get all unique labels from the original 'y' to ensure report consistency
        all_unique_labels = sorted(y.unique())
        class_report_dict = classification_report(
            y_test, y_pred_on_test, output_dict=True, zero_division=0,
            labels=all_unique_labels, target_names=all_unique_labels
        )
        class_report_str = classification_report(
            y_test, y_pred_on_test, zero_division=0,
            labels=all_unique_labels, target_names=all_unique_labels
        )

        print(f"\n--- {model_name_for_log} Evaluation ---")
        print(f"Accuracy: {accuracy_val:.4f}")
        print("Classification Report:\n", class_report_str)

        metrics_for_log = {
            'accuracy': accuracy_val,
            'weighted avg': class_report_dict.get('weighted avg'), # For overall performance
            'macro avg': class_report_dict.get('macro avg')    # For unweighted average performance
        }

        # Save the trained model pipeline
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        model_filename_joblib = f"risk_category_classifier_{model_name_for_log}_{timestamp}.joblib"
        model_full_save_path = os.path.join(MODEL_STORE_PATH_RISK, model_filename_joblib)
        joblib.dump(model_pipeline, model_full_save_path)
        print(f"Saved trained risk classification pipeline to: {model_full_save_path}")
        pipeline_params_for_log['model_path'] = model_full_save_path # Add path to logged params

        log_risk_classification_experiment(model_name_for_log, pipeline_params_for_log, metrics_for_log)

    except Exception as e_train_eval:
        print(f"Error during model training or evaluation for {model_name_for_log}: {e_train_eval}")

    save_risk_classification_log()
    print("\n--- Risk Category Classification Model Training Finished ---")


if __name__ == '__main__':
    # Ensure NLTK resources are ready before attempting any NLP operations
    if not _ensure_nltk_resources():
        print("Exiting script due to missing NLTK resources.")
        sys.exit(1) # Exit if critical resources are unavailable

    # The _PROJECT_ROOT_RISK_TRAIN should be correctly set by _get_project_root_for_risk_training()
    # This standalone execution block assumes it's run as a script.
    # Path re-adjustment might be needed if run from an unexpected CWD.
    # For robustness, we can re-verify/re-set paths if running as main.
    if _PROJECT_ROOT_RISK_TRAIN is None or not os.path.exists(os.path.join(_PROJECT_ROOT_RISK_TRAIN, 'data')):
        print("Re-evaluating PROJECT_ROOT for standalone script execution...")
        _PROJECT_ROOT_RISK_TRAIN = _get_project_root_for_risk_training() # Call again to ensure it's set
        
        # Update global path configurations if PROJECT_ROOT was re-evaluated
        DATA_RAW_PATH = os.path.join(_PROJECT_ROOT_RISK_TRAIN, 'data', 'raw')
        MODEL_STORE_PATH_RISK = os.path.join(_PROJECT_ROOT_RISK_TRAIN, 'models_store', 'risk_detection')
        REPORTS_PATH_RISK = os.path.join(_PROJECT_ROOT_RISK_TRAIN, 'reports', 'experiment_logs')
        _RISK_CLASSIFICATION_LOG_FILE = os.path.join(REPORTS_PATH_RISK, 'risk_classification_experiments.csv')
        
        os.makedirs(MODEL_STORE_PATH_RISK, exist_ok=True)
        os.makedirs(REPORTS_PATH_RISK, exist_ok=True)
        print(f"Updated PROJECT_ROOT for standalone run: {_PROJECT_ROOT_RISK_TRAIN}")

    train_risk_classifier()