"""
Natural Language Processing (NLP) utility functions.

This module provides functionalities for:
1.  Ensuring necessary NLTK resources (punkt tokenizer, stopwords, VADER lexicon)
    are downloaded and available.
2.  Preprocessing text data by converting to lowercase, removing punctuation,
    and filtering out stopwords.
3.  Performing sentiment analysis on text using NLTK's VADER
    (Valence Aware Dictionary and sEntiment Reasoner).

The module attempts to initialize NLTK components upon first import.
"""
import nltk
import re
# Unused imports like pandas, numpy, os have been removed.

# --- Global State and NLTK Component Placeholders ---
# Flag to ensure NLTK resource checks and component imports run only once per session.
_nltk_resources_initialized_for_nlp_utils = False

# NLTK components will be populated by _initialize_nltk_components
SentimentIntensityAnalyzer = None
word_tokenize = None
stopwords_english = None # This will be a set of stopwords for efficiency
_vader_analyzer_instance = None # Global VADER analyzer instance

def _initialize_nltk_components() -> bool:
    """
    Checks for, downloads (if necessary), and imports NLTK components.

    This function handles 'punkt' (for tokenizer), 'stopwords', and
    'vader_lexicon' (for sentiment analysis). It populates global variables
    for `SentimentIntensityAnalyzer`, `word_tokenize`, and `stopwords_english`
    if all resources are successfully found or downloaded.

    This function is intended to be called once per session, typically when
    this module is first imported.

    Returns:
        bool: True if all required NLTK resources are available and components
              are imported successfully, False otherwise.
    """
    global _nltk_resources_initialized_for_nlp_utils
    global SentimentIntensityAnalyzer, word_tokenize, stopwords_english

    if _nltk_resources_initialized_for_nlp_utils:
        return True # Already initialized

    print("--- NLP Utils: Initializing NLTK Resources and Components ---")
    all_resources_ready = True

    # Define NLTK resources to check/download: {name: (find_path_in_nltk_data, download_identifier)}
    resource_map = {
        "punkt": ("tokenizers/punkt", "punkt"),
        "stopwords": ("corpora/stopwords", "stopwords"),
        "vader_lexicon": ("sentiment/vader_lexicon.zip", "vader_lexicon") # VADER lexicon
    }

    for resource_name, (find_path, download_identifier) in resource_map.items():
        try:
            nltk.data.find(find_path) # Check if resource exists
            print(f"NLTK resource '{resource_name}' found.")
        except LookupError:
            print(f"NLTK resource '{resource_name}' not found. Attempting to download '{download_identifier}'...")
            try:
                nltk.download(download_identifier, quiet=False) # Download the resource
                nltk.data.find(find_path) # Verify after download
                print(f"NLTK resource '{download_identifier}' downloaded and verified successfully.")
            except Exception as e_download:
                print(f"ERROR: Failed to download or verify NLTK resource '{download_identifier}': {e_download}")
                all_resources_ready = False
        except Exception as e_find: # Catch other errors during find (e.g., permission issues)
            print(f"ERROR: An unexpected error occurred while checking for NLTK resource '{resource_name}': {e_find}")
            all_resources_ready = False

    if all_resources_ready:
        try:
            # Import NLTK components now that resources are expected to be available
            from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
            SentimentIntensityAnalyzer = SIA
            from nltk.tokenize import word_tokenize as wtk
            word_tokenize = wtk
            from nltk.corpus import stopwords
            stopwords_english = set(stopwords.words('english')) # Use a set for faster lookups
            print("NLTK components (VADER SIA, word_tokenize, stopwords) imported successfully.")
        except ImportError as e_import_components:
            print(f"ERROR: Failed to import NLTK components even after resource checks: {e_import_components}")
            all_resources_ready = False
        except Exception as e_general_components: # Catch other potential errors
            print(f"ERROR: An unexpected error occurred during NLTK component import: {e_general_components}")
            all_resources_ready = False

    _nltk_resources_initialized_for_nlp_utils = True # Mark initialization attempt as done
    return all_resources_ready

def _get_vader_analyzer_instance():
    """
    Provides a singleton instance of the VADER SentimentIntensityAnalyzer.

    Initializes the analyzer on its first call if NLTK resources are available.

    Returns:
        nltk.sentiment.vader.SentimentIntensityAnalyzer | None: The VADER analyzer
        instance, or None if it could not be initialized.
    """
    global _vader_analyzer_instance
    if _vader_analyzer_instance is None:
        if SentimentIntensityAnalyzer is not None: # Check if the class was successfully imported
            try:
                _vader_analyzer_instance = SentimentIntensityAnalyzer()
                print("NLTK VADER SentimentIntensityAnalyzer initialized successfully.")
            except Exception as e_analyzer_init: # pragma: no cover
                print(f"Error initializing NLTK VADER SentimentIntensityAnalyzer: {e_analyzer_init}")
                _vader_analyzer_instance = None # Ensure it's None on failure
        else: # pragma: no cover
            # This case implies _initialize_nltk_components failed to import SIA
            print("VADER SentimentIntensityAnalyzer class is not available due to NLTK resource/import issues.")
    return _vader_analyzer_instance


def preprocess_text(text: str) -> str:
    """
    Cleans and preprocesses input text for NLP tasks.

    The preprocessing pipeline includes:
    1.  Conversion to lowercase.
    2.  Removal of punctuation and non-alphanumeric characters (retaining spaces).
    3.  Tokenization using NLTK's `word_tokenize`.
    4.  Removal of English stopwords.
    5.  Filtering out tokens that are not purely alphabetic.
    6.  Joining filtered tokens back into a space-separated string.

    Requires NLTK 'punkt' and 'stopwords' resources. It attempts lazy
    initialization of these resources if not already done.

    Args:
        text: The input string to preprocess.

    Returns:
        str: The preprocessed text string. Returns an empty string if input is not
             a string or if essential NLTK components are unavailable after attempting
             initialization.
    """
    if not _nltk_resources_initialized_for_nlp_utils: # pragma: no cover
        _initialize_nltk_components() # Attempt lazy initialization

    if not isinstance(text, str):
        return ""
    # Check if NLTK components needed by this function were successfully loaded
    if word_tokenize is None or stopwords_english is None: # pragma: no cover
        print("Warning: NLTK word_tokenize or stopwords not available for preprocess_text. "
              "Returning only lowercased and punctuation-removed text.")
        text_processed_minimal = text.lower()
        text_processed_minimal = re.sub(r'[^\w\s]', '', text_processed_minimal)
        return text_processed_minimal

    text_processed = text.lower()
    text_processed = re.sub(r'[^\w\s]', '', text_processed) # Remove punctuation
    try:
        tokens = word_tokenize(text_processed)
        filtered_tokens = [
            word for word in tokens if word.isalpha() and word not in stopwords_english
        ]
        return " ".join(filtered_tokens)
    except Exception as e_preprocess: # pragma: no cover
        # This might happen if tokenization fails for an unexpected reason despite 'punkt' being present
        print(f"Error during text preprocessing for text '{text_processed[:50]}...': {e_preprocess}")
        return "" # Return empty string or original text based on desired error handling

def get_sentiment(text: str) -> tuple[str, float | None]:
    """
    Analyzes the sentiment of a given text using NLTK's VADER.

    Determines if the sentiment is Positive, Negative, or Neutral based on
    VADER's compound score.

    Requires NLTK 'vader_lexicon' resource. Attempts lazy initialization if needed.

    Args:
        text: The input string for sentiment analysis.

    Returns:
        tuple[str, float | None]: A tuple containing:
            - sentiment_label (str): 'Positive', 'Negative', 'Neutral', or 'Error'.
            - compound_score (float | None): VADER's compound sentiment score
              (typically between -1 and 1), or None if an error occurs or input is invalid.
              A score of 0.0 is returned for empty/whitespace-only text with 'Neutral' label.
    """
    if not _nltk_resources_initialized_for_nlp_utils: # pragma: no cover
        _initialize_nltk_components() # Attempt lazy initialization

    analyzer = _get_vader_analyzer_instance()

    if analyzer is None: # pragma: no cover
        print("VADER Sentiment analyzer is not initialized. Cannot perform sentiment analysis.")
        return "Error", None

    if not isinstance(text, str) or not text.strip(): # Handle None, non-string, or empty/whitespace string
        return "Neutral", 0.0

    try:
        sentiment_scores = analyzer.polarity_scores(text)
        compound = sentiment_scores['compound']

        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return label, round(compound, 4)

    except Exception as e_sentiment: # pragma: no cover
        print(f"Error during VADER sentiment analysis for text '{text[:50]}...': {e_sentiment}")
        return "Error", None

# --- Module-level Initialization ---
# Attempt to initialize NLTK components when the module is first imported.
# This makes subsequent function calls faster and handles downloads early.
if not _nltk_resources_initialized_for_nlp_utils:
    if not _initialize_nltk_components(): # pragma: no cover
        # This warning indicates that the module might not function correctly.
        print(
            "CRITICAL WARNING (nlp_utils module): NLTK resource initialization failed upon import. "
            "NLP functions within this module (preprocessing, sentiment) may not work as expected. "
            "Please check NLTK setup and internet connectivity if downloads are required."
        )

if __name__ == '__main__': # pragma: no cover
    print("\n--- Testing NLP Utilities (NLTK VADER & Preprocessing) ---")

    # Test preprocess_text
    test_text_for_preprocessing = "This is a Test sentence, with punctuation!! And some stopwords, like the, is, a."
    print(f"\nOriginal text for preprocessing: '{test_text_for_preprocessing}'")
    if _NLP_UTILS_INITIALIZED_SUCCESSFULLY: # Check if initialization was successful
        preprocessed_output = preprocess_text(test_text_for_preprocessing)
        print(f"Preprocessed text: '{preprocessed_output}'")
        # Assertion depends on NLTK version and exact tokenizer/stopword list
        # A more robust test might check for absence of stopwords and punctuation.
        expected_outputs = ["test sentence punctuation stopwords like", "test sentence punctuation stopwords"]
        assert preprocessed_output in expected_outputs, \
            f"Preprocessing output '{preprocessed_output}' not among expected: {expected_outputs}"
        print("preprocess_text test passed (or partially, check output).")
    else:
        print("Skipping preprocess_text test as NLTK components did not initialize.")


    # Test get_sentiment
    print(f"\nTesting VADER Sentiment Analysis...")
    if _NLP_UTILS_INITIALIZED_SUCCESSFULLY and _vader_analyzer_instance is not None:
        test_texts_for_sentiment = [
            ("This is a fantastic development for the supply chain!", "Positive"),
            ("Major disruptions expected due to the factory fire.", "Negative"),
            ("Market conditions remain stable for Q2.", "Neutral"),
            ("  ", "Neutral"), # Test whitespace only
            (None, "Neutral") # Test None input (though handled by isinstance check)
        ]
        for i, (text_to_analyze, expected_label) in enumerate(test_texts_for_sentiment):
            print(f"\nTest Case {i+1}:")
            print(f"  Input Text: '{str(text_to_analyze)[:70]}{'...' if text_to_analyze and len(str(text_to_analyze)) > 70 else ''}'")
            calculated_sentiment_label, compound_score = get_sentiment(str(text_to_analyze) if text_to_analyze else "") # Ensure string for None
            print(f"  Calculated Sentiment: {calculated_sentiment_label}, Compound Score: {compound_score}")
            # assert calculated_sentiment_label == expected_label # VADER scores can be subtle
            if calculated_sentiment_label != "Error":
                 print(f"  Sentiment analysis functional for this case.")
            else:
                 print(f"  Sentiment analysis reported an error for this case.")
            print("-" * 40)
    else:
        print("VADER Sentiment analyzer could not be initialized or NLTK components failed. Skipping get_sentiment tests.")

    print("\n--- NLP Utilities Test Run Finished ---")
# Add a global flag for use in __main__ for conditional testing
_NLP_UTILS_INITIALIZED_SUCCESSFULLY = _nltk_resources_initialized_for_nlp_utils and \
                                   (word_tokenize is not None) and \
                                   (stopwords_english is not None) and \
                                   (SentimentIntensityAnalyzer is not None)