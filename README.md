# SupplyChain AI

<p align="center">
  <a href="https://www.python.org/downloads/release/python-390/" target="_blank"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Version"></a>
  <a href="https://streamlit.io" target="_blank"><img src="https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://pandas.pydata.org" target="_blank"><img src="https://img.shields.io/badge/Pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"></a>
  <a href="https://numpy.org" target="_blank"><img src="https://img.shields.io/badge/NumPy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://scikit-learn.org" target="_blank"><img src="https://img.shields.io/badge/Scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"></a>
  <a href="https://www.tensorflow.org" target="_blank"><img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" alt="TensorFlow"></a>
  <a href="https://keras.io" target="_blank"><img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white" alt="Keras"></a>
  <a href="https://www.nltk.org" target="_blank"><img src="https://img.shields.io/badge/NLTK-%230C59A3.svg?style=for-the-badge&logo=nltk&logoColor=white" alt="NLTK"></a>
  <a href="https://facebook.github.io/prophet/" target="_blank"><img src="https://img.shields.io/badge/Prophet-%23007F7F.svg?style=for-the-badge&logo=facebook&logoColor=white" alt="Prophet"></a>
  <a href="https://alkaline-ml.com/pmdarima/" target="_blank"><img src="https://img.shields.io/badge/pmdarima-%234B8BBE.svg?style=for-the-badge" alt="pmdarima"></a>
  <a href="https://xgboost.readthedocs.io/" target="_blank"><img src="https://img.shields.io/badge/XGBoost-%230061A6.svg?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"></a>
  <a href="https://lightgbm.readthedocs.io/" target="_blank"><img src="https://img.shields.io/badge/LightGBM-%238E44AD.svg?style=for-the-badge&logo=lightgbm&logoColor=white" alt="LightGBM"></a>
  <a href="https://matplotlib.org/" target="_blank"><img src="https://img.shields.io/badge/Matplotlib-%231f77b4.svg?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib"></a>
  <a href="https://seaborn.pydata.org/" target="_blank"><img src="https://img.shields.io/badge/Seaborn-%234C72B0.svg?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn"></a>
</p>

<p align="center">
  <a href="#14-license">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License: MIT">
  </a>
</p>

## 🚀 Project Overview

The **Supply Chain Intelligence Platform (SCIP)** is an advanced analytical application meticulously engineered to empower businesses with AI-driven insights for the strategic optimization of their supply chain operations. In an increasingly volatile global market characterized by complex dynamics, SCIP delivers a robust suite of tools designed to address critical challenges through:

* **Precision Demand Forecasting:** By leveraging a comprehensive array of classical time series techniques, machine learning regression models, and sophisticated deep learning architectures (LSTMs), SCIP aims to predict future demand with significantly enhanced accuracy and granularity.
* **Intelligent Inventory Optimization:** The platform translates these high-fidelity demand forecasts into actionable, data-informed inventory strategies. It calculates optimal economic order quantities (EOQ), dynamic safety stock levels, and precise reorder points, effectively balancing customer service imperatives with inventory holding and ordering costs.
* **Proactive Risk Detection & Assessment:** SCIP utilizes Natural Language Processing (NLP) techniques to analyze unstructured external data sources (e.g., news articles, industry reports). This allows for the early identification, categorization, and sentiment-scored assessment of potential supply chain disruptions, enabling businesses to formulate timely and effective mitigation strategies.

This project, developed entirely in Python and presented through an interactive Streamlit web application, demonstrates a cohesive, end-to-end data science workflow. This workflow encompasses data ingestion and rigorous preprocessing, sophisticated feature engineering, comprehensive model training and evaluation across diverse algorithmic families, and the deployment of insightful, user-friendly dashboards. The core objective is to showcase the tangible, transformative benefits of applying artificial intelligence and advanced analytics to build more resilient, agile, and data-centric supply chains.

---

## ✨ Core Modules & Key Functionalities

SCIP is architected around three primary, interconnected modules, each targeting a critical facet of modern supply chain management:

### 1. Demand Forecasting Insights
* **Objective:** To systematically develop, rigorously evaluate, compare, and deploy a diverse range of forecasting models to achieve superior demand prediction accuracy. This module uses the publicly available Walmart Sales Forecasting dataset as a practical, real-world case study.
* **Key Functionalities:**
    * **Interactive Exploratory Data Analysis (EDA):** In-depth visual analysis of historical sales patterns, identification of trends, seasonality, cyclicality, specific holiday impacts, and correlations with external factors like CPI and fuel prices.
    * **Comprehensive Feature Engineering (`src/features/demand_features.py`):**
        * Creation of rich temporal features: year, month, week of year, day of year, day of week.
        * Generation of cyclical (sine/cosine) transformations for month and week of year to capture periodic patterns effectively.
        * Development of extensive lag features for sales data (e.g., sales from 1, 4, 12, 52 weeks prior).
        * Calculation of rolling window statistics (mean, standard deviation, min, max) on past sales data to capture recent trends and volatility, while preventing data leakage using appropriate shifts.
        * Encoding of categorical features like store type and handling of holiday indicators.
    * **Diverse Modeling Suite (Training scripts in `src/models/demand_forecasting/`):**
        * *Classical Time Series Models (`classical_timeseries.py`):* Naive, Seasonal Naive, Exponential Smoothing (ETS - automated best model selection), SARIMA (auto_arima for automated order selection, with exogenous variable support), Prophet (with holiday and regressor support).
        * *Machine Learning Regression (Global Models - `classical_regression.py`):* Random Forest, XGBoost, LightGBM, trained on the full feature set including engineered time-series features.
        * *Deep Learning (`deep_learning.py`):* Long Short-Term Memory (LSTM) networks, designed to capture complex temporal dependencies, with feature and target scaling.
    * **MLOps-like Experiment Tracking:** Systematic logging of all model training experiments, including model type, hyperparameters, features used, training/test periods, performance metrics (MAE, RMSE, MAPE), and paths to saved model artifacts (models, scalers). Log files are stored in `reports/experiment_logs/`.
    * **"Model Performance Comparison" Dashboard (`model_performance_page.py`):** Interactive tables and visualizations (bar charts, box plots) for robust comparison of all trained models across various metrics, aggregated by model family and specific model configurations.
    * **"Forecast Explorer" Tool (`forecast_explorer_page.py`):** An interactive user interface allowing:
        * Selection of specific Store-Department combinations.
        * Choice of any trained forecasting model from the logged experiments.
        * Visualization of historical sales data, actual test set values, and the selected model's predictions on the test set.
        * Generation and display of **new multi-step future forecasts** for a user-defined horizon. This features iterative forecasting capabilities for machine learning regression and LSTM models, where predictions are fed back as inputs for subsequent steps.
        * Inspection of model-specific hyperparameters and paths to saved artifacts.

### 2. Inventory Strategy & Optimization
* **Objective:** To translate the demand forecasts and their inherent uncertainty (quantified, for example, by model RMSE from Module 1) into optimal inventory parameters. This module focuses on effectively balancing the imperative of high customer service levels against the financial implications of inventory holding and ordering costs.
* **Key Functionalities (`inventory_optimization_page.py`, `src/models/inventory_optimization/core_models.py`):**
    * **Core Inventory Models:** Robust calculation of:
        * Economic Order Quantity (EOQ) based on annual demand, ordering costs, and holding costs.
        * Safety Stock (SS) considering both demand variability (e.g., standard deviation of demand, often derived from forecast RMSE) and lead time variability, for a target service level (using Z-scores).
        * Reorder Point (ROP) based on average demand during lead time plus the calculated safety stock.
    * **Seamless Forecast Integration:** Option to automatically populate key demand inputs (average weekly demand, standard deviation of weekly demand) for inventory calculations, using actuals from a model's test period and its RMSE, based on user-selected forecasts from Module 1.
    * **Interactive Parameter Calculator:** A user-friendly interface enabling dynamic input and adjustment of critical parameters such as item costs, ordering costs, annual holding cost rates (as a percentage of item cost), supplier lead time characteristics (average and standard deviation), and desired target service levels.
    * **Inventory Policy Simulation (Discrete Time, (Q,R)-like Policy):**
        * Simulates inventory levels period-by-period (e.g., weekly) over a user-defined horizon.
        * Utilizes the calculated EOQ (as order quantity Q) and ROP (as reorder point R).
        * Offers a choice of demand patterns for the simulation: constant average demand (from inputs) or actual historical demand for a selected store-department.
        * Provides dynamic visualizations of inventory levels over time, clearly marking reorder points, safety stock thresholds, and highlighting stockout occurrences.
        * Summarizes key simulation performance indicators: total simulated demand, total units sold, achieved fill rate/service level (%), total stockout units, and the total number of replenishment orders placed.
    * **Sensitivity Analysis:** An interactive tool allowing users to explore how the calculated EOQ, Safety Stock, and ROP respond to systematic variations in one key input parameter at a time (e.g., holding cost rate, lead time variability), while others are held constant. Results are presented through clear tables and line plots.

### 3. Proactive Risk Alerts (Demonstration)
* **Objective:** To showcase a proof-of-concept for identifying, categorizing, and assessing potential supply chain risks by processing unstructured external data. This module uses a manually curated sample news dataset (`sample_risk_news.csv`) for demonstration.
* **Key Functionalities (`risk_detection_page.py`, `src/models/risk_detection/train_risk_classifier.py`, `src/utils/nlp_utils.py`):**
    * **Sample News Data Processing:** Ingestion, parsing, and structured display of news articles (headlines, summaries, publication dates, sources) relevant to potential supply chain disruptions.
    * **NLP-Powered Insights:**
        * *Sentiment Analysis:* Leverages NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) to determine the sentiment (Positive, Negative, Neutral) and a compound sentiment score for each news summary, providing an immediate emotional or contextual indicator.
        * *Risk Category Classification:* Employs a custom-trained text classification pipeline (TF-IDF Vectorizer + Multinomial Naive Bayes model) to automatically assign news items to predefined risk categories (e.g., "Natural Disaster," "Supplier Financial Issues," "Logistics Disruption," "Port Congestion"). The training script for this model is provided.
    * **Dynamic AI-Generated Risk Scoring:** A rule-based heuristic system combines the AI-predicted risk category (which has an inherent severity) and the VADER sentiment label to generate a dynamic, quantitative risk score (typically on a 0-10 scale) for each news item. This offers a more nuanced risk assessment than category alone.
    * **Interactive "Risk Alerts" Feed:**
        * Presents a filterable and sortable feed of processed news items. Each item displays its headline, summary, source, publication date, VADER sentiment (with a visual emoji indicator), the AI-predicted risk category (shown alongside the manually labeled category from the sample data for comparison), and the AI-generated risk score.
        * Features robust filtering options based on keywords (in headline, summary, or keywords field), manually assigned risk category, VADER sentiment, and the AI-generated risk score range.
        * Includes various sorting options for the risk feed, such as by publication date, AI risk score, or manual risk score.

---

## 4. High-Level Workflow

The platform's architecture and development adhere to a structured data science and application development workflow:

1.  **Data Ingestion & Preparation (`src/data_processing/load_walmart_data.py`):** Raw datasets (e.g., Walmart sales figures, store information, external features, sample news articles) are loaded from source files (CSV, Parquet). Initial cleaning steps include data type conversions (especially for dates), handling of obvious missing values, and merging disparate sources into a coherent base dataset.
2.  **Feature Engineering (`src/features/demand_features.py`, `src/utils/feature_engineering_utils.py`):** Relevant and informative features are meticulously crafted to enhance model performance. For demand forecasting, this includes creating temporal features (year, month, week, day of year, cyclical sine/cosine transformations for seasonality), lag features of the target variable, rolling window statistics (mean, std, min, max over defined past periods), and indicators for holidays or special events. For NLP tasks in risk detection, text preprocessing (cleaning, tokenization, stopword removal) is a key step.
3.  **Model Training & Evaluation (Scripts in `src/models/`):**
    * A diverse array of statistical, machine learning, and deep learning models are trained for their respective tasks (demand forecasting, risk classification).
    * Models are rigorously evaluated using appropriate metrics (e.g., MAE, RMSE, MAPE for forecasting; accuracy, precision, recall, F1-score for classification).
    * Training processes, model hyperparameters, feature sets used, evaluation metrics, and paths to saved model artifacts (models, scalers) are systematically logged in CSV files, simulating MLOps best practices for reproducibility and comparison.
4.  **Inventory Parameter Calculation & Simulation (`src/models/inventory_optimization/core_models.py`):** Established inventory management formulas (EOQ, Safety Stock, ROP) are implemented. These calculations utilize statistical outputs from the demand forecasting models (e.g., mean forecast as average demand, forecast error RMSE as a proxy for demand standard deviation) and user-defined cost/service level parameters. Simulation capabilities allow for testing inventory policies under various demand scenarios.
5.  **Interactive Application Development (Streamlit - `app/main_app.py` and page modules in `app/page_content/`):** A user-friendly web application is constructed using Streamlit to:
    * Present comprehensive Exploratory Data Analysis (EDA) insights from historical data.
    * Offer an interactive dashboard for comparing the performance characteristics of different forecasting models.
    * Enable users to dynamically explore individual model forecasts, generate new multi-step future predictions, and inspect model configurations.
    * Provide tools for calculating optimal inventory levels, interactively simulating inventory policies under different conditions, and performing sensitivity analyses on key inventory parameters.
    * Display a live-updating (conceptually) feed of potential supply chain risks, enriched with AI-driven categorization, sentiment scoring, and filtering capabilities.

---

## 5. Technology Stack

* **Programming Language:** [Python 3.9+](https://www.python.org/downloads/release/python-390/)
* **Core Data Science & Machine Learning:**
    * [Pandas](https://pandas.pydata.org/): Advanced data manipulation, time series handling, and analysis.
    * [NumPy](https://numpy.org/): Fundamental package for numerical computing in Python.
    * [Scikit-learn](https://scikit-learn.org/stable/): Comprehensive suite for machine learning, including preprocessing (MinMaxScaler, LabelEncoder), model selection (train_test_split), feature extraction (TfidfVectorizer), classification models (Multinomial Naive Bayes), regression models (RandomForestRegressor), and evaluation metrics.
    * [Statsmodels](https://www.statsmodels.org/stable/index.html): Statistical modeling, including ETS (Exponential Smoothing) and components for SARIMA analysis.
    * [Pmdarima](https://alkaline-ml.com/pmdarima/): Provides `auto_arima` for automated SARIMA model selection and fitting.
    * [Prophet](https://facebook.github.io/prophet/): Time series forecasting library developed by Meta, effective for data with strong seasonality and holiday effects.
    * [XGBoost](https://xgboost.readthedocs.io/en/stable/): Optimized distributed gradient boosting library designed for speed and performance.
    * [LightGBM](https://lightgbm.readthedocs.io/en/latest/): Fast, distributed, high-performance gradient boosting framework based on decision tree algorithms.
    * [TensorFlow](https://www.tensorflow.org/): End-to-end open-source platform for machine learning.
    * [Keras](https://keras.io/): High-level API for building and training deep learning models, integrated with TensorFlow. Used for LSTM network development.
* **Natural Language Processing (NLP):**
    * [NLTK (Natural Language Toolkit)](https://www.nltk.org/): Suite for text processing, including VADER sentiment analysis, tokenization (punkt), and stopword removal.
* **Scientific Computing & Statistics:**
    * [SciPy](https://scipy.org/): Ecosystem of open-source software for mathematics, science, and engineering, used here for statistical functions like Z-score calculation (`scipy.stats.norm`).
* **Web Application Framework:**
    * [Streamlit](https://streamlit.io/): Open-source app framework for creating and sharing beautiful, custom web apps for machine learning and data science projects in pure Python.
    * [Streamlit Option Menu](https://github.com/victoryhb/streamlit-option-menu): Custom component for creating sidebar/top navigation menus in Streamlit.
* **Data Visualization:**
    * [Matplotlib](https://matplotlib.org/): Comprehensive library for creating static, animated, and interactive visualizations in Python.
    * [Seaborn](https://seaborn.pydata.org/): Statistical data visualization library based on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
* **Utilities:**
    * [Joblib](https://joblib.readthedocs.io/en/latest/): For efficient saving and loading of Python objects, particularly scikit-learn models, scalers, and other ML artifacts.
    * `ast`, `re`, `datetime`, `warnings`, `os`, `sys`: Standard Python libraries for abstract syntax tree manipulation, regular expressions, date/time operations, warning control, operating system interactions, and system-specific parameters/functions, respectively.

---

## 6. Datasets Used

* **Walmart Sales Forecasting Data:**
    * **Source:** Publicly available dataset, originating from the [Kaggle - Walmart Recruiting - Store Sales Forecasting competition](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting).
    * **Files Expected:** `train.csv`, `stores.csv`, `features.csv` (to be placed in the `data/raw/` directory).
    * **Purpose:** This dataset forms the backbone for training, evaluating, and demonstrating the diverse range of demand forecasting models within Module 1. The historical sales, store attributes, and external features (like CPI, temperature, holiday flags) are crucial for building accurate predictive models. The outputs from these models (forecasts and error metrics like RMSE) subsequently serve as critical inputs for the inventory optimization calculations performed in Module 2.
* **Sample Risk News Data:**
    * **File:** `sample_risk_news.csv` (expected to be present in the `data/raw/` directory).
    * **Content:** A small, curated collection of approximately 25 sample news headlines and summaries. Each entry is manually annotated with a `risk_category` (e.g., "Logistics Disruption", "Natural Disaster") and a baseline `risk_score` for comparison purposes.
    * **Purpose:** This dataset is utilized in Module 3 (Proactive Risk Alerts) to demonstrate the Natural Language Processing capabilities. It allows the system to showcase sentiment analysis using VADER, AI-driven risk category classification (via a trained TF-IDF + Naive Bayes model), and the generation of a dynamic AI risk score for each news item. This provides a tangible, albeit simplified, feed for the "Risk Alerts" dashboard.

---

## 7. Installation & Setup

### Prerequisites
* Python 3.9 or a newer version installed on your system.
* `pip` (the Python package installer) available and updated.
* A virtual environment manager is highly recommended (e.g., `venv` which is built into Python, or Conda).

### Steps

1.  **Clone the Repository (If applicable):**
    If you have access to a Git repository for this project, clone it:
    ```bash
    git clone [YOUR_PROJECT_GITHUB_URL_HERE]
    cd [PROJECT_DIRECTORY_NAME] # e.g., supply_chain_ai_platform
    ```
    If you have the project files directly, navigate to the project's root directory.

2.  **Create and Activate a Python Virtual Environment:**
    It's best practice to use a virtual environment to manage project dependencies.
    (Using `.venv` as the environment name is a common convention).
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate   # On macOS/Linux
    # .venv\Scripts\activate    # On Windows PowerShell
    # conda create -n scip_env python=3.9 # Example for Conda, then: conda activate scip_env
    ```

3.  **Install Dependencies:**
    A `requirements.txt` file is provided in the project root, listing all necessary Python packages. Install them using:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: To generate or update `requirements.txt` from your active, working environment in the future, you can run: `pip freeze > requirements.txt`)*

4.  **Download NLTK Resources:**
    The application's NLP utility scripts (`src/utils/nlp_utils.py`) are designed to attempt automatic download of required NLTK resources (punkt, stopwords, vader_lexicon) upon their first use if they are not found. However, if you encounter issues (e.g., due to network firewalls or proxy settings), or prefer to download them proactively, you can run the following commands in a Python interpreter within your activated virtual environment:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    ```

5.  **Prepare Data Files:**
    * Ensure the Walmart sales data CSV files (`train.csv`, `features.csv`, `stores.csv`) are placed into the `data/raw/` directory within your project structure.
    * Verify that the `sample_risk_news.csv` file is also present in the `data/raw/` directory.
    * The primary feature-engineered dataset (e.g., `walmart_data_featured.parquet`) will be generated in the `data/processed/` directory when you run the feature engineering script (see "Usage Instructions" below). Intermediate cleaned data might also be stored there.

---

## 8. Usage Instructions

1.  **Activate Virtual Environment:** Ensure your Python virtual environment (e.g., `.venv` or your Conda environment) is activated.
2.  **Navigate to Project Root:** Open your terminal or command prompt and navigate to the root directory of the SCIP project.
3.  **Data Processing & Feature Engineering (Run once initially, or when raw data changes):**
    This script loads the raw Walmart data, performs initial cleaning, engineers a comprehensive set of features, and saves the processed dataset.
    ```bash
    python src/features/demand_features.py
    ```
    This will generate `walmart_data_featured.parquet` (and potentially `walmart_data_cleaned.parquet`) in the `data/processed/` directory.

4.  **Train Models (Optional - Run if you wish to regenerate models and their corresponding experiment logs):**
    The Streamlit application is designed to load and utilize pre-existing model files (from `models_store/`) and experiment logs (from `reports/experiment_logs/`) if they are present. If you want to retrain all models or experiment with changes:
    * **Demand Forecasting Models:**
        ```bash
        python src/models/demand_forecasting/classical_timeseries.py
        python src/models/demand_forecasting/classical_regression.py
        python src/models/demand_forecasting/deep_learning.py
        ```
    * **Risk Category Classification Model:**
        ```bash
        python src/models/risk_detection/train_risk_classifier.py
        ```
    * *Important Note:* Model training, especially for the deep learning (LSTM) models and potentially exhaustive searches by `auto_arima` or hyperparameter tuning for regression models (if implemented), can be computationally intensive and may take a significant amount of time depending on your hardware.

5.  **Run the Streamlit Application:**
    To launch the SCIP web application:
    ```bash
    streamlit run app/main_app.py
    ```
    For potentially more stable execution during development, especially if encountering issues related to live file watching by Streamlit (which can sometimes conflict with complex libraries like TensorFlow/PyTorch):
    ```bash
    streamlit run app/main_app.py --server.fileWatcherType none
    ```
    After running the command, the application will typically open automatically in your default web browser. If not, the terminal output will provide a local URL (usually `http://localhost:8501`) to access the application.

---

## 9. Key Demonstrations & Outputs (via Streamlit App)

The interactive Streamlit application serves as the primary interface to explore the platform's capabilities and insights:

* **Overview Page (`introduction_page.py`):** Provides a welcoming, high-level introduction to the SCIP platform, outlining its vision, the different analytical modules, and the key business problems it aims to solve within the supply chain domain.
* **Data Insights Page (`data_exploration_page.py`):** Offers a comprehensive Exploratory Data Analysis (EDA) of the Walmart sales data. Users can interactively explore historical sales trends, seasonality patterns, the impact of holidays, distributions of various features (e.g., Temperature, Fuel Price, CPI, Unemployment), and bivariate analyses showing correlations between these features and weekly sales.
* **Model Performance Page (`model_performance_page.py`):** Presents a comparative dashboard of all trained demand forecasting models. It showcases aggregated performance metrics (MAE, RMSE, MAPE) in tabular format and through visualizations like bar charts (average RMSE/MAPE by model family) and box plots (distribution of metrics per model family), enabling users to identify top-performing models.
* **Forecast Explorer Page (`forecast_explorer_page.py`):** A powerful and interactive tool for in-depth exploration of individual model forecasts. Users can:
    * Select specific Store-Department combinations.
    * Choose any trained forecasting model from the logged experiments.
    * Visualize actual historical sales alongside the model's predictions on the held-out test set and the actual test set values for direct comparison.
    * Generate and visualize **new, multi-step future forecasts** for a user-defined horizon (e.g., next 12, 26, 52 weeks). This page demonstrates iterative forecasting techniques for regression and LSTM models.
    * Inspect the specific hyperparameters and configuration used for the selected model run.
* **Inventory Strategy Page (`inventory_optimization_page.py`):** An interactive module enabling users to:
    * Calculate optimal inventory parameters (EOQ, Safety Stock, Reorder Point) by either leveraging demand forecasts (average demand and RMSE for demand standard deviation) from Module 1 or by manually inputting these values along with cost and lead time parameters.
    * Perform **inventory policy simulations** to visualize how inventory levels would behave over time under the calculated (or custom) parameters and different demand scenarios (e.g., constant average demand or actual historical demand).
    * Conduct **sensitivity analysis** to understand how changes in key input parameters (like holding cost, order cost, or service level) affect the optimal EOQ, Safety Stock, and ROP.
* **Risk Alerts Page (`risk_detection_page.py`):** Demonstrates a proof-of-concept for proactive risk detection. It displays a feed of sample news items that have been processed and enriched with:
    * **AI-driven VADER sentiment analysis** (Positive, Negative, Neutral with scores).
    * **AI-predicted risk categories** (e.g., "Logistics", "Natural Disaster") from a custom TF-IDF + Naive Bayes model.
    * A **dynamically generated AI risk score** based on the predicted category and sentiment.
    * Users can filter and sort these alerts by keywords, category, sentiment, or risk score to focus on the most critical signals.

*(Consider adding 2-3 key screenshots or a short GIF of your application here to visually showcase the UI and key features. For example, a screenshot of the Forecast Explorer plot, the Inventory Simulation, or the Risk Alerts feed.)*
---

## 10. Business Applications & Value

This SCIP platform, even in its demonstration phase, robustly highlights several key business applications and delivers tangible value propositions:

* **Improved Demand Planning & Forecasting Accuracy:** By employing and systematically evaluating a diverse suite of forecasting models, businesses can achieve more reliable and nuanced demand predictions. This leads to better alignment of supply with actual customer needs, thereby reducing the significant costs associated with forecasting errors (e.g., lost sales due to stockouts, excess inventory holding costs).
* **Optimized Inventory Management & Working Capital:** Data-driven calculation and simulation of EOQ, Safety Stock, and Reorder Points help organizations minimize overall inventory holding costs, reduce the financial risk of product obsolescence or spoilage (especially for perishable goods), and prevent costly stockouts. This directly improves working capital efficiency and cash flow.
* **Enhanced Supply Chain Resilience & Agility:** The proactive risk detection module, by identifying, categorizing, and scoring potential disruptions from external data sources, allows businesses to anticipate challenges (e.g., supplier defaults, port congestions, geopolitical instability) and implement mitigation strategies more quickly. This makes the supply chain more robust and adaptable to unforeseen events.
* **Data-Driven Strategic & Operational Decision Support:** The platform provides managers and planners with quantitative insights and interactive tools for setting informed inventory policies, understanding forecast uncertainties, evaluating the efficacy of different forecasting models, and assessing potential operational risks. This fosters a culture of data-informed decision-making across the supply chain.
* **Reduced Waste & Improved Resource Allocation:** More accurate forecasting and optimized inventory control directly contribute to minimizing overstocking, particularly for products with short lifecycles or those prone to obsolescence. This leads to reduced waste (physical and financial) and a more efficient allocation of valuable resources (capital, warehouse space, labor).
* **Cross-functional Collaboration:** By providing a common platform with shared data and insights, SCIP can facilitate better communication and collaboration between different departments such as sales, marketing, operations, and finance.

---

## 11. Limitations

* **Sample Data for Risk Module:** The current risk detection capabilities are demonstrated using a small, static, and manually curated sample of news data. A production-grade system would necessitate integration with real-time, diverse, and high-volume news feeds (e.g., APIs like NewsAPI, GDELT, or specialized supply chain risk intelligence services) and would require a significantly larger and continuously evolving dataset for training more robust and nuanced NLP models.
* **Forecasting of Exogenous Variables:** For forecasting models that utilize external regressors (e.g., CPI, Fuel Price, macroeconomic indicators), the future forecasts currently rely on simplistic assumptions for the future values of these regressors (e.g., last known value carry-forward). Achieving higher accuracy in a production setting would require dedicated, advanced forecasting of these exogenous variables themselves.
* **Model Generalizability and Maintenance:** The forecasting and risk models are trained on specific datasets (Walmart sales data, sample news). Their direct applicability and performance on other datasets, product categories, or different business contexts would require substantial retraining, fine-tuning, and ongoing model performance monitoring and maintenance.
* **No Real-time Data Integration or Streaming Analytics:** The platform, in its current demonstration form, operates on static datasets. Real-world deployment would necessitate the development of robust data pipelines for continuous or batch data ingestion, processing, and model retraining to adapt to evolving patterns and market conditions.
* **Simulated MLOps Practices:** While the project incorporates MLOps-like practices such as structured experiment logging and systematic saving of model artifacts, a full-fledged MLOps pipeline (encompassing automated retraining triggers, CI/CD for models, continuous model monitoring in production, data and model versioning) is beyond the scope of this demonstration project.
* **Scalability for Very Large Datasets:** While Pandas and other libraries used are efficient, extremely large datasets (terabytes) might require distributed computing solutions (e.g., Spark) for feature engineering and model training, which are not implemented here.

---

## 12. Future Work & Potential Enhancements

This platform serves as a strong foundation. Future development could focus on:

* **Risk Detection Module (Module 3) Enhancements:**
    * Integrate real-time news APIs (e.g., NewsAPI, GDELT) and other relevant data feeds (e.g., social media, weather services, shipping alerts) for dynamic and continuous risk signal ingestion.
    * Develop more sophisticated NLP models for risk classification, named entity recognition (NER) to extract specific entities like locations or company names, and event extraction using Transformer-based architectures (e.g., BERT, RoBERTa, GPT variants).
    * Implement a more advanced, potentially machine-learning-based, risk scoring model that incorporates multiple factors, severity levels, and probabilities.
    * Add interactive map visualizations for geographically relevant risks (e.g., natural disasters, port congestion, geopolitical hotspots).
    * Develop an alerting and notification system for high-priority risks.
* **New Module: Supplier Performance Analytics & Reliability Forecasting:**
    * Develop models to score supplier reliability and performance based on historical data (e.g., on-time delivery rates, quality defect rates, compliance records).
    * Forecast potential supplier-specific disruptions or performance degradation.
    * Integrate supplier risk into overall supply chain risk assessment.
* **New Module: Overall Platform Integration & Executive Dashboard:**
    * Create a unified executive dashboard summarizing key performance indicators (KPIs), forecasts, inventory health, and critical risk alerts across all modules.
    * Develop a "what-if" scenario planning tool that allows users to simulate the impact of different forecast scenarios, inventory policies, and potential risk events on key supply chain metrics.
* **Advanced Forecasting & Inventory Techniques:**
    * Explore and implement hierarchical forecasting methods (e.g., top-down, bottom-up, middle-out) for products, locations, or other relevant dimensions to ensure forecast consistency.
    * Implement probabilistic forecasting techniques (e.g., Quantile Regression, Bayesian methods, MC Dropout for LSTMs) to better quantify forecast uncertainty, which can then be used for more sophisticated safety stock calculations (e.g., based on target CSL directly from forecast distribution).
    * Investigate and implement more complex inventory optimization algorithms, such as multi-echelon inventory optimization (MEIO) or stochastic inventory models that explicitly handle uncertainty.
* **Full MLOps Integration & Automation:**
    * Implement robust MLOps pipelines using tools like Kubeflow, MLflow, or cloud-specific services for automated model retraining, deployment, version control (data, code, models), and continuous model performance monitoring in a simulated production environment.
* **User Authentication, Personalization & Scalability:**
    * Add user authentication and role-based access control for a secure multi-user environment.
    * Allow for personalization of dashboards and reports.
    * Explore options for scaling data processing and model training for larger datasets (e.g., Dask, Spark).

---

## 13. Developer

* **Name:** Ramesh Shrestha
* **Email:** shrestha.ramesh000@gmail.com
* **LinkedIn:** [linkedin.com/in/rameshsta](https://linkedin.com/in/rameshsta)
* **GitHub Project Repository:** `[YOUR_PROJECT_GITHUB_URL_HERE]` *(Please replace this with the actual URL if you host it)*

---

## 14. License

This project is licensed under the MIT License.