"""
Renders the introduction and overview page for the SupplyChain AI platform.

This page serves as the landing page, outlining the platform's vision,
the business problems it aims to address, and the core functionalities
demonstrated, particularly in demand forecasting and inventory strategy.
"""
import streamlit as st

def render_introduction_page():
    """
    Displays the content for the introduction/overview page.

    This includes sections on the platform's vision, key business problems
    tackled, core functionalities showcased (with a focus on demand forecasting
    and inventory strategy), and a note on future conceptual functionalities.
    """
    # The main title for the application is typically handled in the main app script.
    # If this page were to have its own sub-title, it could be added here.
    # e.g., st.title("🌟 Project Overview")

    st.header("Platform Vision & Goals")
    st.markdown("""
    This platform demonstrates an AI-powered approach to enhance supply chain operations,
    with an initial focus on **Advanced Demand Forecasting** and **Intelligent Inventory Strategy**.
    Our goal is to showcase how data-driven insights can lead to significant cost reductions,
    improved efficiency, and increased resilience in supply chain management.
    """)

    with st.expander("Key Business Problems Addressed", expanded=True):
        st.markdown("""
        - **Inaccurate Demand Forecasting:** Leading to stockouts or overstocking, impacting revenue and customer satisfaction.
        - **High Inventory Holding Costs:** Tying up valuable capital and increasing the risk of product obsolescence or spoilage.
        - **Inefficient Supplier Management:** Potentially leading to unreliable supply, inconsistent quality, and suboptimal pricing.
        - **Supply Chain Disruptions:** Difficulty in adapting and responding to unforeseen events (e.g., geopolitical issues, natural disasters, pandemics).
        """)

    with st.expander("Core Functionalities Demonstrated", expanded=True):
        st.markdown("""
        **1. Demand Forecasting Insights:**
        * Analysis of historical sales data (utilizing a Walmart Sales Forecasting dataset as an example).
        * Development and comparative analysis of various forecasting models:
            * Classical Time Series Methods: Naive, Seasonal Naive, ETS (Error, Trend, Seasonality), SARIMA (Seasonal Autoregressive Integrated Moving Average), Prophet.
            * Machine Learning Regression Techniques: Random Forest, XGBoost, LightGBM.
            * Deep Learning Architectures: LSTM (Long Short-Term Memory) networks.
        * Interactive exploration of generated forecasts and detailed model performance metrics.

        **2. Inventory Strategy & Optimization:**
        * Calculation of fundamental inventory parameters: Economic Order Quantity (EOQ), Safety Stock (SS), and Reorder Point (ROP).
        * Integration of demand forecasts to dynamically inform and adjust inventory parameters.
        * Simulation of different inventory policies to visualize their performance and cost implications.
        * Sensitivity analysis to understand how changes in inventory parameters affect outcomes.

        **(Future Functionalities - Conceptual):**
        * Proactive Supply Chain Risk Detection & Mitigation Planning.
        * Supplier Performance Analytics & Scorecarding.
        * Network Optimization & Logistics Planning.
        """)

    st.success(
        "Use the navigation bar at the top to explore different sections of the platform "
        "and delve into specific functionalities."
    )

if __name__ == "__main__":
    # This block allows for basic standalone testing or viewing of this page's content.
    # In a multi-page Streamlit app, this page is typically imported and rendered by a main app script.
    st.set_page_config(layout="centered", page_title="Introduction") # Basic config for test
    render_introduction_page()