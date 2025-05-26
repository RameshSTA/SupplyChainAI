"""
Renders the introduction and overview page for the SupplyChain AI platform.

This page serves as the landing page, outlining the platform's vision,
the business problems it aims to address, and the core functionalities
demonstrated, particularly in demand forecasting and inventory strategy.
"""
import streamlit as st

def render_introduction_page():
    """
    Renders a professional and detailed introduction to the SupplyChainAI platform,
    with all content visible by default.

    This page articulates the platform's strategic vision, addresses critical
    business challenges, quantifies its value proposition, details core
    functionalities, outlines future potential, and guides businesses on
    integrating AI for competitive advantage.
    """

    st.header("🚀 SupplyChainAI: Vision & Strategic Imperative")
    st.markdown("""
    In an era of unprecedented global market dynamism, achieving a **hyper-responsive, intelligently orchestrated, and data-centric supply chain** is paramount for sustained competitive dominance. 
    **SupplyChainAI** represents a paradigm shift, designed to elevate traditional supply chain paradigms into sophisticated, predictive, and self-optimising operational ecosystems.

    Our core mission is to compellingly demonstrate how the strategic application of **Artificial Intelligence (AI)**—initially focused on **Advanced Demand Forecasting** and **Intelligent Inventory Optimisation**—empowers organisations to:

    * Realise substantial **reductions in operational and inventory holding costs.**
    * Achieve transformative **gains in end-to-end operational efficiency and throughput.**
    * Cultivate robust **organisational resilience against market disruptions and demand volatility.**
    * Secure a distinct and sustainable **competitive advantage in complex global markets.**
    """)

    st.markdown("---") # Visual separator

    st.subheader("🎯 Key Business Challenges Addressed")
    st.markdown("""
    SupplyChainAI is engineered to directly confront and mitigate pervasive challenges that impede enterprise growth, constrict profitability, and diminish market responsiveness:

    * **Pervasive Demand Uncertainty:** Overcoming the limitations of conventional forecasting in highly volatile environments, thereby minimising the financial and reputational impact of stockouts and excess inventory.
    * **Suboptimal Capital Allocation:** Addressing excessive working capital tied up in inventory, reducing holding costs, and mitigating risks of product obsolescence or spoilage.
    * **Fragmented Operational Visibility & Agility:** Countering the effects of siloed data and reactive decision-making to enhance the capacity for proactive adaptation to unforeseen global disruptions (e.g., geopolitical instability, climatic events, supplier inconsistencies).
    * **Escalating Competitive & Margin Pressures:** Providing crucial optimisation levers for businesses striving to enhance service delivery, reduce operational friction, and protect margins in fiercely competitive arenas.
    """)

    st.markdown("---") # Visual separator

    st.subheader("📈 Quantifiable Business Impact & Value Proposition")
    st.markdown("""
    By architecting AI-driven solutions to these challenges, SupplyChainAI offers a clear trajectory towards significant and measurable business outcomes:

    * **Enhanced Financial Performance:** Driving profitability through strategic cost containment (inventory, logistics, operations) and revenue protection against stockouts.
    * **Elevated Customer Experience & Brand Loyalty:** Ensuring superior product availability, improving delivery reliability, and fostering more responsive customer engagement.
    * **Optimised Operational Throughput:** Streamlining complex planning cycles, enabling intelligent automation in decision-support, and maximising resource utilisation across the entire value network.
    * **Fortified Risk Management & Business Continuity:** Instilling the foresight to anticipate potential vulnerabilities and the agility to orchestrate effective responses, thereby safeguarding revenue and market standing.
    * **Data-Driven Strategic Superiority:** Converting vast datasets into actionable, predictive intelligence that underpins superior strategic planning and tactical execution, essential for industry leadership.
    """)
    
    st.markdown("---") # Visual separator

    st.subheader("🛠️ Core Platform Functionalities: A Demonstration")
    st.markdown("""
    This platform provides a practical demonstration of applied AI through the following key, interconnected modules:

    **1. Advanced Demand Forecasting Engine:**
    * In-depth analysis of historical sales data (exemplified using the Walmart Sales dataset) to discern complex trends, seasonality, and causal factors.
    * Rigorous development, training, and comparative evaluation of a diverse portfolio of forecasting methodologies:
        * *Classical Time Series Models:* Naive, Seasonal Naive, ETS (Error, Trend, Seasonality), SARIMA (Seasonal Autoregressive Integrated Moving Average), Prophet.
        * *Machine Learning Regression Algorithms:* Random Forest, XGBoost, LightGBM.
        * *Deep Learning Sequential Models:* LSTM (Long Short-Term Memory) networks.
    * Interactive dashboards facilitating granular forecast exploration, comprehensive model performance assessment (MAE, RMSE, MAPE), and insightful error analysis.

    **2. Intelligent Inventory Strategy & Financial Optimisation:**
    * Algorithmic calculation of critical inventory control parameters: Economic Order Quantity (EOQ), Safety Stock (SS), and Reorder Point (ROP), tailored to specific business objectives.
    * Dynamic integration of demand forecast distributions (mean and variance) to continuously refine and optimise inventory parameters against target service levels and cost structures.
    * Sophisticated simulation capabilities to project inventory dynamics, service level achievements, and total cost of ownership under diverse operational policies and demand scenarios.
    * Comprehensive sensitivity analysis tools to elucidate the financial and operational impact of variations in key inputs such as holding costs, order costs, lead times, or service level commitments.
    """)
    
    st.markdown("---") # Visual separator

    st.subheader("🔮 Future Vision & Strategic Capability Expansion")
    st.markdown("""
    The architectural ambition for SupplyChainAI extends beyond its current focus, envisioning a holistic, intelligent operations nerve centre. Conceptual future enhancements include:

    * **Predictive Risk & Resilience Orchestration:** Employing advanced NLP and anomaly detection for early warning of multi-echelon supply chain vulnerabilities; incorporating simulation for robust mitigation strategy formulation and enhanced business continuity.
    * **Cognitive Supplier Intelligence & Collaboration:** AI-augmented supplier segmentation, dynamic performance monitoring (across quality, reliability, cost, ESG metrics), and tools for collaborative forecasting and optimised procurement strategies.
    * **Adaptive Logistics & Network Optimisation:** Real-time, AI-driven optimisation of logistics routes and modes, predictive transportation management, and strategic network design balancing cost, velocity, and sustainability.
    * **Sustainable & Circular Supply Chain Analytics:** Tools for tracking and optimising environmental impact, waste reduction, and adherence to circular economy principles.
    * **Integrated Business Planning (IBP) with AI:** Seamlessly connecting strategic, tactical, and operational planning cycles, powered by AI for more accurate and agile enterprise-wide decision-making.
    """)

    st.markdown("---") # Visual separator
    
    st.subheader("🌍 Integrating AI for Enduring Global Competitiveness")
    st.markdown("""
    Transitioning to AI-augmented supply chain management is a strategic imperative for sustained global leadership. Organisations can embark on this transformative journey by:

    1.  **Fortifying the Data Ecosystem:** Championing data governance, quality, and seamless integration across enterprise systems as the immutable foundation for impactful AI.
    2.  **Targeting High-Value Pilot Initiatives:** Commencing with focused use cases (e.g., critical product demand forecasting, high-value inventory optimisation) to demonstrate tangible ROI and foster organisational buy-in.
    3.  **Cultivating Cross-Functional Synergy:** Dismantling operational silos to ensure that AI-derived insights are collaboratively leveraged across sales, marketing, operations, and finance for holistic benefit.
    4.  **Nurturing AI Talent & Data Acumen:** Investing in specialised AI talent and concurrently elevating data literacy across all organisational levels.
    5.  **Adopting Agile, Iterative Deployment:** Implementing AI solutions through an agile framework that allows for continuous model refinement, adaptation to new data streams, and responsiveness to evolving business landscapes.
    6.  **Prioritising Ethical AI & Change Management:** Implementing AI solutions responsibly, ensuring transparency, fairness, and accountability. Manage the human aspect of technological change effectively to ensure adoption and trust.
    7.  **Scaling & Integrating Strategically:** Once value is proven, strategically scale successful AI solutions across the broader supply chain network and integrate them into core business processes and decision-making workflows.

    By strategically embedding AI, businesses can transform their supply chains from cost centres into powerful engines for growth, innovation, and sustained global leadership.
    """)
            
    st.info(
        "**Navigate through the platform modules using the sidebar to explore these AI-driven supply chain solutions in detail.**",
        icon="🧭"
    )

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="SupplyChainAI - Strategic Overview", page_icon="🔗")
    render_introduction_page()