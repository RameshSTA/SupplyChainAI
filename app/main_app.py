"""
Main Streamlit application file for the SupplyChain AI Dashboard.

This script sets up and runs a multi-page Streamlit application
with a custom fixed header, navigation, and footer. It defines
the overall page structure, styling, and navigation logic between
different views of the dashboard.
"""
import streamlit as st
import os
import sys
from streamlit_option_menu import option_menu
# import base64 # Uncomment if needed for other image/SVG handling

# --- Mock Page Rendering Functions ---
# These functions are placeholders for actual page content.
# In a full application, they would typically be imported from separate modules.

def render_introduction_page():
    """Renders the overview page of the dashboard."""
    st.header("Overview Page")
    st.write("Welcome to the SupplyChain AI Dashboard.")
    # Example content to demonstrate scrolling
    for i in range(60):
        st.write(f"This is line {i+1} of content to test scrolling properly. "
                 f"Lorem ipsum dolor sit amet, consectetur adipiscing elit.")

def render_data_exploration_page():
    """Renders the data insights page."""
    st.header("Data Insights Page")

def render_model_performance_page():
    """Renders the model performance evaluation page."""
    st.header("Model Performance Page")

def render_forecast_explorer_page():
    """Renders the forecast exploration page."""
    st.header("Forecast Explorer Page")

def render_inventory_optimization_page():
    """Renders the inventory strategy and optimization page."""
    st.header("Inventory Strategy Page")

def render_risk_detection_page():
    """Renders the risk detection and alerts page."""
    st.header("Risk Alert Page")

def render_about_me_page():
    """Renders the 'About Me' page."""
    st.header("About Me Page")

# --- Project Path Setup & Page Imports ---
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    # Assuming 'app/page_content' structure for page modules
    PAGE_CONTENT_PATH = os.path.join(PROJECT_ROOT, 'app', 'page_content')
    SRC_PATH = os.path.join(PROJECT_ROOT, 'src') # General source directory

    if PAGE_CONTENT_PATH not in sys.path:
        sys.path.insert(0, PAGE_CONTENT_PATH)
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)
    if PROJECT_ROOT not in sys.path: # Add project root itself if needed
        sys.path.insert(0, PROJECT_ROOT)

    # Attempt to import actual page rendering functions
    from introduction_page import render_introduction_page
    from data_exploration_page import render_data_exploration_page
    from model_performance_page import render_model_performance_page
    from forecast_explorer_page import render_forecast_explorer_page
    from inventory_optimization_page import render_inventory_optimization_page
    from risk_detection_page import render_risk_detection_page
    from about_me_page import render_about_me_page
except Exception as e:
    st.warning(f"Using mock pages due to import/path issue: {e}")
    # If imports fail, the mock functions defined above in this file will be used.
    pass

# --- Page Configuration ---
st.set_page_config(
    page_title="SupplyChain AI",
    page_icon="🔗📈", # Can be an emoji or a URL to an image
    layout="wide",
    initial_sidebar_state="collapsed", # "auto", "expanded", "collapsed"
)

# --- Custom SVG Logo ---
LOGO_SVG = """
<svg width="45" height="45" viewBox="0 0 110 110" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="110" height="110" rx="20" fill="#E8F0FE"/>
    <path d="M30 80L30 40C30 34.4772 34.4772 30 40 30H70C75.5228 30 80 34.4772 80 40V80" stroke="#4285F4" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M55 80V55" stroke="#34A853" stroke-width="8" stroke-linecap="round"/>
    <circle cx="55" cy="45" r="7" fill="#FBBC05"/>
    <path d="M40 65H70" stroke="#EA4335" stroke-width="8" stroke-linecap="round"/>
</svg>
"""

# --- Page Definitions ---
# Defines the pages, their corresponding rendering functions, and icons for navigation.
PAGES = {
    "Overview": {"func": render_introduction_page, "icon": "house-door"},
    "Data Insights": {"func": render_data_exploration_page, "icon": "bar-chart-line"},
    "Performance": {"func": render_model_performance_page, "icon": "graph-up-arrow"},
    "Forecast": {"func": render_forecast_explorer_page, "icon": "calendar-check"},
    "Inventory": {"func": render_inventory_optimization_page, "icon": "box-seam"},
    "Risk Alert": {"func": render_risk_detection_page, "icon": "exclamation-triangle"},
    "About Me": {"func": render_about_me_page, "icon": "person-circle"},
}
PAGE_NAMES = list(PAGES.keys())
PAGE_ICONS = [PAGES[p]["icon"] for p in PAGE_NAMES]

# --- Global CSS Styling ---
# Defines constants for layout dimensions and color palette.
HEADER_TOP_ROW_HEIGHT_PX = 70
HEADER_BOTTOM_ROW_HEIGHT_PX = 60
HEADER_TOTAL_HEIGHT_PX = HEADER_TOP_ROW_HEIGHT_PX + HEADER_BOTTOM_ROW_HEIGHT_PX
FOOTER_HEIGHT_PX = 75

# Color Palette
HEADER_BG = "#FFFFFF"
FOOTER_BG = "#F8F9FA"
TEXT_DARK = "#212529"
TEXT_MEDIUM = "#6C757D"
PRIMARY_BLUE = "#4285F4"
BORDER_COLOR = "#DEE2E6"

# Injects custom CSS for styling the application layout and components.
# This CSS creates a fixed header and footer, and a scrollable content area.
# For larger CSS, consider moving to a separate style.css file and loading it.
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&display=swap');
        @import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.3/font/bootstrap-icons.css");

        /* --- Global & Body --- */
        body {{ font-family: 'Roboto', sans-serif; background-color: #FFFFFF; }}
        html, body {{ height: 100%; overflow: hidden; }} /* Prevents double scrollbars */

        /* --- Hide Streamlit Default UI Elements --- */
        header[data-testid="stHeader"],
        button[data-testid="stSidebarNav"],
        section[data-testid="stSidebar"] {{ display: none !important; }}

        /* --- Main Content Padding --- */
        .main .block-container {{ padding: 0 !important; }}

        /* --- Main App Wrapper (to enable flex layout for fixed header/footer) --- */
        .stApp > div:first-child > div:first-child > div:first-child {{
            display: flex !important;
            flex-direction: column !important;
            height: 100vh !important;
            overflow: hidden !important;
        }}

        /* --- Header Container (Fixed) --- */
        /* This is the first direct child div inside the flex container above */
        .stApp > div:first-child > div:first-child > div:first-child > div:nth-child(1) {{
            position: fixed !important; top: 0 !important; left: 0 !important; right: 0 !important;
            height: {HEADER_TOTAL_HEIGHT_PX}px !important; background-color: {HEADER_BG} !important;
            z-index: 999998 !important; padding: 0 !important;
            border-bottom: 1px solid {BORDER_COLOR} !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            display: flex !important; flex-direction: column !important;
            flex-shrink: 0 !important; /* Prevents header from shrinking */
        }}

        /* --- Content Container (Padded & Scrollable) --- */
        /* This is the second direct child div */
        .stApp > div:first-child > div:first-child > div:first-child > div:nth-child(2) {{
            margin-top: {HEADER_TOTAL_HEIGHT_PX}px !important;
            margin-bottom: {FOOTER_HEIGHT_PX}px !important;
            padding: 25px 2rem 30px 2rem !important; /* Top, R/L, Bottom */
            width: 100% !important; overflow-y: auto !important; overflow-x: hidden !important;
            flex-grow: 1 !important; /* Allows content to take available space */
        }}

        /* --- Footer Container (Fixed & Centered) --- */
        /* This is the third direct child div */
        .stApp > div:first-child > div:first-child > div:first-child > div:nth-child(3) {{
            position: fixed !important; bottom: 0 !important; left: 0 !important; right: 0 !important;
            height: {FOOTER_HEIGHT_PX}px !important; background-color: {FOOTER_BG} !important;
            border-top: 1px solid {BORDER_COLOR} !important; z-index: 999999 !important;
            display: flex !important; justify-content: center !important; align-items: center !important;
            padding: 0 1rem !important;
            flex-shrink: 0 !important; /* Prevents footer from shrinking */
            width: 100% !important;
        }}

        /* --- Header Row 1 (Title and Logo) & Row 2 (Navigation) --- */
        .header-row-1 {{ display: flex !important; align-items: center !important; height: {HEADER_TOP_ROW_HEIGHT_PX}px !important; width: 100% !important; padding: 0 2rem !important; }}
        .title-section {{ display: flex; align-items: center; }}
        .title-section .project-logo {{ margin-right: 1rem; }}
        .title-section h2 {{ font-family: 'Poppins', sans-serif; font-size: 1.9rem; font-weight: 600; color: {TEXT_DARK} !important; line-height: 1; white-space: nowrap; margin: 0; }}
        .title-section h2 span {{ font-weight: 700; color: {PRIMARY_BLUE}; }}

        .header-row-2 {{ display: flex !important; align-items: center !important; justify-content: center !important; height: {HEADER_BOTTOM_ROW_HEIGHT_PX}px !important; width: 100% !important; padding: 0 1rem !important; overflow: hidden; background-color: {HEADER_BG} !important; }}

        /* --- streamlit_option_menu styling --- */
        .header-row-2 nav.menu > ul {{ display: flex !important; flex-direction: row !important; align-items: center !important; justify-content: center !important; height: 100% !important; width: 100% !important; margin: 0 !important; padding: 0 !important; background-color: {HEADER_BG} !important; flex-wrap: nowrap !important; }}
        .header-row-2 nav.menu > ul > li {{ list-style: none !important; padding: 0 !important; margin: 0 5px !important; background-color: {HEADER_BG} !important; flex-shrink: 0 !important; }}
        .header-row-2 nav.menu > ul > li > a {{ display: flex !important; flex-direction: row !important; align-items: center !important; justify-content: center !important; text-decoration: none !important; color: {TEXT_MEDIUM} !important; font-weight: 500 !important; font-size: 0.95rem !important; padding: 8px 18px !important; border-radius: 6px !important; transition: color 0.2s ease, background-color 0.2s ease !important; border-bottom: 3px solid transparent !important; line-height: 1.3 !important; background-color: {HEADER_BG} !important; white-space: nowrap !important; text-align: center !important; }}
        .header-row-2 nav.menu > ul > li > a > i {{ margin-right: 8px !important; margin-bottom: 0 !important; font-size: 1.1rem !important; color: {TEXT_MEDIUM} !important; line-height: 1 !important; }}
        .header-row-2 nav.menu > ul > li > a:hover {{ color: {PRIMARY_BLUE} !important; background-color: #f1f3f4 !important; }}
        .header-row-2 nav.menu > ul > li > a:hover > i {{ color: {PRIMARY_BLUE} !important; }}
        .header-row-2 nav.menu > ul > li > a[aria-selected="true"] {{ color: {PRIMARY_BLUE} !important; background-color: #E8F0FE !important; border-bottom: 3px solid {PRIMARY_BLUE} !important; font-weight: 700 !important; }}
        .header-row-2 nav.menu > ul > li a[aria-selected="true"] > i {{ color: {PRIMARY_BLUE} !important; }}

        /* --- Footer Content Styling --- */
        .footer {{
            width: 100%;
            text-align: center;
        }}
        .footer-content {{
            color: {TEXT_MEDIUM};
            font-size: 0.85rem;
            line-height: 1.4;
        }}
        .footer-content p {{
            margin: 2px 0 !important; /* Reduced vertical margin for compactness */
        }}
        .footer-content a {{
            color: {PRIMARY_BLUE};
            text-decoration: none !important;
            margin: 0 8px; /* Space out the links */
            font-weight: 500;
        }}
        .footer-content a:hover {{
            text-decoration: underline !important;
            color: {TEXT_DARK};
        }}
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State for Current Page ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview" # Default page

# --- Define Layout Containers ---
# These containers correspond to the CSS-defined fixed/scrollable areas.
# They must be defined in the order they should appear in the DOM for CSS to target them correctly.
header_container = st.container()
content_container = st.container()
footer_container = st.container()

# --- Build Header ---
with header_container:
    st.markdown(
        f"""
        <div class="header-row-1">
            <div class="title-section">
                <div class="project-logo">{LOGO_SVG}</div>
                <h2>Supply<span>Chain AI</span></h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Navigation menu in the second row of the header
    st.markdown('<div class="header-row-2">', unsafe_allow_html=True)
    selected_page = option_menu(
        menu_title=None, # No title for the menu itself
        options=PAGE_NAMES,
        icons=PAGE_ICONS,
        menu_icon=None, # No global menu icon
        default_index=PAGE_NAMES.index(st.session_state.current_page),
        orientation="horizontal",
        styles={ # Custom styles for the option_menu
            "container": {"padding": "0!important", "background-color": "transparent"},
        }
    )
    st.markdown('</div>', unsafe_allow_html=True)

# --- Handle Page Navigation ---
# If a new page is selected in the menu, update session state and trigger a rerun.
if selected_page and selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    st.rerun()

# --- Build Content Area ---
with content_container:
    page_config = PAGES.get(st.session_state.current_page)
    if page_config and callable(page_config.get("func")):
        page_config["func"]() # Call the render function for the current page
    else:
        # Fallback if page is not found or function is not callable
        st.warning(f"Page '{st.session_state.current_page}' not found or invalid. Returning to Overview.")
        st.session_state.current_page = "Overview"
        overview_func = PAGES.get("Overview", {}).get("func")
        if callable(overview_func):
            overview_func()
        st.rerun() # Rerun to reflect the switch to the Overview page

# --- Build Footer ---
with footer_container:
    footer_html_content = f"""
    <div class="footer">
        <div class="footer-content">
            <p>Developed by Ramesh Shrestha</p>
            <p>&copy; 2024-2025 SupplyChain AI. All Rights Reserved.</p>
            <p>
                <a href="mailto:shrestha.ramesh000@gmail.com" target="_blank">Contact Us</a> |
                <a href="#" target="_blank">Privacy Policy</a> |
                <a href="#" target="_blank">Terms of Service</a>
            </p>
        </div>
    </div>
    """
    st.markdown(footer_html_content, unsafe_allow_html=True)

# --- Sidebar Troubleshooting Note ---
# Provides guidance for users if layout issues occur due to CSS overrides.
st.sidebar.warning(
    """
    **Layout Note:**
    This app uses custom CSS for a fixed header/footer layout.
    If you encounter display issues:
    1.  **Hard Refresh:** Press `Ctrl+Shift+R` (or `Cmd+Shift+R` on Mac).
    2.  **Inspect Elements:** Use browser developer tools to check CSS.
    3.  **Streamlit Updates:** Future Streamlit versions might affect this CSS.
    """
)