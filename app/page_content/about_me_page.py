"""
Renders the "About Me" page for the Streamlit application.

This page displays personal information, contact details, a summary,
and career aspirations, along with a profile image. It includes logic
to locate project assets like images.
"""
import streamlit as st
import os
import base64

def _get_project_root() -> str:
    """
    Determines the project root directory for asset loading.

    This utility function assumes a specific project structure where this file
    (`about_me_page.py`) is located at:
    `PROJECT_ROOT/app/page_content/about_me_page.py`
    It navigates three levels up from the current file's directory to find
    the project root.

    Returns:
        str: The absolute path to the project root directory.

    Side Effects:
        Issues a Streamlit warning (`st.warning`) if `__file__` is not defined
        (an uncommon scenario in standard module execution) and it has to
        fall back to using the current working directory. This fallback may
        affect asset loading if paths are not as expected.
    """
    try:
        # Navigate three levels up: page_content -> app -> PROJECT_ROOT
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_path = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
        return project_root_path
    except NameError:
        # __file__ is not defined. This can occur in some specialized execution
        # contexts (e.g., when code is run via exec() or in some interactive shells).
        # In standard module imports or script executions, __file__ is available.
        project_root_fallback = os.getcwd()
        st.warning(
            f"Could not determine project root using __file__ (it was undefined). "
            f"Falling back to current working directory: {project_root_fallback}. "
            "Assets (like the profile image) may not load correctly if the expected "
            "'app/assets/' path is not relative to this directory."
        )
        return project_root_fallback

def render_about_me_page():
    """
    Sets up and displays the content for the 'About Me' page.

    This includes a profile image, contact information, social media links,
    a professional summary, and career aspirations. Assets like the profile
    image are loaded relative to the project root, determined by the
    `_get_project_root` helper function.
    """
    project_root = _get_project_root()
    assets_dir = os.path.join(project_root, "app", "assets") # Standardized assets path
    profile_image_filename = "ramesh.jpeg"
    profile_image_path = os.path.join(assets_dir, profile_image_filename)

    st.header("👋 About Me & My Aspirations")
    st.markdown("---")

    col_img, col_text = st.columns([1, 2.5], gap="large")

    with col_img:
        if os.path.exists(profile_image_path):
            try:
                with open(profile_image_path, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode()

                # CSS for styling the profile image (round, border, shadow)
                st.markdown(f"""
                    <style>
                        .profile-image-container {{
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            padding-top: 10px;
                            padding-bottom: 20px;
                        }}
                        .profile-image-container img {{
                            width: 200px;
                            height: 200px;
                            border-radius: 50%;
                            object-fit: cover;
                            border: 4px solid #E8F0FE; /* Light blue border, theme-consistent */
                            box-shadow: 0 6px 12px rgba(0,0,0,0.15), 0 8px 24px rgba(0,0,0,0.1);
                        }}
                    </style>
                    <div class="profile-image-container">
                        <img src="data:image/jpeg;base64,{b64_string}" alt="Ramesh Shrestha - Profile">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error loading profile image: {e}")
                st.markdown(
                    f"<p style='text-align:center; color:red;'><i>Could not load profile photo. "
                    f"Ensure '{profile_image_filename}' is in '{assets_dir}'.</i></p>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                f"<p style='text-align:center; color:orange;'><i>Profile photo "
                f"(`{profile_image_filename}`) not found in `{assets_dir}`. "
                f"Please add it for a complete profile.</i></p>",
                unsafe_allow_html=True
            )
            # Fallback placeholder if the image is not found
            st.markdown(
                """
                <div style='width:200px; height:200px; border-radius:50%; background-color:#e0e0e0;
                            display:flex; justify-content:center; align-items:center; margin:auto;
                            margin-bottom:20px; border: 4px solid #E8F0FE;'>
                    <span style='font-size:50px;'>RS</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_text:
        st.subheader("Ramesh Shrestha")
        st.markdown("##### Aspiring Data Scientist | NLP & Machine Learning Enthusiast")
        st.markdown("---")
        st.markdown("📧 `shrestha.ramesh000@gmail.com`")
        st.markdown("📞 `+61 452 083 046`") # Using a standard format for phone number
        # Links with icons for LinkedIn and GitHub
        st.markdown(
            """
            <a href="https://linkedin.com/in/rameshsta" target="_blank" style="text-decoration: none; margin-right: 15px; color: #0077B5;">
                <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" alt="LinkedIn" width="24" height="24" style="vertical-align:middle; margin-right:5px;">LinkedIn
            </a>
            <a href="https://github.com/RameshSTA" target="_blank" style="text-decoration: none; color: #333;">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" width="24" height="24" style="vertical-align:middle; margin-right:5px;">GitHub
            </a>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Summary")
    # Using st.info or a custom div for better visual separation of the summary
    st.markdown("""
    <div style="background-color: #f9f9f9; border-left: 5px solid #4285F4; padding: 15px 20px; border-radius: 5px; margin-bottom:20px; font-size: 0.95em; line-height: 1.6;">
    I am a results-oriented and aspiring Data Scientist with a strong foundation in machine learning, Natural Language Processing (NLP), and MLOps, currently based in Sydney. My practical experience includes developing scalable data pipelines, deploying models via REST APIs, and applying advanced statistical techniques to derive actionable business insights. I have contributed to projects in customer analytics, recommendation systems, and workforce intelligence, complemented by real-world experience in the retail sector at Woolworths and technical internships. I am passionate about solving complex, data-centric problems and thrive in collaborative, Agile environments focused on delivering measurable impact.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Career Aspirations & Goals")
    st.markdown("""
    <div style="background-color: #f9f9f9; border-left: 5px solid #34A853; padding: 15px 20px; border-radius: 5px; font-size: 0.95em; line-height: 1.6;">
    My primary goal is to leverage my skills in data science and AI to contribute to innovative solutions that drive business growth and efficiency. I am particularly interested in roles where I can:
    <ul>
        <li style="margin-bottom: 8px;">Apply and expand my expertise in NLP and machine learning to extract meaningful insights and create tangible value from complex datasets.</li>
        <li style="margin-bottom: 8px;">Collaborate within dynamic, cross-functional teams to tackle challenging problems and learn from experienced professionals in the field.</li>
        <li style="margin-bottom: 8px;">Continuously develop my knowledge in cutting-edge AI technologies, including deep learning architectures, MLOps practices, and cloud-based data solutions.</li>
        <li style="margin-bottom: 8px;">Contribute to a company culture that champions innovation, data-driven decision-making, and impactful outcomes.</li>
    </ul>
    I am eager to bring my dedication, analytical mindset, and proactive learning approach to a challenging role where I can make significant contributions to an organization's success.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # This block allows for standalone testing of this page.
    # For assets to load correctly when run standalone, ensure this script
    # is executed from a context where the relative path to 'app/assets/' is valid,
    # or that _get_project_root() correctly infers/falls back to a usable path.
    st.set_page_config(layout="wide")
    render_about_me_page()