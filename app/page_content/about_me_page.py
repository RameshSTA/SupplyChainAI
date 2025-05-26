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

    Assumes this script (`about_me_page.py`) is located at:
    `PROJECT_ROOT/app/page_content/about_me_page.py`.
    Navigates up two levels to find the project root.

    Returns:
        str: The absolute path to the project root directory.
    """
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_path = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
        return project_root_path
    except NameError: # pragma: no cover
        project_root_fallback = os.getcwd()
        st.warning(
            f"Could not determine project root using `__file__`. "
            f"Falling back to CWD: {project_root_fallback}. Assets may not load correctly."
        )
        return project_root_fallback

def render_about_me_page():
    """
    Sets up and displays the content for the 'About Me' page.

    This includes a profile image, contact information, a professional summary,
    an outline of key experiences and certifications, and career aspirations,
    all tailored for a professional audience.
    """
    project_root = _get_project_root()
    assets_dir = os.path.join(project_root, "app", "assets")
    profile_image_filename = "ramesh.jpeg" # Ensure this image is in app/assets/
    profile_image_path = os.path.join(assets_dir, profile_image_filename)

    st.header("👋 Hi there, This is Ramesh Shrestha")
    st.markdown("---")

    col_img, col_text = st.columns([1, 2.8], gap="large")

    with col_img:
        if os.path.exists(profile_image_path):
            try:
                with open(profile_image_path, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode()
                st.markdown(f"""
                    <style>
                        .profile-image-container {{
                            display: flex; justify-content: center; align-items: flex-start;
                            padding-top: 20px; padding-bottom: 20px;
                        }}
                        .profile-image-container img {{
                            width: 220px; height: 220px; border-radius: 50%;
                            object-fit: cover; border: 5px solid #F0F2F6;
                            box-shadow: 0 8px 16px rgba(0,0,0,0.1), 0 10px 30px rgba(0,0,0,0.08);
                        }}
                    </style>
                    <div class="profile-image-container">
                        <img src="data:image/jpeg;base64,{b64_string}" alt="Ramesh Shrestha - Profile Photo">
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e: # pragma: no cover
                st.error(f"Error loading profile image: {e}")
        else: # pragma: no cover
            st.markdown(
                f"<p style='text-align:center; color:orange;'><i>Profile photo ('{profile_image_filename}') not found in `{assets_dir}`. Placeholder shown.</i></p>",
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <div style='width:220px; height:220px; border-radius:30%; background-color:#EAECEE;
                            display:flex; justify-content:center; align-items:center; margin: 20px auto;
                            border: 5px solid #F0F2F6; text-align: center;'>
                    <span style='font-size:60px; color: #5D6D7E; font-weight: bold;'>RS</span>
                </div>
                """, unsafe_allow_html=True)

    with col_text:
        st.subheader("Ramesh Shrestha")
        st.markdown("##### Aspiring Data Scientist & Machine Learning Engineer")
        st.markdown("Sydney, NSW, Australia")
        st.markdown("---")
        st.markdown("""
        <div style="line-height: 2.0;">
            <span style="font-size: 1.1em;">📧</span>&nbsp; <a href="mailto:shrestha.ramesh000@gmail.com" style="text-decoration:none;">shrestha.ramesh000@gmail.com</a><br>
            <span style="font-size: 1.1em;">📞</span>&nbsp; 0452 083 046<br>
            <a href="https://linkedin.com/in/rameshsta" target="_blank" style="text-decoration: none; margin-right: 15px; color: #0077B5;">
                <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" alt="LinkedIn" width="20" height="20" style="vertical-align:middle; margin-right:6px;">LinkedIn
            </a>
            <a href="https://github.com/RameshSTA" target="_blank" style="text-decoration: none; color: #333;">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" width="20" height="20" style="vertical-align:middle; margin-right:6px;">GitHub
            </a>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Professional Summary
    st.markdown("#### Professional Summary")
    st.markdown("""
    <div style="background-color: #F8F9F9; border-left: 6px solid #007bff; padding: 18px 22px; border-radius: 5px; margin-bottom:25px; font-size: 1.0em; line-height: 1.7;">
    A highly motivated and analytical professional, holding a Bachelor of Software Engineering with a Specialisation in AI, and currently advancing practical expertise through a Professional Year in IT. My core strength lies in translating complex data into actionable insights and developing end-to-end AI/ML solutions.
    <br><br>
    I possess a robust skill set in Python, SQL, and R, complemented by a deep understanding of machine learning algorithms (including predictive analytics, classification, and deep learning), Natural Language Processing (NLP), statistical methods, and data visualisation techniques. This is further solidified by hands-on internship experience involving data analysis and the application of machine learning models. My project portfolio, particularly "SupplyChainAI" and "E-Commerce Feedback Mining," demonstrates a capacity for independent development and deployment of interactive, data-driven applications using tools like Streamlit.
    <br><br>
    I am committed to continuous professional development, as evidenced by multiple industry-recognised certifications, and am passionate about contributing to innovative projects within collaborative, agile environments.
    </div>
    """, unsafe_allow_html=True)

    # Key Credentials & Experience Highlights
    st.markdown("#### Key Credentials & Experience Highlights")
    st.markdown("""
    <div style="background-color: #F0F8FF; border-left: 6px solid #17A2B8; padding: 18px 22px; border-radius: 5px; margin-bottom:25px; font-size: 1.0em; line-height: 1.7;">
    My journey into data science and AI is built upon a strong academic foundation, enhanced by specialised training and practical application:
    <ul>
        <li style="margin-bottom: 10px;"><strong>Academic Foundation:</strong> Bachelor of Software Engineering (Specialisation in Artificial Intelligence) from Torrens University Australia, providing comprehensive knowledge in software development principles and AI concepts. Currently augmenting this with a Professional Year in IT at QIBA.</li>
        <li style="margin-bottom: 10px;"><strong>Specialised Certifications:</strong> Actively pursued advanced credentials to deepen expertise, including the <em>Google Advanced Data Analytics Professional Certificate</em>, the <em>Deep Learning Specialization (DeepLearning.AI)</em>, and the <em>Machine Learning Specialization (Stanford University / DeepLearning.AI)</em>. Also completed a practical <em>Software Engineering Job Simulation with JPMorgan Chase & Co. (Forage)</em>.</li>
        <li style="margin-bottom: 10px;"><strong>Data Analysis & ML Internship (Hightech Mastermind Pty Ltd):</strong> Gained valuable hands-on experience in a professional setting, contributing to data solution design, assisting in the development and validation of machine learning models, performing exploratory data analysis, and automating analytical reporting.</li>
        <li style="margin-bottom: 10px;"><strong>End-to-End AI Project Development:</strong> Independently conceptualised, developed, and deployed AI-powered applications such as "SupplyChainAI" (for demand forecasting, inventory optimisation, and risk alerting) and "E-Commerce Feedback Mining" (for sentiment analysis and topic modelling). These projects showcase proficiency in Python, NLP, forecasting techniques, MLOps fundamentals, and creating interactive dashboards with Streamlit.</li>
        <li style="margin-bottom: 10px;"><strong>Mentorship & Communication:</strong> Honed communication and leadership skills by mentoring participants at Business Analyst and Data Analyst Hackathons. Developed strong interpersonal and problem-solving abilities through customer-facing roles (e.g., at Woolworths), crucial for effective stakeholder engagement.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


    # Career Aspirations & Goals
    st.markdown("#### Career Aspirations & Professional Goals")
    st.markdown("""
    <div style="background-color: #F8F9F9; border-left: 6px solid #28a745; padding: 18px 22px; border_radius: 5px; font-size: 1.0em; line-height: 1.7;">
    My primary career objective is to secure a challenging and impactful role as a Data Scientist or Machine Learning Engineer where I can leverage my technical skills and analytical mindset to contribute to innovative, data-driven solutions. I aim to:
    <ul>
        <li style="margin-bottom: 10px;">Apply and continuously expand my expertise in machine learning, deep learning, and NLP to extract meaningful insights, build predictive models, and create tangible business value from complex datasets.</li>
        <li style="margin-bottom: 10px;">Thrive within dynamic, cross-functional teams, collaborating effectively to tackle ambitious projects, solve intricate problems, and learn from seasoned professionals in the AI and data science domain.</li>
        <li style="margin-bottom: 10px;">Actively contribute to the full lifecycle of machine learning projects, from ideation and data exploration through to model deployment, monitoring, and operationalisation (MLOps).</li>
        <li style_margin_bottom: 10px;">Dedicate myself to ongoing learning and professional growth, particularly in cutting-edge AI technologies, cloud-based data solutions, and the ethical application of AI to ensure responsible innovation.</li>
        <li style_margin_bottom: 10px;">Contribute positively to an organisational culture that champions innovation, data-informed decision-making, and the pursuit of excellence in creating impactful outcomes.</li>
    </ul>
    I am enthusiastic about bringing my dedication, analytical capabilities, and proactive learning approach to an organisation where I can make significant contributions and grow alongside industry leaders.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__": # pragma: no cover
    st.set_page_config(layout="wide", page_title="About Ramesh Shrestha")
    render_about_me_page()