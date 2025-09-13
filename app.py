import streamlit as st
import joblib
from src.file_utils import extract_text_from_pdf, extract_text_from_docx
from src.app_helpers import skills_extraction, fit_score_computation

# Streamlit Page Config
st.set_page_config(page_title="Resume Screening App", page_icon="📄", layout="centered")

# Custom CSS 
st.markdown("""
    <style>
        .main {
            background-color: #0e1117; /* dark background */
            color: #fafafa; /* white text */
        }
        .stTitle {
            text-align: center;
            font-size: 36px !important;
            color: #4da6ff !important; /* blue heading */
        }
        .stSubtitle {
            text-align: center;
            font-size: 18px !important;
            color: #a6c9ff !important;
        }
        .stContainer {
            background-color: #1c1f26;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 0px 10px rgba(77,166,255,0.3);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Load model & vectorizer
model = joblib.load("src/resume_model.pkl")
tfidf = joblib.load("src/tfidf.pkl")

# Sidebar Navigation
app_mode = st.sidebar.selectbox('📌 Navigation', ['Home', 'Prediction','About Author'])

# PAGE 1: HOME
if app_mode == 'Home':
    st.markdown('<h1 class="stTitle">📄 Automated Resume Screening</h1>', unsafe_allow_html=True)
    st.markdown('<p class="stSubtitle">Upload your resume and get instant predictions about your best-fit role & skills match</p>', unsafe_allow_html=True)

    st.markdown("""
    Recruiters receive hundreds of resumes for one role, making manual screening slow and inconsistent.  
    This app helps by:  

    - 📂 Reading resumes in **.pdf / .docx** format  
    - 🤖 Using **NLP & ML models** to predict job role  
    - 📊 Generating a **Fit Score** based on skills match  
    - ⚡ Providing instant, unbiased results through a web interface  

    ---
    🚀 A practical showcase of NLP + ML in solving a real-world HR challenge.
    """)

# PAGE 2: PREDICTION 
elif app_mode == 'Prediction':
    st.markdown('<h1 class="stTitle">🔎 Resume Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="stSubtitle">Upload your resume to predict job role & fit score</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload Resume File (.pdf or .docx)", type=["pdf", "docx"])

    if uploaded_file:
        # Extract text
        if uploaded_file.name.endswith(".docx"):
            text = extract_text_from_docx(uploaded_file)
        else:
            text = extract_text_from_pdf(uploaded_file)

        clean_text = text.lower()
        vectorized = tfidf.transform([clean_text])
        pred_role = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized).max()

        # Skill extraction + fit score
        skills_found = skills_extraction(text, pred_role)
        fit_score = fit_score_computation(prob, skills_found, pred_role)
        
        # Display extracted text in expander
        with st.expander("📑 View Extracted Resume Text"):
            st.text_area("Extracted Resume Content", text, height=300)

        # Styled Result Container
        with st.container(border=True):
            st.write(f"🎯 Predicted Role: **{pred_role}**")
            st.write(f"📊 Fit Score: **{fit_score}%**")
            st.write(f"🛠️ Matched Skills: {', '.join(skills_found) if skills_found else 'None'}")
# PAGE 3: ABOUT AUTHOR
elif app_mode == 'About Author':
    st.markdown('<h1 class="stTitle">👩‍💻 About the Author</h1>', unsafe_allow_html=True)
    st.markdown('<p class="stSubtitle">Know the creator behind this project</p>', unsafe_allow_html=True)

    st.write("### Mariam Khan")
    st.write("📍 Karachi, Pakistan")
    st.write("📧 Email: **khanmariam684@gmail.com**")
    st.write("🔗 [LinkedIn](https://www.linkedin.com/in/mariam-khan0424) | [GitHub](https://github.com/mariamkhan04)")

    st.write("### About Me 🎯")
    st.markdown("""
    I am an undergraduate doing BSCS from Karachi University(UBIT), passionate about **Data Science, Machine Learning, and AI**.  
    Skilled in **Python, JavaScript, Pandas, NumPy, Scikit-learn, and data visualization tools**, I enjoy working on 
    **analysis-driven and ML-based projects** that transform raw data into meaningful insights and practical solutions.  
    """)
    
    st.write("### Why this Project? 🚀")
    st.markdown("""
    This project was part of my **Data Science Bootcamp challenge**, where I learned and implemented key strategies like 
    **text preprocessing, feature engineering, model training, and deployment** completely on my own.  
    It reflects both my technical learning journey and ability to apply concepts to solve real-world problems.  
    """)