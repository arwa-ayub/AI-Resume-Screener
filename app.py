import streamlit as st
import pandas as pd
from engine import ResumeEngine

# Configuration
st.set_page_config(page_title="AI Resume Screener", page_icon="🤖", layout="wide")

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def init_engine():
    return ResumeEngine()

try:
    engine = init_engine()
except Exception as e:
    st.error(f"System Initialization Error: {e}")
    st.stop()

# Header
st.title("🤖AI-Powered Resume Screener")
st.markdown("#### Modular Semantic Matching Engine for Advanced Candidate Evaluation")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Job Description")
    job_input = st.text_area("Paste the role requirements here...", height=300, placeholder="We are looking for a Python Developer with experience in...")

with col2:
    st.subheader("📄 Candidate Resume")
    resume_file = st.file_uploader("Upload PDF Resume", type=["pdf"])
    if resume_file:
        st.success("File uploaded successfully!")

if st.button("Analyze Alignment", use_container_width=True):
    if not job_input or not resume_file:
        st.warning("Please provide both the Job Description and the Resume PDF.")
    else:
        with st.spinner("Analyzing semantics and segmenting resume structure..."):
            text = engine.extract_text(resume_file)
            if text:
                results = engine.compute_scores(job_input, text)
                keywords = engine.get_keywords(job_input, text)
                
                st.divider()
                
                # Top Level Metrics
                m1, m2 = st.columns([1, 3])
                with m1:
                    st.metric("Global Match Score", f"{results['overall']}%")
                with m2:
                    st.progress(results['overall'] / 100)
                    st.caption("Weighted score based on Skills, Experience, and Education alignment.")

                # Category Breakdown
                st.subheader("📊 Structural Breakdown")
                b1, b2, b3 = st.columns(3)
                b1.metric("Skills Alignment", f"{results['breakdown']['skills']}%")
                b2.metric("Experience Match", f"{results['breakdown']['experience']}%")
                b3.metric("Education Fit", f"{results['breakdown']['education']}%")

                # Keyword Insights
                st.subheader("💡 Matching Insights (Explainable AI)")
                if keywords:
                    st.write("The system detected overlapping core competencies:")
                    cols = st.columns(len(keywords) if len(keywords) < 5 else 5)
                    for idx, k in enumerate(keywords):
                        cols[idx % 5].info(f"**{k}**")
                else:
                    st.info("No significant technical keyword overlaps detected.")
                
                # Visual Chart
                chart_data = pd.DataFrame({
                    "Category": ["Skills", "Experience", "Education"],
                    "Match %": [results['breakdown']['skills'], results['breakdown']['experience'], results['breakdown']['education']]
                })
                st.bar_chart(chart_data.set_index("Category"))

            else:
                st.error("Text extraction failed. Please ensure the PDF is not encrypted or empty.")

st.sidebar.title("Engineering Specs")
st.sidebar.info("""
- **Core:** SBERT (all-MiniLM-L6-v2)
- **Architecture:** Modular Engine
- **Logic:** Category-Weighted Scoring
- **XAI:** Boolean Semantic Mapping
""")