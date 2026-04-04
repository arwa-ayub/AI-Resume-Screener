import streamlit as st
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import pandas as pd
import re

# 1. SETUP THE PAGE
st.set_page_config(page_title="AI Resume Matcher Pro", page_icon="🤖")
st.title("🤖 AI Resume Screener")
st.write("Using **Semantic NLP** to find the perfect candidate.")

# 2. LOAD THE AI BRAIN
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# 3. HELPER FUNCTIONS (The "Why" logic)
def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_insights(text1, text2):
    # This finds common words with more than 4 letters (to find skills like 'Python', 'React')
    words1 = set(re.findall(r'\w{5,}', text1.lower()))
    words2 = set(re.findall(r'\w{5,}', text2.lower()))
    common = list(words1.intersection(words2))
    return common[:8] # Return top 8 matching words

# 4. THE USER INTERFACE
job_desc = st.text_area("📋 Job Description", placeholder="Paste the job requirements here...", height=150)
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if st.button("Analyze Match"):
    if job_desc and uploaded_file:
        with st.spinner("AI is examining the context..."):
            # Get text
            resume_text = extract_text(uploaded_file)
            
            # AI Math (Semantic Meaning)
            emb1 = model.encode(job_desc, convert_to_tensor=True)
            emb2 = model.encode(resume_text, convert_to_tensor=True)
            score = float(util.cos_sim(emb1, emb2)[0][0]) * 100

            # --- DISPLAY RESULTS ---
            st.divider()
            st.subheader(f"Match Result: {score:.1f}%")
            
            # 5. THE VISUAL BAR (New Feature)
            st.progress(score / 100)

            if score > 70:
                st.success("🌟 Strong Match: Profile aligns perfectly.")
            elif score > 40:
                st.warning("⚠️ Moderate Match: Some skills overlap, but gaps exist.")
            else:
                st.error("📉 Low Match: Profile does not align well.")

            # 6. THE "INSIGHT" SECTION (New Feature - The 'Why')
            st.subheader("🔍 Match Insights")
            matching_skills = get_insights(job_desc, resume_text)
            
            if matching_skills:
                st.write("The AI found these matching skills/keywords:")
                # Display skills as pretty tags
                st.write(", ".join([f"`{word.upper()}`" for word in matching_skills]))
            else:
                st.write("No direct word overlaps. The score is based on overall sentence meaning.")
                
            # 7. THE CHART (New Feature)
            chart_data = pd.DataFrame({"Match": [score], "Gap": [100-score]}, index=["Analysis"])
            st.bar_chart(chart_data)

    else:
        st.error("Please provide both the Job Description and the Resume!")