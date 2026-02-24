# app.py
import streamlit as st
from pathlib import Path
from Resume2 import get_Analysis 
import plotly.graph_objects as go
import streamlit as st
from st_circular_progress import CircularProgress
import time
from langchain_openai import OpenAIEmbeddings

api_key = st.secrets["OPENAI_API_KEY"]
# llm.openai_api_key=api_key
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=api_key
# )
# ---- Ngrok tunnel ----
# ngrok.set_auth_token("3A3fScLsWJY8clWsAuMBsjv6kpi_7wrpg5g41QNrbRXCMsMgs")
# public_url = ngrok.connect(addr=8501)
# st.write(f"Public URL (share this): {public_url}")

st.set_page_config(page_title="Resume Analyzer", layout="wide")
st.title("📝 Resume Analyzer & Job Matcher")

# Upload CV
cv_file = st.file_uploader("Upload Candidate CV (PDF)", type="pdf")

# Job Description input
job_description = st.text_area(
    "Enter Job Description",
    placeholder="Paste the job description here..."
)
job_description="""🚀 We’re Hiring: Junior / Early-Career AI Engineer (0–3 Years Experience)
Are you passionate about building real-world AI systems?
 We’re looking for a motivated Junior AI Engineer to join our team and contribute to the development, evaluation, and deployment of AI models in production environments.

🔹 What You’ll Do:
Develop, train, and evaluate ML & LLM models
Prepare and preprocess datasets for training and validation
Implement reproducible experimentation workflows
Integrate AI models into backend services and inference pipelines
Support model performance optimization and monitoring
Collaborate with software, platform, and QA teams
Document experiments and technical findings clearly

🔹 What We’re Looking For:
Bachelor’s or Master’s in Computer Science, Engineering, or a related field
0–3 years of hands-on ML/AI experience
Strong Python programming skills
Experience with PyTorch, TensorFlow, or similar frameworks
Solid understanding of ML fundamentals (training, validation, metrics)
Familiarity with Git and software engineering best practices
Exposure to LLMs, embeddings, generative AI, fine-tuning, and benchmarking
Experience with APIs, Docker, or cloud environments is a plus
Familiarity with digital design, verification, and the software development lifecycle is a plus"""

def display_skills_inline(skills, color="#4CAF50"):
    skills_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;">'

    for skill in skills:
        skills_html += f"""
        <span style="
            background:{color};
            color:white;
            padding:6px 12px;
            border-radius:20px;
            font-size:13px;
            font-weight:500;
            white-space:nowrap;
        ">
            {skill}
        </span>
        """

    skills_html += "</div>"

    return skills_html

if st.button("Analyze"):

    if cv_file is None:
        st.warning("Please upload a CV file first.")
    elif not job_description:
        st.warning("Please enter a job description.")
    else:
        # Save uploaded CV temporarily
        temp_cv_path = Path("temp_cv.pdf")
        with open(temp_cv_path, "wb") as f:
            f.write(cv_file.read())

        # Run analysis
        with st.spinner("Analyzing CV..."):
            try:
            #======= First Row =======#
                score_col1, score_col2, score_col3 = st.columns([1,2,1])
                result = get_Analysis(temp_cv_path, job_description,api_key)
                color=''
                if result.match_score >= 75:
                    color = 'green' 
                elif result.match_score >= 50: 
                    color = "#FFFF7E"  # Yellow in hex
                else: 
                    color = '#FF0000'  # Red in hex
                
                score = int(result.match_score)
                st.success("Analysis Complete!")
                with score_col2:
                    progress = CircularProgress(
                        label="Your match score",
                        value=score,
                        size="large",
                        color=color,
                        track_color="#e0e0e0",
                        key="progress_main"  
                    )

                    progress.st_circular_progress()

                    for i in range(score + 1):
                        progress.update_value(i)  
                        time.sleep(0.02)
            #======== 2ND row ==========#
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("""
                    <div style="
                        padding:18px;
                        border-radius:14px;
                        background:#f7fff8;
                        border:1px solid #e6f4ea;
                    ">
                    <h4 style="margin-bottom:10px;">✅ Matching Skills</h4>
                    """, unsafe_allow_html=True)
                    st.html(display_skills_inline(result.matching_skills, "#22c55e"))
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <div style="
                        padding:18px;
                        border-radius:14px;
                        background:#fff7f7;
                        border:1px solid #fde2e2;
                    ">
                    <h4 style="margin-bottom:10px;">⚠️ Missing Skills</h4>
                    """, unsafe_allow_html=True)
                    st.html(display_skills_inline(result.missing_skills, "#a30016"))
                    st.markdown("</div>", unsafe_allow_html=True)


                # ===== Row 3 — Insights =====
                st.divider()

                st.subheader("ℹ️ Seniority / Notes")
                st.info(result.seniority_mismatch)

                st.subheader("🏆 Recommendation")
                st.success(result.recommendation)

            except Exception as e:
                st.error(f"Error during analysis: {e}")

        temp_cv_path.unlink()
