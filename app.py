import streamlit as st
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# CONFIG
# -------------------------

st.title("AI Resume Screening Tool")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Skills database
SKILLS_DB = [
    "python", "java", "c++", "sql", "mysql", "postgresql",
    "django", "flask", "fastapi", "react", "node.js",
    "aws", "docker", "kubernetes", "git", "linux",
    "machine learning", "tensorflow", "pytorch",
    "data structures", "algorithms"
]

# -------------------------
# FUNCTIONS
# -------------------------

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def clean_text(text):
    return " ".join(text.split())


def calculate_match_score(jd_text, resume_text):
    jd_embedding = model.encode([jd_text])
    resume_embedding = model.encode([resume_text])

    score = cosine_similarity(jd_embedding, resume_embedding)[0][0]
    return float(score)


def extract_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS_DB:
        if skill in text:
            found.append(skill)

    return found


def get_fit_label(score):
    percentage = score * 100

    if percentage >= 40:
        return "🔥 Strong Fit"
    elif percentage >= 20:
        return "👍 Medium Fit"
    else:
        return "⚠️ Weak Fit"

# -------------------------
# UI INPUTS
# -------------------------

jd_text = st.text_area("Paste Job Description")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    accept_multiple_files=True
)

# -------------------------
# MAIN LOGIC
# -------------------------

if st.button("Analyze"):

    results = []

    jd_text = clean_text(jd_text)
    jd_skills = extract_skills(jd_text)

    for file in uploaded_files:

        resume_text = extract_text_from_pdf(file)
        resume_text = clean_text(resume_text)

        # Reject unreadable PDFs
        if len(resume_text) < 300:
            st.warning(
                f"{file.name} is not a text-based PDF or has very little text."
            )
            continue

        # Match score
        score = calculate_match_score(jd_text, resume_text)

        # Skill matching
        resume_skills = extract_skills(resume_text)

        matched_skills = [s for s in jd_skills if s in resume_skills]
        missing_skills = [s for s in jd_skills if s not in resume_skills]

        results.append({
            "name": file.name,
            "score": score,
            "fit": get_fit_label(score),
            "matched": matched_skills,
            "missing": missing_skills
        })

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    # -------------------------
    # TOP 3 CANDIDATES
    # -------------------------

    st.subheader("⭐ Top Recommended Candidates")

    top_candidates = results[:3]

    for i, c in enumerate(top_candidates, start=1):
        st.write(
            f"{i}. {c['name']} — {c['fit']} ({c['score']*100:.2f}%)"
        )

    # -------------------------
    # DOWNLOAD REPORT
    # -------------------------

    report_data = []

    for r in results:
        report_data.append({
            "Name": r["name"],
            "Match Score (%)": round(r["score"] * 100, 2),
            "Fit Level": r["fit"],
            "Matched Skills": ", ".join(r["matched"]),
            "Missing Skills": ", ".join(r["missing"])
        })

    df = pd.DataFrame(report_data)

    st.download_button(
        label="⬇️ Download Shortlist Report (CSV)",
        data=df.to_csv(index=False),
        file_name="resume_shortlist_report.csv",
        mime="text/csv"
    )

    # -------------------------
    # FULL RANKED LIST
    # -------------------------

    st.subheader("Ranked Candidates")

    for r in results:
        st.markdown("---")
        st.subheader(r["name"])

        st.write(f"Match Score: {r['score']*100:.2f}%")
        st.write("Fit Level:", r["fit"])

        st.write("✅ Matched Skills:", ", ".join(r["matched"]))
        st.write("❌ Missing Skills:", ", ".join(r["missing"]))
