import streamlit as st
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Screening Tool")

model = SentenceTransformer("all-MiniLM-L6-v2")

SKILLS_DB = [
    "python", "java", "c++", "sql", "mysql", "postgresql",
    "django", "flask", "fastapi", "react", "node.js",
    "aws", "docker", "kubernetes", "git", "linux",
    "machine learning", "tensorflow", "pytorch",
    "data structures", "algorithms"
]

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
    return [skill for skill in SKILLS_DB if skill in text]


def get_fit_label(score):
    p = score * 100
    if p >= 40:
        return "🔥 Strong Fit"
    elif p >= 20:
        return "👍 Medium Fit"
    return "⚠️ Weak Fit"


jd_text = st.text_area("Paste Job Description")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    accept_multiple_files=True
)


if st.button("Analyze"):

    results = []

    jd_text = clean_text(jd_text)
    jd_skills = extract_skills(jd_text)

    for file in uploaded_files:

        resume_text = clean_text(extract_text_from_pdf(file))

        if len(resume_text) < 300:
            st.warning(f"{file.name} has very little readable text.")
            continue

        score = calculate_match_score(jd_text, resume_text)

        resume_skills = extract_skills(resume_text)

        matched = [s for s in jd_skills if s in resume_skills]
        missing = [s for s in jd_skills if s not in resume_skills]

        results.append({
            "name": file.name,
            "score": score,
            "fit": get_fit_label(score),
            "matched": matched,
            "missing": missing
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    st.subheader("⭐ Top Recommended Candidates")

    for i, c in enumerate(results[:3], start=1):
        st.write(f"{i}. {c['name']} — {c['fit']} ({c['score']*100:.2f}%)")

    report = pd.DataFrame([
        {
            "Name": r["name"],
            "Match Score (%)": round(r["score"]*100, 2),
            "Fit Level": r["fit"],
            "Matched Skills": ", ".join(r["matched"]),
            "Missing Skills": ", ".join(r["missing"])
        }
        for r in results
    ])

    st.download_button(
        "⬇️ Download Shortlist Report (CSV)",
        report.to_csv(index=False),
        "resume_shortlist_report.csv",
        "text/csv"
    )

    st.subheader("Ranked Candidates")

    for r in results:
        st.markdown("---")
        st.subheader(r["name"])
        st.write(f"Match Score: {r['score']*100:.2f}%")
        st.write("Fit Level:", r["fit"])
        st.write("✅ Matched Skills:", ", ".join(r["matched"]))
        st.write("❌ Missing Skills:", ", ".join(r["missing"]))
