import re
from typing import List, Set, Tuple

import streamlit as st
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util


# ---------------------------
# App configuration
# ---------------------------
st.set_page_config(
    page_title="AI Resume & Job Description Matcher",
    page_icon="📄",
    layout="wide",
)


# ---------------------------
# Model loading (cached)
# ---------------------------
@st.cache_resource
def load_nlp_model():
    """
    Load the spaCy English model.
    Make sure 'en_core_web_sm' is installed.
    """
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        # If the model is not installed, show a clear message.
        st.error(
            "spaCy model 'en_core_web_sm' is not installed.\n\n"
            "Run this in your terminal:\n"
            "`python -m spacy download en_core_web_sm`"
        )
        st.stop()
    return nlp_model


@st.cache_resource
def load_embedding_model():
    """
    Load the SentenceTransformer model.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


nlp = load_nlp_model()
embed_model = load_embedding_model()


# ---------------------------
# Skill / keyword extraction
# ---------------------------

# A small, extensible dictionary of common skills/tools/keywords.
KNOWN_SKILLS = [
    # Programming languages
    "python", "java", "c++", "c", "c#", "javascript", "typescript", "go", "rust",
    "scala", "ruby", "php", "r",
    # ML / Data / AI
    "machine learning", "deep learning", "data science", "nlp",
    "natural language processing", "computer vision", "recommendation systems",
    "reinforcement learning", "statistics", "time series",
    # Frameworks / Libraries
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch",
    "keras", "matplotlib", "seaborn", "xgboost", "lightgbm", "spaCy",
    "hugging face transformers", "transformers",
    # Tools / DevOps
    "git", "docker", "kubernetes", "jenkins", "linux", "bash", "shell scripting",
    "aws", "azure", "gcp", "google cloud", "ci/cd",
    # Web / APIs / Backend
    "rest api", "restful apis", "fastapi", "django", "flask",
    "microservices",
    # Data / Storage
    "sql", "nosql", "postgresql", "mysql", "mongodb", "data engineering",
    # General skills
    "agile", "scrum", "project management", "communication", "leadership",
    "teamwork", "problem solving",
]


def normalize_term(term: str) -> str:
    """
    Normalize a term for comparison.
    Lowercase, strip, and keep alphanumerics, +, #, ., and spaces.
    """
    term = term.strip().lower()
    term = re.sub(r"[^a-z0-9#+.\s]", "", term)
    term = re.sub(r"\s+", " ", term)
    return term


def extract_skills_keywords(text: str) -> List[str]:
    """
    Extract skills/tools/keywords from text using:
    - spaCy NER entities
    - Keyword matching from a known skill list
    Returns a sorted list of normalized unique terms.
    """
    if not text or not text.strip():
        return []

    doc = nlp(text)
    skills: Set[str] = set()

    # 1. From spaCy Named Entity Recognition
    # We treat certain entity types as potential skills/tools/keywords.
    ner_labels_to_use = {
        "ORG", "PRODUCT", "LANGUAGE", "NORP", "WORK_OF_ART", "FAC"
    }
    for ent in doc.ents:
        if ent.label_ in ner_labels_to_use:
            norm = normalize_term(ent.text)
            if norm and len(norm) > 1:
                skills.add(norm)

    # 2. From keywords dictionary (case-insensitive)
    lower_text = text.lower()
    for skill in KNOWN_SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"s?\b"
        if re.search(pattern, lower_text):
            norm = normalize_term(skill)
            if norm:
                skills.add(norm)

    # 3. Optional: nouns & proper nouns as extra keywords (basic heuristic)
    for token in doc:
        if (
            token.pos_ in {"NOUN", "PROPN"} and
            not token.is_stop and
            len(token.text) > 2
        ):
            norm = normalize_term(token.lemma_)
            if norm and len(norm) > 2:
                skills.add(norm)

    # Return sorted unique list
    return sorted(skills)


# ---------------------------
# Similarity computation
# ---------------------------
def compute_similarity(resume_text: str, jd_text: str) -> float:
    """
    Compute cosine similarity between resume and job description
    using SentenceTransformer embeddings.
    Returns similarity as a percentage (0–100).
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    emb1 = embed_model.encode(resume_text, convert_to_tensor=True)
    emb2 = embed_model.encode(jd_text, convert_to_tensor=True)

    cos_sim = util.cos_sim(emb1, emb2)  # Shape: [1, 1]
    score = float(cos_sim[0][0])  # -1 to 1
    score = max(0.0, min(1.0, score))  # Clamp to [0,1] for safety
    return score * 100.0


# ---------------------------
# Skill comparison
# ---------------------------
def compare_skills(
    resume_skills: List[str],
    jd_skills: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Compare skills between resume and job description.
    Returns (matched, missing, extra) as sorted lists.
    """
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    matched = sorted(resume_set & jd_set)
    missing = sorted(jd_set - resume_set)
    extra = sorted(resume_set - jd_set)

    return matched, missing, extra


def prettify_skill_list(skills: List[str]) -> List[str]:
    """
    Basic prettification of normalized skills for display.
    """
    pretty = []
    for s in skills:
        # Handle some common acronyms explicitly
        acronyms = {"nlp", "aws", "gcp", "sql", "ci/cd", "ml", "ai"}
        if s in acronyms:
            pretty.append(s.upper())
        else:
            pretty.append(s.title())
    return pretty


# ---------------------------
# Template-based suggestion generator
# ---------------------------
def infer_profession_from_resume(resume_text: str) -> str:
    """
    Very simple heuristic to infer profession from resume text.
    If we can't infer reliably, fall back to 'professional'.
    """
    # Look at first 2–3 sentences or lines
    first_chunk = " ".join(resume_text.splitlines()[:3])
    # Try patterns like "I am a/an X", "as a X", etc.
    patterns = [
        r"\bI am an? ([A-Za-z ]+)",
        r"\bI'm an? ([A-Za-z ]+)",
        r"\bAs an? ([A-Za-z ]+)",
        r"\bworking as an? ([A-Za-z ]+)",
    ]
    for pat in patterns:
        m = re.search(pat, first_chunk, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            # Basic cleanup
            candidate = re.sub(r"\bprofessional\b", "", candidate, flags=re.IGNORECASE)
            candidate = candidate.strip(" ,.")
            if candidate:
                return candidate

    # Fallback: try noun chunks, pick the first one that looks like a role
    doc = nlp(first_chunk)
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        if any(w.lower() in text.lower() for w in ["engineer", "developer", "manager", "analyst", "scientist"]):
            return text

    return "professional"


def infer_industry_from_jd(jd_text: str) -> str:
    """
    Simple heuristic to infer an industry/goal phrase from the JD.
    """
    jd_lower = jd_text.lower()

    # Look for explicit "industry" or "domain" phrases
    m = re.search(r"in the ([a-zA-Z ]+) industry", jd_lower)
    if m:
        return m.group(1).strip() + " industry"

    m = re.search(r"in the ([a-zA-Z ]+) domain", jd_lower)
    if m:
        return m.group(1).strip() + " domain"

    # Look for common domains
    domains = [
        "finance", "fintech", "healthcare", "e-commerce", "retail",
        "education", "edtech", "logistics", "supply chain", "saas",
        "cloud", "telecom", "media", "gaming",
        "artificial intelligence", "machine learning", "data science"
    ]
    for d in domains:
        if d in jd_lower:
            # Make it a nice phrase
            if "industry" in d or "domain" in d:
                return d
            return d + " domain"

    # Fallback generic phrase
    return "this role"


def infer_company_domain_from_jd(jd_text: str) -> str:
    """
    Rough heuristic for company domain / organization phrase.
    """
    doc = nlp(jd_text)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    if orgs:
        return f"{orgs[0]} and its mission"

    # Fallback
    return "your organization"


def generate_resume_suggestion(
    resume_text: str,
    jd_text: str,
    matched_skills: List[str],
) -> str:
    """
    Generate a job-relevant resume summary using a simple template.
    No external APIs; purely rule-based.
    """
    profession = infer_profession_from_resume(resume_text)
    industry = infer_industry_from_jd(jd_text)
    company_domain = infer_company_domain_from_jd(jd_text)

    # Use up to top 3 matched skills
    pretty_matched = prettify_skill_list(matched_skills)
    if pretty_matched:
        top_skills = ", ".join(pretty_matched[:3])
    else:
        top_skills = "relevant skills for this role"

    template = (
        "I am a skilled {profession} with experience in {skills}. "
        "I am passionate about {industry} and eager to contribute to {company_domain}."
    )

    suggestion = template.format(
        profession=profession,
        skills=top_skills,
        industry=industry,
        company_domain=company_domain,
    )
    return suggestion


# ---------------------------
# UI components
# ---------------------------
st.title("📄 AI Resume & Job Description Matcher")
st.markdown(
    """
This tool compares your **resume** against a **job description** using semantic similarity
and skill overlap to help you tailor your application.
"""
)

# Layout: two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Resume Input")
    resume_text = st.text_area(
        "Paste your resume text here",
        height=300,
        key="resume_input",
        placeholder=(
            "Example:\n"
            "Machine Learning Engineer with 3+ years of experience building NLP models, "
            "deploying ML pipelines, and optimizing model performance using Python, "
            "PyTorch, and cloud services..."
        ),
    )

with col2:
    st.subheader("💼 Job Description Input")
    jd_text = st.text_area(
        "Paste the job description here",
        height=300,
        key="jd_input",
        placeholder=(
            "Example:\n"
            "We are looking for a Machine Learning Engineer to work on NLP-powered "
            "products. The ideal candidate has experience with Python, deep learning "
            "frameworks (PyTorch/TensorFlow), REST APIs, and deploying models to AWS..."
        ),
    )

run_button = st.button("🔍 Run Matching")


# ---------------------------
# Main logic on button click
# ---------------------------
if run_button:
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please provide both **resume** and **job description** text.")
        st.stop()

    # --- Similarity Score ---
    st.markdown("## 🎯 Match Score")

    similarity_percent = compute_similarity(resume_text, jd_text)

    # Progress bar
    st.progress(int(similarity_percent))

    # Color-coded badge
    if similarity_percent >= 75:
        color = "#2e7d32"  # green
        label = "Strong Match"
    elif similarity_percent >= 50:
        color = "#f9a825"  # yellow/orange
        label = "Moderate Match"
    else:
        color = "#c62828"  # red
        label = "Weak Match"

    st.markdown(
        f"""
        <div style="
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            background-color: {color};
            color: white;
            font-weight: 600;
            display: inline-block;
            margin-top: 0.5rem;
        ">
            Match Score: {similarity_percent:.1f}% — {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Skill Extraction ---
    st.markdown("## 🧩 Skill Comparison")

    resume_skills_raw = extract_skills_keywords(resume_text)
    jd_skills_raw = extract_skills_keywords(jd_text)

    matched, missing, extra = compare_skills(resume_skills_raw, jd_skills_raw)

    pretty_resume_skills = prettify_skill_list(resume_skills_raw)
    pretty_jd_skills = prettify_skill_list(jd_skills_raw)
    pretty_matched = prettify_skill_list(matched)
    pretty_missing = prettify_skill_list(missing)
    pretty_extra = prettify_skill_list(extra)

    # Show raw extracted skills in expanders
    exp1, exp2 = st.columns(2)
    with exp1:
        with st.expander("🔍 Extracted Skills/Keywords from Resume"):
            if pretty_resume_skills:
                st.write(", ".join(pretty_resume_skills))
            else:
                st.write("_No clear skills/keywords detected._")
    with exp2:
        with st.expander("🔍 Extracted Skills/Keywords from Job Description"):
            if pretty_jd_skills:
                st.write(", ".join(pretty_jd_skills))
            else:
                st.write("_No clear skills/keywords detected._")

    # Three-column comparison
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### ✅ Matched Skills")
        if pretty_matched:
            for s in pretty_matched:
                st.write(f"- {s}")
        else:
            st.write("_No matched skills found._")

    with c2:
        st.markdown("### ❌ Missing Skills (in resume)")
        if pretty_missing:
            for s in pretty_missing:
                st.write(f"- {s}")
        else:
            st.write("_No obvious missing skills detected._")

    with c3:
        st.markdown("### ⚙️ Extra Skills (in resume)")
        if pretty_extra:
            for s in pretty_extra:
                st.write(f"- {s}")
        else:
            st.write("_No extra skills beyond the job description detected._")

    # --- Suggestions Section ---
    st.markdown("## 💡 Suggestions for Improvement")

    suggestion_text = generate_resume_suggestion(
        resume_text=resume_text,
        jd_text=jd_text,
        matched_skills=matched,
    )

    st.markdown("**Suggested Resume Summary (Template-based):**")
    st.write(suggestion_text)

    st.markdown(
        """
_You can copy this summary into your resume and tweak it further to reflect your
experience, achievements, and the specific role._"""
    )

else:
    st.info("Fill in both text areas and click **Run Matching** to see your match score and suggestions.")
