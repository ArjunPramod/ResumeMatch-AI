# 📄 ResumeMatch AI  
### *Match your resume with job descriptions and get skill analysis plus improvement suggestions.*

ResumeMatch AI is a friendly tool that helps candidates and recruiters understand how well a resume aligns with a job description. It highlights matching skills, missing skills, and provides improvement suggestions to help applicants tailor their resumes effectively.

This tool makes resume evaluation simple, transparent, and fast — perfect for recruiters, hiring managers, and job seekers.

---

## ⭐ Key Features

### 🔍 Match Score
ResumeMatch AI compares a resume and a job description to generate a clear **0–100% match score**, making it easy to see how well a candidate fits the role at a glance.

---

### 🧩 Skill Comparison
The app automatically identifies skills from both documents and organizes them into:

- ✅ **Matched Skills** – skills present in both resume and job description  
- ❌ **Missing Skills** – skills required in the job description but not found in the resume  
- ⚙️ **Extra Skills** – additional strengths the candidate has beyond the job requirements  

This structured comparison helps recruiters understand candidate relevance instantly.

---

### 💡 Resume Improvement Suggestions
ResumeMatch AI provides a refined, job-focused resume summary based on the strongest aligned skills.  
This helps job seekers improve their resume before applying and ensures recruiters receive higher-quality applications.

---

### 🎨 Easy-to-Use Interface
The interface is clean, intuitive, and built for quick use:

- Two text boxes — one for the resume, one for the job description  
- Instant scoring and analysis  
- No login required  
- No data stored  

Ideal for both casual users and hiring teams.

---

## 🧰 How It Works (Simple Overview)

The app uses open-source natural language processing tools to:

- Understand the meaning of sentences  
- Extract relevant skills and keywords  
- Compare resume content with job requirements  
- Generate tailored improvement suggestions  

Everything runs locally or on your deployment — **no paid APIs required**.

---

## 🚀 How to Use

1. Paste a **resume** in the first text box  
2. Paste the **job description** in the second text box  
3. Click **Run Matching**  
4. Review your:
   - Match Score  
   - Skill Analysis  
   - Suggested Resume Summary  

The entire process takes **just a few seconds**.

---

## 🧑‍💼 Who Is This For?

- **Recruiters & HR Teams** – quick snapshot of candidate fit  
- **Job Seekers** – improve resume alignment before applying  
- **Career Coaches** – deliver clearer guidance  
- **Hiring Managers** – accelerate resume screening  

---

## 📦 Installation (For technical users)

If you'd like to run this app locally:

```bash
git clone https://github.com/ArjunPramod/ResumeMatch-AI.git
cd ResumeMatch-AI
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

streamlit run app.py
