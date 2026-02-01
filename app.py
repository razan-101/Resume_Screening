# ============================================================
# Privacy-Aware Resume Screening Using NLP
# Edge–Cloud Multi-Tenant Architecture with Recruiter Assignment
# ============================================================

import streamlit as st
import pickle
import re
import os
import random
import pdfplumber
import docx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

# ------------------------------------------------------------
# CLOUD LAYER
# Load trained NLP models (TF-IDF + Classifier)
# ------------------------------------------------------------
MODEL_DIR = "./model"
CLF_PATH = os.path.join(MODEL_DIR, "clf.pkl")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")

if not (os.path.exists(CLF_PATH) and os.path.exists(TFIDF_PATH)):
    st.error("Model files not found in 'model/' directory.")
    st.stop()

with open(CLF_PATH, "rb") as f:
    clf = pickle.load(f)

with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)

# ------------------------------------------------------------
# EDGE LAYER
# Privacy-aware preprocessing
# ------------------------------------------------------------
def remove_pii(text):
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'\+?\d[\d -]{8,12}\d', '[PHONE]', text)
    return text


def clean_resume(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()


def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


# ------------------------------------------------------------
# MULTI-TENANT COMPANY STRUCTURE
# ------------------------------------------------------------
company_structure = {
    "Tech Solutions Pvt Ltd": {
        "roles": [
            "Java Developer", "Python Developer", "Web Designing",
            "DotNet Developer", "Database", "DevOps Engineer",
            "Blockchain", "SAP Developer"
        ],
        "recruiters": [
            "Alice – Backend Team",
            "Bob – Frontend Team",
            "Charlie – DevOps Team"
        ]
    },
    "Data & Analytics Corp": {
        "roles": [
            "Data Science", "Hadoop", "ETL Developer",
            "Business Analyst", "Operations Manager", "PMO"
        ],
        "recruiters": [
            "David – Data Engineering",
            "Eva – Analytics",
            "Frank – Program Management"
        ]
    },
    "Quality & Security Systems Ltd": {
        "roles": [
            "Testing", "Automation Testing", "Network Security Engineer"
        ],
        "recruiters": [
            "Grace – QA",
            "Henry – Automation",
            "Ian – Security"
        ]
    },
    "Enterprise & Sales Solutions": {
        "roles": ["Sales", "HR", "Advocate"],
        "recruiters": [
            "Jack – Sales",
            "Karen – HR",
            "Leo – Legal"
        ]
    },
    "Engineering & Manufacturing Group": {
        "roles": [
            "Mechanical Engineer", "Electrical Engineering", "Civil Engineer"
        ],
        "recruiters": [
            "Mona – Mechanical",
            "Nathan – Electrical",
            "Olivia – Civil"
        ]
    },
    "Creative & Wellness Services": {
        "roles": ["Arts", "Health and Fitness"],
        "recruiters": [
            "Paul – Creative",
            "Quincy – Wellness",
            "Rachel – Lifestyle"
        ]
    }
}


def assign_recruiter(recruiters):
    return random.choice(recruiters)


# ------------------------------------------------------------
# STREAMLIT APPLICATION
# ------------------------------------------------------------
def main():
    st.title("Privacy-Aware Resume Screening System")

    st.sidebar.header("Company Selection")
    tenant = st.sidebar.selectbox(
        "Select Company (Tenant)",
        list(company_structure.keys())
    )

    st.sidebar.header("Resume Upload")
    uploaded_file = st.file_uploader(
        "Upload Resume (TXT, PDF, DOCX)",
        type=["txt", "pdf", "docx"]
    )

    if uploaded_file is not None:

        file_type = uploaded_file.name.split(".")[-1].lower()

        if file_type == "txt":
            resume_text = uploaded_file.read().decode("utf-8", errors="ignore")

        elif file_type == "pdf":
            resume_text = extract_text_from_pdf(uploaded_file)

        elif file_type == "docx":
            resume_text = extract_text_from_docx(uploaded_file)

        else:
            st.error("Unsupported file type")
            st.stop()

        # -------- EDGE PROCESSING --------
        private_text = remove_pii(resume_text)
        cleaned_text = clean_resume(private_text)

        # -------- CLOUD PROCESSING --------
        features = tfidf.transform([cleaned_text])
        prediction_id = clf.predict(features)[0]

        category_mapping = {
            15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
            20: "Python Developer", 24: "Web Designing", 12: "HR",
            13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
            18: "Operations Manager", 6: "Data Science", 22: "Sales",
            16: "Mechanical Engineer", 1: "Arts", 7: "Database",
            11: "Electrical Engineering", 14: "Health and Fitness",
            19: "PMO", 4: "Business Analyst", 9: "DotNet Developer",
            2: "Automation Testing", 17: "Network Security Engineer",
            21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
        }

        predicted_role = category_mapping.get(prediction_id, "Unknown")

        st.subheader("Resume Screening & Assignment Result")
        st.info(f"Selected Company: {tenant}")
        st.info(f"Predicted Job Role: {predicted_role}")

        company_roles = company_structure[tenant]["roles"]
        company_recruiters = company_structure[tenant]["recruiters"]

        if predicted_role in company_roles:
            assigned_person = assign_recruiter(company_recruiters)
            st.success("✅ Resume matches company requirements")
            st.success(f"Routed to: {tenant}")
            st.success(f"Assigned Recruiter: {assigned_person}")
        else:
            st.error("❌ Resume does NOT match company requirements")
            st.warning(
                f"{tenant} is currently hiring for:\n"
                f"{', '.join(company_roles)}"
            )

        if st.checkbox("Show Word Cloud"):
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                stopwords=set(stopwords.words("english"))
            ).generate(cleaned_text)

            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)


if __name__ == "__main__":
    main()
