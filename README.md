# Privacy-Aware Resume Screening Using NLP  
### Edge–Cloud Multi-Tenant Recruitment System

This project demonstrates a **privacy-aware resume screening system** built using **Natural Language Processing (NLP)** and a **multi-tenant cloud architecture**.  
It simulates how modern recruitment platforms automatically screen resumes while protecting sensitive user data.

---

## 🚀 Project Highlights

✔ Privacy-aware preprocessing at the **edge layer**  
✔ NLP-based resume classification in the **cloud layer**  
✔ Multi-tenant SaaS architecture (multiple companies)  
✔ Grouped Job Descriptions (JDs) per company  
✔ Recruiter-level resume assignment  
✔ Supports **TXT, PDF, and DOCX** resumes  
✔ Word cloud visualization of resume content  

---

## 🧠 System Architecture

**Edge Layer**
- Resume upload
- Personally Identifiable Information (PII) removal
- Text cleaning

**Cloud Layer**
- TF-IDF feature extraction
- Machine Learning model for job role prediction

**Multi-Tenant Routing Layer**
- Resume routed to the correct company based on predicted role
- Assigned to a recruiter inside that company

---

## 📂 Project Structure

```
resume-screening-app/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Required Python packages
├── model/
│   ├── clf.pkl           # Trained classification model
│   └── tfidf.pkl         # TF-IDF vectorizer
├── dataset/              # Training dataset (reference)
```

---

## 🛠️ Installation & Setup (Step-by-Step)

### 1️⃣ Install Anaconda (if not installed)
Download from: https://www.anaconda.com/

---

### 2️⃣ Create a Project Environment

Open **Anaconda Prompt** and run:

```bash
conda create -n resume_nlp python=3.10
conda activate resume_nlp
```

---

### 3️⃣ Install Required Libraries

Navigate to the project folder:

```bash
cd path\to\resume-screening-app
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download NLTK Data (One Time Only)

Run Python once:

```bash
python
```

Then:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
exit()
```

---

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

Your browser will open automatically at:
```
http://localhost:8501
```

---

## 🧪 How to Use

1. Select a **Company (Tenant)** from the sidebar  
2. Upload a resume file (`.txt`, `.pdf`, or `.docx`)  
3. The system will:
   - Remove sensitive data
   - Predict the job role
   - Check if the resume matches company requirements
   - Assign it to a recruiter  

---

## 🗣️ Viva Explanation (Short Version)

> “The system performs privacy-aware preprocessing at the edge, uses NLP in the cloud for resume classification, and routes resumes to the correct tenant and recruiter based on job role matching.”

---

## 📌 Notes

- Models are pre-trained and stored in the `model/` folder  
- No cloud deployment required (architecture is logically simulated)  
- Designed for academic demonstration of Edge–Cloud and Multi-Tenant concepts  
