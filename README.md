# Privacy-Aware Resume Screening Using NLP

### Edge–Cloud Multi-Tenant Recruitment System

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-brightgreen)](https://privacy-aware-resume-screening.streamlit.app)

Live Application:
https://privacy-aware-resume-screening.streamlit.app

---

This project demonstrates a **privacy-aware resume screening system** built using **Natural Language Processing (NLP)** and a **multi-tenant cloud architecture**.
It simulates how modern recruitment platforms automatically screen resumes while protecting sensitive user data.

---

## 🚀 Project Highlights

✔ Privacy-aware preprocessing at the **edge layer**
✔ NLP-based resume classification in the **cloud layer**
✔ Multi-tenant SaaS architecture (multiple companies)
✔ Grouped Job Descriptions (JDs) per company
✔ Recruiter-level resume assignment
✔ Candidate ranking based on model confidence
✔ Unique tracking ID for each candidate
✔ Backend-powered candidate tracking system
✔ Supports **TXT, PDF, and DOCX** resumes
✔ Word cloud visualization

---

## 🧠 System Architecture

### 🔹 Edge Layer (Streamlit Frontend)

* Resume upload
* PII masking (email, phone)
* Resume text extraction
* Word cloud visualization

### 🔹 Cloud Layer (FastAPI Backend)

* TF-IDF feature extraction
* ML model prediction
* Eligibility check
* Recruiter assignment
* Ranking system

### 🔹 Data Layer

* SQLite database (`resume_screening.db`)
* Stores candidate data, tracking ID, ranking, and status

---

## 📂 Project Structure

```
resume-screening-app/
│
├── app.py
├── backend.py
├── resume_screening.db
├── requirements.txt
│
├── model/
│   ├── clf.pkl
│   └── tfidf.pkl
│
├── dataset/
├── images/
├── .streamlit/
├── .devcontainer/
```

---

## 🛠️ Installation & Setup (Step-by-Step)

### 1️⃣ Install Anaconda (if not installed)

Download from:
https://www.anaconda.com/

---

### 2️⃣ Create a Project Environment

Open **Anaconda Prompt** and run:

```bash
conda create -n resume_nlp python=3.10
conda activate resume_nlp
```

---

### 3️⃣ Install Required Libraries

Navigate to your project folder:

```bash
cd path\to\resume-screening-app
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download NLTK Data (One Time Only)

Run Python:

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

## ⚠️ Additional Setup for spaCy (Important)

This project uses **spaCy for advanced NLP-based PII removal**.

After installing requirements, download the language model:

```bash
python -m spacy download en_core_web_sm
```

### 🔄 Optional Auto-Setup (Fallback)

```python

import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

exit()
```

### 💡 Why this is needed

* `pip install` installs only the spaCy library
* Language models are separate downloads
* Prevents runtime errors if model is missing

---

## ▶️ Running the Application (VERY IMPORTANT)

🚨 You MUST run backend and frontend separately

---

### 🖥️ Terminal 1 — Start Backend

Run this command directly in terminal:

```bash
python backend.py
```

OR

```bash
uvicorn backend:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

Check backend:

```
http://127.0.0.1:8000/health
```

---

### 🖥️ Terminal 2 — Start Frontend

```bash
streamlit run app.py
```

App runs at:

```
http://localhost:8501
```

---

### ⚠️ Important Rules

* Always start **backend first**
* Then start **Streamlit app**
* If backend is not running → frontend will show error

---

## 🧪 How to Use

### 🔹 Resume Page (Recruiter)

1. Enter candidate name
2. Enter candidate email
3. Select company
4. Upload resume (`.txt`, `.pdf`, `.docx`)
5. Click **Check Eligibility**

👉 System will:

* Remove sensitive data
* Predict job role
* Check eligibility
* Assign recruiter
* Generate **Tracking ID**

---

### 🔹 Interviewer Page

* View shortlisted candidates
* See ranking and recruiter assignment
* Filter by company or interviewer

---

### 🔹 Candidate Page

1. Enter Tracking ID
2. View:

   * Status
   * Selection result
   * Recruiter
   * Ranking

---

## 📡 Backend API Endpoints

* `GET /health`
* `GET /tenants`
* `POST /candidates/upload-resume`
* `GET /interviewers/candidates`
* `GET /candidates/status/{tracking_id}`

---

## 🗄️ Database

Uses:

```
resume_screening.db
```

Stores:

* Candidate details
* Tracking ID
* Role prediction
* Status
* Ranking

---

## 🗣️ Viva Explanation

> “This system uses Streamlit as the frontend and FastAPI as the backend.
> Resumes are processed using NLP, stored in a database, ranked based on model confidence, and tracked using a unique ID in a multi-tenant architecture.”

---

## 📌 Notes

* Model files must exist in `model/` folder
* Backend must run on `http://127.0.0.1:8000`
* Streamlit communicates via REST APIs
* Database auto-initializes on first run

---

## 🚀 Future Improvements

* Login system (Recruiter / Interviewer roles)
* Advanced NLP (NER-based anonymization)
* Cloud deployment (AWS / Azure)
* Interview status updates

---
