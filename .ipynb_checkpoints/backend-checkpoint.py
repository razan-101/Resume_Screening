from __future__ import annotations

import os
import pickle
import sqlite3
import uuid
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CLF_PATH = os.path.join(MODEL_DIR, "clf.pkl")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")
DB_PATH = os.path.join(BASE_DIR, "resume_screening.db")

# ---------------- COMPANY STRUCTURE ----------------
company_structure = {
    "Tech Solutions Pvt Ltd":{
        "roles":["Java Developer","Python Developer","Web Designing","DotNet Developer","Database","DevOps Engineer","Blockchain","SAP Developer"],
        "recruiters":["Arjun – Backend Team","Rohan – Frontend Team","Vikram – DevOps Team"]
    },
    "Data & Analytics Corp":{
        "roles":["Data Science","Hadoop","ETL Developer","Business Analyst","Operations Manager","PMO"],
        "recruiters":["Amit – Data Engineering","Neha – Analytics","Suresh – Program Management"]
    },
    "Quality & Security Systems Ltd":{
        "roles":["Testing","Automation Testing","Network Security Engineer"],
        "recruiters":["Priya – QA","Karan – Automation","Rahul – Security"]
    },
    "Enterprise & Sales Solutions":{
        "roles":["Sales","HR","Advocate"],
        "recruiters":["Ankit – Sales","Pooja – HR","Manish – Legal"]
    },
    "Engineering & Manufacturing Group":{
        "roles":["Mechanical Engineer","Electrical Engineering","Civil Engineer"],
        "recruiters":["Rajesh – Mechanical","Deepak – Electrical","Sneha – Civil"]
    },
    "Creative & Wellness Services":{
        "roles":["Arts","Health and Fitness"],
        "recruiters":["Meera – Creative","Kavya – Wellness","Ananya – Lifestyle"]
    }
}

# ---------------- CATEGORY MAPPING ----------------
category_mapping = {
    15:"Java Developer",23:"Testing",8:"DevOps Engineer",20:"Python Developer",
    24:"Web Designing",12:"HR",13:"Hadoop",3:"Blockchain",10:"ETL Developer",
    18:"Operations Manager",6:"Data Science",22:"Sales",16:"Mechanical Engineer",
    1:"Arts",7:"Database",11:"Electrical Engineering",14:"Health and Fitness",
    19:"PMO",4:"Business Analyst",9:"DotNet Developer",2:"Automation Testing",
    17:"Network Security Engineer",21:"SAP Developer",5:"Civil Engineer",0:"Advocate"
}

# ---------------- LOAD MODEL ----------------
with open(CLF_PATH,"rb") as f: clf = pickle.load(f)
with open(TFIDF_PATH,"rb") as f: tfidf = pickle.load(f)

# ---------------- APP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DB ----------------
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.on_event("startup")
def startup():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracking_id TEXT UNIQUE,
            tenant TEXT,
            batch_name TEXT,
            predicted_role TEXT,
            eligible INTEGER,
            assigned_recruiter TEXT,
            ranking_score REAL,
            test_status TEXT DEFAULT 'pending',
            interview_status TEXT DEFAULT 'pending',
            final_status TEXT DEFAULT 'pending',
            applied_at TEXT,
            updated_at TEXT
        )
        """)

# ---------------- TENANTS ----------------
@app.get("/tenants")
def get_tenants():
    return {"tenants": company_structure}

# ---------------- 🔥 FIXED SAVE FINAL ----------------
@app.post("/save-final")
def save_final(data: str = Form(...), tenant: str = Form(...)):

    data = eval(data)
    recruiters = company_structure[tenant]["recruiters"]

    with get_conn() as conn:
        for i, d in enumerate(data):

            recruiter = recruiters[i % len(recruiters)]  # round-robin assignment

            conn.execute("""
            INSERT OR IGNORE INTO candidates (
                tracking_id, tenant, predicted_role,
                eligible, assigned_recruiter,
                ranking_score, applied_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                d["tracking_id"],
                tenant,
                d["Role"],
                int(d["Eligible"]),
                recruiter,
                d["Score"],
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat()
            ))

    return {"message": "Candidates saved successfully"}

# ---------------- INTERVIEWER ----------------
@app.get("/interviewers/candidates")
def get_candidates(tenant: str = Query(...), interviewer: str = Query(...)):
    with get_conn() as conn:
        rows = conn.execute("""
        SELECT * FROM candidates
        WHERE tenant=? AND assigned_recruiter=? AND eligible=1
        """,(tenant,interviewer)).fetchall()

    return [dict(r) for r in rows]

# ---------------- UPDATE ----------------
@app.put("/update-status")
def update_status(
    tracking_id: str = Form(...),
    field: str = Form(...),
    value: str = Form(...)
):
    with get_conn() as conn:
        conn.execute(f"""
        UPDATE candidates
        SET {field}=?, updated_at=?
        WHERE tracking_id=?
        """,(value, datetime.now(timezone.utc).isoformat(), tracking_id))

    return {"message":"updated"}

# ---------------- CANDIDATE ----------------
@app.get("/status/{tracking_id}")
def candidate_status(tracking_id:str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM candidates WHERE tracking_id=?",
            (tracking_id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Not found")

    return dict(row)

# ---------------- RUN ----------------
if __name__=="__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000)