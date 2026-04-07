from __future__ import annotations

import os
import pickle
import sqlite3
import uuid
import json
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
    "Tech Solutions Pvt Ltd": {
        "roles": [
            "Java Developer", "Python Developer", "Web Designing",
            "DotNet Developer", "Database", "DevOps Engineer",
            "Blockchain", "SAP Developer"
        ],
        "recruiters": [
            "Arjun – Backend Team",
            "Rohan – Frontend Team",
            "Vikram – DevOps Team"
        ]
    },
    "Data & Analytics Corp": {
        "roles": [
            "Data Science", "Hadoop", "ETL Developer",
            "Business Analyst", "Operations Manager", "PMO"
        ],
        "recruiters": [
            "Amit – Data Engineering",
            "Neha – Analytics",
            "Suresh – Program Management"
        ]
    },
    "Quality & Security Systems Ltd": {
        "roles": [
            "Testing", "Automation Testing", "Network Security Engineer"
        ],
        "recruiters": [
            "Priya – QA",
            "Karan – Automation",
            "Rahul – Security"
        ]
    },
    "Enterprise & Sales Solutions": {
        "roles": ["Sales", "HR", "Advocate"],
        "recruiters": [
            "Ankit – Sales",
            "Pooja – HR",
            "Manish – Legal"
        ]
    },
    "Engineering & Manufacturing Group": {
        "roles": [
            "Mechanical Engineer", "Electrical Engineering", "Civil Engineer"
        ],
        "recruiters": [
            "Rajesh – Mechanical",
            "Deepak – Electrical",
            "Sneha – Civil"
        ]
    },
    "Creative & Wellness Services": {
        "roles": ["Arts", "Health and Fitness"],
        "recruiters": [
            "Meera – Creative",
            "Kavya – Wellness",
            "Ananya – Lifestyle"
        ]
    }
}

# ---------------- CATEGORY MAPPING ----------------
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and Fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate"
}

# ---------------- LOAD MODEL ----------------
with open(CLF_PATH, "rb") as f:
    clf = pickle.load(f)

with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)

# ---------------- FASTAPI ----------------
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
            predicted_role TEXT,
            eligible INTEGER,
            process_status TEXT,
            assigned_recruiter TEXT,
            ranking_score REAL,
            test_status TEXT DEFAULT 'pending',
            interview_status TEXT DEFAULT 'pending',
            final_status TEXT DEFAULT 'pending',
            applied_at TEXT,
            updated_at TEXT
        )
        """)

# ---------------- UPLOAD ----------------
@app.post("/candidates/upload-resume")
async def upload_resume(
    tenant: str = Form(...),
    cleaned_text: str = Form(...),
    resume: UploadFile = File(...)
):
    features = tfidf.transform([cleaned_text])
    pred = int(clf.predict(features)[0])
    predicted_role = category_mapping.get(pred, "Unknown")

    roles = company_structure[tenant]["roles"]
    eligible = predicted_role in roles

    recruiter = None

    with get_conn() as conn:
        if eligible:
            count = conn.execute(
                "SELECT COUNT(*) FROM candidates WHERE tenant=? AND eligible=1",
                (tenant,)
            ).fetchone()[0]

            recruiter = company_structure[tenant]["recruiters"][
                count % len(company_structure[tenant]["recruiters"])
            ]

        tracking_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn.execute("""
        INSERT INTO candidates (
            tracking_id, tenant, predicted_role,
            eligible, process_status,
            assigned_recruiter, ranking_score,
            applied_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tracking_id,
            tenant,
            predicted_role,
            int(eligible),
            "selected_for_interview" if eligible else "rejected",
            recruiter,
            0.0,
            now,
            now
        ))

    return {"tracking_id": tracking_id, "assigned_recruiter": recruiter}

# ---------------- INTERVIEWER ----------------
@app.get("/interviewers/candidates")
def get_candidates(tenant: str = Query(...), interviewer: str = Query(...)):
    with get_conn() as conn:
        rows = conn.execute("""
        SELECT * FROM candidates
        WHERE tenant=? AND assigned_recruiter=? AND eligible=1
        """, (tenant, interviewer)).fetchall()

    return [dict(r) for r in rows]

# ---------------- UPDATE ----------------
@app.put("/interview/update-status")
def update_status(
    tracking_id: str = Form(...),
    test_status: str = Form(...),
    interview_status: str = Form(...),
    final_status: str = Form(...)
):
    with get_conn() as conn:
        conn.execute("""
        UPDATE candidates
        SET test_status=?, interview_status=?, final_status=?
        WHERE tracking_id=?
        """, (test_status, interview_status, final_status, tracking_id))

    return {"message": "Status updated"}

# ---------------- RUN ----------------
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000)