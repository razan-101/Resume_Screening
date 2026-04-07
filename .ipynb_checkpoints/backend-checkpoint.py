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
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CLF_PATH = os.path.join(MODEL_DIR, "clf.pkl")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")
DB_PATH = os.path.join(BASE_DIR, "resume_screening.db")


# ---------------- FULL COMPANY STRUCTURE ----------------
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


# ---------------- FULL CATEGORY MAPPING ----------------
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
    0: "Advocate",
}


# ---------------- LOAD MODEL ----------------
if not os.path.exists(CLF_PATH) or not os.path.exists(TFIDF_PATH):
    raise RuntimeError("Model files missing in /model folder")

with open(CLF_PATH, "rb") as f:
    clf = pickle.load(f)

with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)


# ---------------- FASTAPI ----------------
app = FastAPI(title="Resume Screening Backend")

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


def init_db():
    with get_conn() as conn:
        # MAIN TABLE
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

        # FINAL BATCH TABLE
        conn.execute("""
        CREATE TABLE IF NOT EXISTS final_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_name TEXT,
            tenant TEXT,
            tracking_id TEXT,
            predicted_role TEXT,
            ranking_score REAL,
            assigned_recruiter TEXT,
            created_at TEXT
        )
        """)


@app.on_event("startup")
def startup():
    init_db()


# ---------------- SIMILARITY ----------------
def compute_similarity(text, roles):
    try:
        vec = tfidf.transform([text])
        return float(vec.sum())
    except:
        return 0.0


# ---------------- BASIC APIs ----------------
@app.get("/")
def root():
    return {"message": "Backend running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tenants")
def tenants():
    return {"tenants": company_structure}


# ---------------- UPLOAD ----------------
@app.post("/candidates/upload-resume")
async def upload_resume(
    tenant: str = Form(...),
    cleaned_text: str = Form(...),
    resume: UploadFile = File(...)
):
    if tenant not in company_structure:
        raise HTTPException(400, "Invalid tenant")

    if len(cleaned_text.strip()) < 10:
        raise HTTPException(400, "Weak resume")

    features = tfidf.transform([cleaned_text])
    pred = int(clf.predict(features)[0])
    predicted_role = category_mapping.get(pred, "Unknown")

    roles = company_structure[tenant]["roles"]
    ranking_score = compute_similarity(cleaned_text, roles)

    eligible = predicted_role in roles
    recruiter = None

    with get_conn() as conn:
        if eligible:
            count = conn.execute(
                "SELECT COUNT(*) FROM candidates WHERE tenant=? AND eligible=1",
                (tenant,)
            ).fetchone()[0]

            recruiter = company_structure[tenant]["recruiters"][count % len(company_structure[tenant]["recruiters"])]

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
            tracking_id, tenant, predicted_role,
            int(eligible),
            "selected_for_interview" if eligible else "rejected",
            recruiter, ranking_score,
            now, now
        ))

    return {
        "tracking_id": tracking_id,
        "predicted_role": predicted_role,
        "eligible": eligible,
        "ranking_score": ranking_score,
        "assigned_recruiter": recruiter
    }


# ---------------- FILTERED INTERVIEWER DATA ----------------
@app.get("/interviewers/candidates")
def get_candidates(
    tenant: str = Query(...),
    interviewer: str = Query(...)
):
    with get_conn() as conn:
        rows = conn.execute("""
        SELECT * FROM candidates
        WHERE tenant=? AND assigned_recruiter=? AND eligible=1
        ORDER BY ranking_score DESC
        """, (tenant, interviewer)).fetchall()

    return [dict(r) for r in rows]


# ---------------- UPDATE STATUS ----------------
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
        SET test_status=?, interview_status=?, final_status=?, updated_at=?
        WHERE tracking_id=?
        """, (
            test_status,
            interview_status,
            final_status,
            datetime.now(timezone.utc).isoformat(),
            tracking_id
        ))

    return {"message": "Status updated"}


# ---------------- FINAL BATCH ----------------
@app.post("/final-batch")
def save_batch(
    batch_name: str = Form(...),
    tenant: str = Form(...),
    data: str = Form(...)
):
    candidates = json.loads(data)

    with get_conn() as conn:
        for c in candidates:
            conn.execute("""
            INSERT INTO final_batches (
                batch_name, tenant, tracking_id,
                predicted_role, ranking_score,
                assigned_recruiter, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                batch_name,
                tenant,
                c["tracking_id"],
                c["predicted_role"],
                c["ranking_score"],
                c["assigned_recruiter"],
                datetime.now(timezone.utc).isoformat()
            ))

    return {"message": "Batch saved successfully"}


# ---------------- STATUS ----------------
@app.get("/candidates/status/{tid}")
def status(tid: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM candidates WHERE tracking_id=?",
            (tid,)
        ).fetchone()

    if not row:
        return {"error": "not found"}

    return dict(row)


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port)