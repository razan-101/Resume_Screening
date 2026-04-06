from __future__ import annotations

import os
import pickle
import sqlite3
import uuid
import json
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CLF_PATH = os.path.join(MODEL_DIR, "clf.pkl")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")
DB_PATH = os.path.join(BASE_DIR, "resume_screening.db")


# ---------------- LOAD MODEL ----------------
with open(CLF_PATH, "rb") as f:
    clf = pickle.load(f)

with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)


# ---------------- COMPANY ----------------
company_structure = {
    "Tech Solutions Pvt Ltd": {"roles": ["Java Developer","Python Developer","Web Designing","DevOps Engineer"],"recruiters": ["Arjun","Rohan","Vikram"]},
    "Data & Analytics Corp": {"roles": ["Data Science","Hadoop","ETL Developer"],"recruiters": ["Amit","Neha","Suresh"]},
}

category_mapping = {
    20: "Python Developer",
    6: "Data Science",
    15: "Java Developer",
    8: "DevOps Engineer",
}


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


def init_db():
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
            applied_at TEXT,
            updated_at TEXT
        )
        """)

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


# ---------------- API ----------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tenants")
def tenants():
    return {"tenants": company_structure}


@app.post("/candidates/upload-resume")
async def upload_resume(
    tenant: str = Form(...),
    cleaned_text: str = Form(...),
    resume: UploadFile = File(...)
):
    if len(cleaned_text.strip()) < 10:
        raise HTTPException(400, "Weak resume")

    features = tfidf.transform([cleaned_text])
    pred = int(clf.predict(features)[0])

    role = category_mapping.get(pred, "Unknown")
    score = compute_similarity(cleaned_text, company_structure[tenant]["roles"])

    eligible = role in company_structure[tenant]["roles"]
    recruiter = None

    with get_conn() as conn:
        if eligible:
            count = conn.execute("SELECT COUNT(*) FROM candidates WHERE tenant=?", (tenant,)).fetchone()[0]
            recruiter = company_structure[tenant]["recruiters"][count % 3]

        tracking_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn.execute("""
        INSERT INTO candidates VALUES (NULL,?,?,?,?,?,?,?,?,?)
        """, (
            tracking_id, tenant, role,
            int(eligible), "selected" if eligible else "rejected",
            recruiter, score, now, now
        ))

    return {
        "tracking_id": tracking_id,
        "predicted_role": role,
        "eligible": eligible,
        "ranking_score": score,
        "assigned_recruiter": recruiter
    }


@app.get("/interviewers/candidates")
def get_candidates():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM candidates WHERE eligible=1").fetchall()
    return [dict(r) for r in rows]


@app.post("/final-batch")
def save_batch(batch_name: str = Form(...), tenant: str = Form(...), data: str = Form(...)):
    candidates = json.loads(data)

    with get_conn() as conn:
        for c in candidates:
            conn.execute("""
            INSERT INTO final_batches VALUES (NULL,?,?,?,?,?,?,?)
            """, (
                batch_name, tenant,
                c["tracking_id"], c["predicted_role"],
                c["ranking_score"], c["assigned_recruiter"],
                datetime.now().isoformat()
            ))

    return {"msg": "saved"}


@app.get("/candidates/status/{tid}")
def status(tid: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM candidates WHERE tracking_id=?", (tid,)).fetchone()
    return dict(row) if row else {"error": "not found"}


if __name__ == "__main__":
    uvicorn.run("backend:app", port=8000, reload=True)