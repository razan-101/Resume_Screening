from __future__ import annotations

import os, pickle, sqlite3, uuid
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

# ---------------- CATEGORY ----------------
category_mapping = {
    15:"Java Developer",23:"Testing",8:"DevOps Engineer",20:"Python Developer",
    24:"Web Designing",12:"HR",13:"Hadoop",3:"Blockchain",10:"ETL Developer",
    18:"Operations Manager",6:"Data Science",22:"Sales",16:"Mechanical Engineer",
    1:"Arts",7:"Database",11:"Electrical Engineering",14:"Health and Fitness",
    19:"PMO",4:"Business Analyst",9:"DotNet Developer",2:"Automation Testing",
    17:"Network Security Engineer",21:"SAP Developer",5:"Civil Engineer",0:"Advocate"
}

# ---------------- LOAD MODEL ----------------
with open(CLF_PATH,"rb") as f: clf=pickle.load(f)
with open(TFIDF_PATH,"rb") as f: tfidf=pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DB ----------------
def get_conn():
    conn=sqlite3.connect(DB_PATH)
    conn.row_factory=sqlite3.Row
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
            assigned_recruiter TEXT,
            hybrid_score REAL,
            rank INTEGER,
            test_status TEXT DEFAULT 'pending',
            interview_status TEXT DEFAULT 'pending',
            final_status TEXT DEFAULT 'pending',
            interview_started INTEGER DEFAULT 0,
            applied_at TEXT
        )
        """)

# ---------------- TENANTS ----------------
@app.get("/tenants")
def tenants():
    return {"tenants":company_structure}

# ---------------- SAVE FINAL ----------------
@app.post("/save-final")
def save_final(data: list = Form(...), tenant: str = Form(...)):
    data = eval(data)
    with get_conn() as conn:
        for d in data:
            conn.execute("""
            INSERT OR IGNORE INTO candidates (
                tracking_id, tenant, predicted_role,
                eligible, assigned_recruiter,
                hybrid_score, rank, applied_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,(
                d["tracking_id"], tenant, d["Role"],
                int(d["Eligible"]), d["Recruiter"],
                d["Score"], d["Rank"],
                datetime.now(timezone.utc).isoformat()
            ))
    return {"msg":"saved"}

# ---------------- INTERVIEWER ----------------
@app.get("/interviewers/candidates")
def get_candidates(tenant:str=Query(...),interviewer:str=Query(...)):
    with get_conn() as conn:
        rows=conn.execute("""
        SELECT * FROM candidates
        WHERE tenant=? AND assigned_recruiter=? AND eligible=1
        """,(tenant,interviewer)).fetchall()
    return [dict(r) for r in rows]

# ---------------- UPDATE ----------------
@app.put("/update-status")
def update_status(tracking_id:str=Form(...),field:str=Form(...),value:str=Form(...)):
    with get_conn() as conn:
        conn.execute(f"UPDATE candidates SET {field}=? WHERE tracking_id=?",(value,tracking_id))
    return {"msg":"ok"}

# ---------------- STATUS ----------------
@app.get("/status/{tid}")
def status(tid:str):
    with get_conn() as conn:
        r=conn.execute("SELECT * FROM candidates WHERE tracking_id=?",(tid,)).fetchone()
    if not r: raise HTTPException(404)
    return dict(r)

if __name__=="__main__":
    uvicorn.run("backend:app",host="0.0.0.0",port=8000)