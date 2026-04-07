from __future__ import annotations

import io
import os
import pickle
import random
import re
import sqlite3
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, List

import docx
import pdfplumber
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CLF_PATH = os.path.join(MODEL_DIR, "clf.pkl")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")
DB_PATH = os.path.join(BASE_DIR, "resume_screening.db")

company_structure = {
    "Tech Solutions Pvt Ltd": {
        "roles": ["Java Developer", "Python Developer", "Web Designing", "DotNet Developer", "Database", "DevOps Engineer", "Blockchain", "SAP Developer"],
        "recruiters": ["Aravind – Backend Team", "Ishaan – Frontend Team", "Advait – DevOps Team"]
    },
    "Data & Analytics Corp": {
        "roles": ["Data Science", "Hadoop", "ETL Developer", "Business Analyst", "Operations Manager", "PMO"],
        "recruiters": ["Vihaan – Data Engineering", "Ananya – Analytics", "Pranav – Program Management"]
    },
    "Quality & Security Systems Ltd": {
        "roles": ["Testing", "Automation Testing", "Network Security Engineer"],
        "recruiters": ["Kritika – QA", "Rishi – Automation", "Siddharth – Security"]
    },
    "Enterprise & Sales Solutions": {
        "roles": ["Sales", "HR", "Advocate"],
        "recruiters": ["Varun – Sales", "Tanvi – HR", "Aditya – Legal"]
    },
    "Engineering & Manufacturing Group": {
        "roles": ["Mechanical Engineer", "Electrical Engineering", "Civil Engineer"],
        "recruiters": ["Kartik – Mechanical", "Arnav – Electrical", "Saanvi – Civil"]
    },
    "Creative & Wellness Services": {
        "roles": ["Arts", "Health and Fitness"],
        "recruiters": ["Kiara – Creative", "Zoya – Wellness", "Vivaan – Lifestyle"]
    }
}

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


# ---------------- ROLE KEYWORDS FOR HYBRID SIMILARITY ----------------
role_keywords = {
    "Java Developer": "java spring hibernate maven microservices multithreading j2ee sql",
    "Testing": "testing selenium manual automation junit testng bugzilla jira",
    "DevOps Engineer": "devops docker kubernetes jenkins aws azure terraform ansible git",
    "Python Developer": "python django flask fastapi pandas numpy scrapy restful celery",
    "Web Designing": "html css javascript react angular bootstrap ui ux design jquery",
    "HR": "recruitment human resources payroll onboarding compliance hiring employee relations",
    "Hadoop": "hadoop spark hive hbase mapreduce big data cloudera flume sqoop",
    "Blockchain": "blockchain ethereum smart contracts solidity crypto bitcoin decentralization dlt",
    "ETL Developer": "etl informatica data warehousing talend sql server ssis pentaho datastage",
    "Operations Manager": "operations management supply chain logistics strategy budget process optimization",
    "Data Science": "data science machine learning python scikit-learn statistics visualization deep learning",
    "Sales": "sales marketing lead generation crm negotiation business development account management",
    "Mechanical Engineer": "mechanical engineering cad autocad solidworks thermodynamics manufacturing fluid mechanics",
    "Arts": "fine arts digital art creative design illustration photography graphic design",
    "Database": "database sql oracle mysql postgresql mongodb dba performance tuning backup",
    "Electrical Engineering": "electrical engineering circuits power systems matlab pcb design control systems",
    "Health and Fitness": "health fitness personal training nutrition wellness physical therapy exercise science",
    "PMO": "pmo project management planning governance reporting risk management stakeholders",
    "Business Analyst": "business analysis requirements gathering agile scrum data modeling jira",
    "DotNet Developer": "dotnet c# asp.net mvc web api entity framework sql server wcf",
    "Automation Testing": "automation selenium appium cucumber jenkins testng soapui",
    "Network Security Engineer": "network security firewalls vpn cisco ids ips cybersecurity pentesting",
    "SAP Developer": "sap abap hana erp basis fico mm sd bi bw",
    "Civil Engineer": "civil engineering construction structural design autocad surveying project management",
    "Advocate": "legal law litigation research corporate law contracts compliance drafting"
}

# ---------------- HYBRID SIMILARITY ----------------
def hybrid_similarity(text, role):
    keywords = role_keywords.get(role, "")
    if not keywords: return 0.5
    
    # Cosine Similarity
    v_text = tfidf.transform([text])
    v_keys = tfidf.transform([keywords])
    cos_sim = cosine_similarity(v_text, v_keys)[0][0]

    # Jaccard Similarity
    words_text = set(text.split())
    words_keywords = set(keywords.split())
    intersection = len(words_text.intersection(words_keywords))
    union = len(words_text.union(words_keywords))
    jac_sim = intersection / union if union > 0 else 0

    return (cos_sim + jac_sim) / 2

if not (os.path.exists(CLF_PATH) and os.path.exists(TFIDF_PATH)):
    raise RuntimeError("Model files not found in the model directory.")

with open(CLF_PATH, "rb") as clf_file:
    clf = pickle.load(clf_file)

with open(TFIDF_PATH, "rb") as tfidf_file:
    tfidf = pickle.load(tfidf_file)


class ScreeningResponse(BaseModel):
    candidate_id: Optional[int] = None
    tracking_id: str
    tenant: str
    predicted_role: str
    eligible: bool
    process_status: str
    assigned_recruiter: Optional[str] = None
    ranking_score: float
    selection_rank: Optional[int] = None
    applied_at: Optional[str] = None
    cleaned_text: Optional[str] = None


class InterviewerCandidateResponse(BaseModel):
    id: int
    tracking_id: str
    tenant: str
    predicted_role: str
    assigned_recruiter: str
    ranking_score: float
    selection_rank: Optional[int] = None
    applied_at: str
    process_status: str
    test_status: str
    interview_status: str
    final_status: str


class CandidateStatusResponse(BaseModel):
    id: int
    tracking_id: str
    tenant: str
    predicted_role: str
    eligible: bool
    process_status: str
    test_status: str
    interview_status: str
    interview_time: Optional[str] = None
    final_status: str
    assigned_recruiter: Optional[str] = None
    selection_rank: Optional[int] = None
    applied_at: str
    updated_at: str
    cleaned_text: Optional[str] = None


app = FastAPI(
    title="Resume Screening Backend",
    version="1.0.0",
    description="Backend APIs for resume upload, candidate selection, interviewer ranking, and candidate status tracking.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tracking_id TEXT NOT NULL UNIQUE,
                tenant TEXT NOT NULL,
                predicted_role TEXT NOT NULL,
                eligible INTEGER NOT NULL,
                process_status TEXT NOT NULL,
                test_status TEXT DEFAULT 'pending',
                interview_status TEXT DEFAULT 'pending',
                interview_time TEXT,
                final_status TEXT DEFAULT 'pending',
                assigned_recruiter TEXT,
                ranking_score REAL NOT NULL,
                selection_rank INTEGER,
                cleaned_text TEXT,
                applied_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def remove_pii(text: str) -> str:
    # Mask Emails
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", " ", text)
    # Mask Phone Numbers
    text = re.sub(r"(\+?\d[\d -]{8,12}\d)", " ", text)
    # Mask LinkedIn/URLs
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"linkedin\.com/\S+", " ", text)
    
    # Mask common location keywords (e.g., Odisha, India)
    locations = ["odisha", "india", "bhubaneswar", "bangalore", "hyderabad", "pune", "mumbai", "delhi"]
    for loc in locations:
        text = re.sub(rf"\b{loc}\b", " ", text, flags=re.IGNORECASE)
        
    # Mask common resume header labels that clutter word clouds
    labels = ["name", "phone", "email", "linkedin", "location", "address", "mobile", "contact"]
    for label in labels:
        text = re.sub(rf"\b{label}\b", " ", text, flags=re.IGNORECASE)

    # Name removal heuristic: 
    # Usually the name is at the top. We'll strip the first line if it looks like a name header
    lines = text.split('\n')
    if lines and len(lines[0].split()) < 5: # Heuristic for a name header
        lines[0] = " "
    
    return "\n".join(lines)


def clean_resume(text: str) -> str:
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def extract_text_from_upload(file_name: str, content: bytes) -> str:
    extension = os.path.splitext(file_name)[1].lower()
    if extension == ".txt":
        return content.decode("utf-8", errors="ignore")
    if extension == ".pdf":
        text = ""
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    if extension == ".docx":
        document = docx.Document(io.BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    raise HTTPException(status_code=400, detail="Unsupported file type. Use TXT, PDF, or DOCX.")


def classify_resume(cleaned_text: str) -> tuple[str, float]:
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Resume content could not be extracted.")

    features = tfidf.transform([cleaned_text])
    prediction_id = int(clf.predict(features)[0])
    predicted_role = category_mapping.get(prediction_id, "Unknown")
    ranking_score = 1.0

    if hasattr(clf, "predict_proba") and hasattr(clf, "classes_"):
        probabilities = clf.predict_proba(features)[0]
        class_indexes = {int(label): index for index, label in enumerate(clf.classes_)}
        predicted_index = class_indexes.get(prediction_id)
        if predicted_index is not None:
            ranking_score = float(probabilities[predicted_index])
        else:
            ranking_score = float(max(probabilities))

    return predicted_role, round(ranking_score, 4)


def assign_recruiter(tenant: str) -> str:
    recruiters = company_structure[tenant]["recruiters"]
    return random.choice(recruiters)


def recalculate_ranks(connection: sqlite3.Connection, tenant: str) -> None:
    selected_candidates = connection.execute(
        """
        SELECT id
        FROM candidates
        WHERE tenant = ? AND eligible = 1
        ORDER BY ranking_score DESC, applied_at ASC, id ASC
        """,
        (tenant,),
    ).fetchall()

    connection.execute(
        "UPDATE candidates SET selection_rank = NULL WHERE tenant = ?",
        (tenant,),
    )

    for index, row in enumerate(selected_candidates, start=1):
        connection.execute(
            "UPDATE candidates SET selection_rank = ? WHERE id = ?",
            (index, row["id"]),
        )


def serialize_candidate(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "tracking_id": row["tracking_id"],
        "tenant": row["tenant"],
        "predicted_role": row["predicted_role"],
        "eligible": bool(row["eligible"]),
        "process_status": row["process_status"],
        "test_status": row["test_status"],
        "interview_status": row["interview_status"],
        "interview_time": row["interview_time"],
        "final_status": row["final_status"],
        "assigned_recruiter": row["assigned_recruiter"],
        "ranking_score": round(float(row["ranking_score"]), 4),
        "selection_rank": row["selection_rank"],
        "cleaned_text": row["cleaned_text"],
        "applied_at": row["applied_at"],
        "updated_at": row["updated_at"],
    }


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.get("/")
def read_root() -> dict:
    return {
        "message": "Resume screening backend is running.",
        "endpoints": {
            "tenants": "/tenants",
            "upload_resume": "/candidates/upload-resume",
            "final_batch": "/final-batch",
            "interviewer_candidates": "/interviewers/candidates",
            "candidate_status": "/candidates/status/{tracking_id}",
            "candidate_details": "/candidates/details/{tracking_id}",
            "update_status": "/interview/update-status"
        },
    }


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/tenants")
def list_tenants() -> dict:
    return {"tenants": company_structure}


@app.post("/candidates/upload-resume", response_model=ScreeningResponse)
async def upload_resume(
    full_name: str = Form(...),
    email: str = Form(...),
    tenant: str = Form(...),
    resume: UploadFile = File(...),
    tracking_id: Optional[str] = Form(None)
) -> ScreeningResponse:
    normalized_tenant = tenant.strip()

    if normalized_tenant not in company_structure:
        raise HTTPException(status_code=400, detail="Invalid tenant selected.")

    file_content = await resume.read()
    resume_text = extract_text_from_upload(resume.filename or "resume.txt", file_content)
    private_text = remove_pii(resume_text)
    cleaned_text = clean_resume(private_text)
    
    # AI Role Prediction
    features = tfidf.transform([cleaned_text])
    prediction_id = int(clf.predict(features)[0])
    predicted_role = category_mapping.get(prediction_id, "Unknown")
    
    # Hybrid Similarity Score
    ranking_score = hybrid_similarity(cleaned_text, predicted_role)

    company_roles = company_structure[normalized_tenant]["roles"]
    eligible = predicted_role in company_roles
    
    # Assign recruiter/interviewer placeholder for the preview table
    recruiters = company_structure[normalized_tenant]["recruiters"]
    assigned_recruiter = recruiters[0] # Default placeholder, will be re-assigned fairly in batch

    final_tid = tracking_id if tracking_id else str(uuid.uuid4())[:8]

    return ScreeningResponse(
        tracking_id=final_tid,
        tenant=normalized_tenant,
        predicted_role=predicted_role,
        eligible=eligible,
        process_status="selected_for_interview" if eligible else "rejected_in_screening",
        ranking_score=round(ranking_score, 4),
        assigned_recruiter=assigned_recruiter,
        cleaned_text=cleaned_text
    )


@app.post("/final-batch")
def final_batch(
    tenant: str = Form(...),
    data: str = Form(...)
):
    try:
        data_list = json.loads(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON data")

    recruiters = company_structure.get(tenant, {}).get("recruiters", [])
    if not recruiters:
        raise HTTPException(status_code=400, detail="Company not found")

    with get_connection() as connection:
        print(f"DEBUG: Saving batch of {len(data_list)} candidates for tenant {tenant}")
        for i, d in enumerate(data_list):
            # Force Round-Robin distribution among the recruiters for this tenant
            recruiter = recruiters[i % len(recruiters)]
            now = datetime.now(timezone.utc).isoformat()
            
            connection.execute(
                """
                INSERT OR IGNORE INTO candidates (
                    tracking_id, tenant, predicted_role, eligible,
                    process_status, assigned_recruiter, ranking_score,
                    cleaned_text, applied_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["tracking_id"], tenant, d["predicted_role"], int(d["eligible"]),
                    d["process_status"], recruiter, d["ranking_score"],
                    d.get("cleaned_text", ""), now, now
                )
            )
        
        recalculate_ranks(connection, tenant)
    
    return {"message": "Batch saved successfully"}


@app.put("/interview/update-status")
def update_interview_status(
    tracking_id: str = Form(...),
    test_status: str = Form(None),
    interview_status: str = Form(None),
    interview_time: str = Form(None),
    final_status: str = Form(None)
):
    with get_connection() as connection:
        updates = []
        params = []
        if test_status:
            updates.append("test_status=?")
            params.append(test_status)
        if interview_status:
            updates.append("interview_status=?")
            params.append(interview_status)
        if interview_time:
            updates.append("interview_time=?")
            params.append(interview_time)
        if final_status:
            updates.append("final_status=?")
            params.append(final_status)
        
        if not updates:
            return {"message": "No fields to update"}
            
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(tracking_id)
        
        query = f"UPDATE candidates SET {', '.join(updates)}, updated_at=? WHERE tracking_id=?"
        connection.execute(query, params)

    return {"message": "Status updated"}


@app.get("/interviewers/candidates", response_model=list[InterviewerCandidateResponse])
def get_interviewer_candidates(
    tenant: Optional[str] = Query(default=None),
    interviewer: Optional[str] = Query(default=None),
) -> list[InterviewerCandidateResponse]:
    query = """
        SELECT *
        FROM candidates
        WHERE eligible = 1
    """
    parameters: list[str] = []

    if tenant:
        query += " AND tenant = ?"
        parameters.append(tenant)
    if interviewer:
        query += " AND assigned_recruiter = ?"
        parameters.append(interviewer)

    query += " ORDER BY tenant ASC, selection_rank ASC, ranking_score DESC, applied_at ASC"

    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [
        InterviewerCandidateResponse(
            id=row["id"],
            tracking_id=row["tracking_id"],
            tenant=row["tenant"],
            predicted_role=row["predicted_role"],
            assigned_recruiter=row["assigned_recruiter"],
            ranking_score=round(float(row["ranking_score"]), 4),
            selection_rank=row["selection_rank"],
            applied_at=row["applied_at"],
            process_status=row["process_status"],
            test_status=row["test_status"],
            interview_status=row["interview_status"],
            final_status=row["final_status"]
        )
        for row in rows
    ]


@app.get("/candidates/details/{tracking_id}", response_model=CandidateStatusResponse)
def get_candidate_details(tracking_id: str) -> CandidateStatusResponse:
    return get_candidate_status(tracking_id)


@app.get("/candidates/status/{tracking_id}", response_model=CandidateStatusResponse)
def get_candidate_status(tracking_id: str) -> CandidateStatusResponse:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM candidates WHERE tracking_id = ?",
            (tracking_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    return CandidateStatusResponse(**serialize_candidate(row))


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8001, reload=True)
