import io, re, hashlib, json, requests, streamlit as st, pdfplumber, docx, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BACKEND_URL = "https://resume-backend-i634.onrender.com"

st.set_page_config(page_title="Resume Screening", layout="wide")

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "tenant" not in st.session_state:
    st.session_state.tenant = None

# ---------------- PASSWORDS ----------------
passwords = {
    "Tech Solutions Pvt Ltd":{
        "Arjun – Backend Team":"arjun1234",
        "Rohan – Frontend Team":"rohan1234",
        "Vikram – DevOps Team":"vikram1234"
    },
    "Data & Analytics Corp":{
        "Amit – Data Engineering":"amit1234",
        "Neha – Analytics":"neha1234",
        "Suresh – Program Management":"suresh1234"
    },
    "Quality & Security Systems Ltd":{
        "Priya – QA":"priya1234",
        "Karan – Automation":"karan1234",
        "Rahul – Security":"rahul1234"
    },
    "Enterprise & Sales Solutions":{
        "Ankit – Sales":"ankit1234",
        "Pooja – HR":"pooja1234",
        "Manish – Legal":"manish1234"
    },
    "Engineering & Manufacturing Group":{
        "Rajesh – Mechanical":"rajesh1234",
        "Deepak – Electrical":"deepak1234",
        "Sneha – Civil":"sneha1234"
    },
    "Creative & Wellness Services":{
        "Meera – Creative":"meera1234",
        "Kavya – Wellness":"kavya1234",
        "Ananya – Lifestyle":"ananya1234"
    }
}

# ---------------- UTIL ----------------
def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file.getvalue())) as pdf:
            return "".join(p.extract_text() or "" for p in pdf.pages)
    elif file.name.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file.getvalue()))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return file.getvalue().decode(errors="ignore")

def clean_text(text):
    return re.sub(r"[^a-zA-Z ]"," ",text).lower()

def extract_pii(text):
    email = re.findall(r"\S+@\S+", text)
    phone = re.findall(r"\+?\d[\d -]{8,12}\d", text)
    name = text.split("\n")[0]
    return name, email[0] if email else "", phone[0] if phone else ""

def generate_hash(name,email,phone):
    return hashlib.sha256(f"{name}-{email}-{phone}".encode()).hexdigest()

# ---------------- UI ----------------
st.title("🚀 Privacy-Aware Resume Screening System")
tabs = st.tabs(["📂 Recruiter","👨‍💼 Interviewer","🧑 Candidate"])

# ================= RECRUITER =================
with tabs[0]:
    st.header("📂 Batch Resume Processing")

    res = requests.get(f"{BACKEND_URL}/tenants")

    if res.status_code == 200:
        tenants = list(res.json().get("tenants", {}).keys())
    else:
        st.error("Backend not working")
        tenants = []

    tenant = st.selectbox("🏢 Select Company", tenants)

    files = st.file_uploader("📄 Upload Resumes", accept_multiple_files=True)

    if st.button("⚡ Process Batch"):

        texts, raw_files, hashes = [], [], []
        seen = set()

        for f in files:
            text = extract_text(f)
            name,email,phone = extract_pii(text)
            h = generate_hash(name,email,phone)

            if h in seen:
                st.error(f"Duplicate → {f.name}")
                continue

            seen.add(h)
            texts.append(text)
            raw_files.append(f)
            hashes.append(h)

        if texts:
            tfidf_matrix = TfidfVectorizer().fit_transform(texts)
            sim = cosine_similarity(tfidf_matrix)
        else:
            sim = []

        keep, removed = [], set()

        for i in range(len(texts)):
            if i in removed: continue
            keep.append(i)

            for j in range(i+1,len(texts)):
                if sim[i][j] > 0.85:
                    removed.add(j)
                    st.warning(f"Fuzzy duplicate → {raw_files[j].name}")

        results = []

        for i in keep:
            f = raw_files[i]
            uid = hashes[i][:8]
            clean = clean_text(texts[i])

            res = requests.post(
                f"{BACKEND_URL}/candidates/upload-resume",
                data={"tenant": tenant, "cleaned_text": clean},
                files={"resume": (f.name, f.getvalue())}
            )

            data = res.json()

            results.append({
                "tracking_id": data.get("tracking_id"),
                "UID": uid,
                "Role": data.get("predicted_role"),
                "Score": data.get("ranking_score"),
                "Eligible": data.get("eligible"),
                "Recruiter": data.get("assigned_recruiter")
            })

        if results:
            df = pd.DataFrame(results).sort_values("Score",ascending=False)
            st.dataframe(df, use_container_width=True)

# ================= INTERVIEWER =================
with tabs[1]:
    st.header("👨‍💼 Interviewer Dashboard")

    if not st.session_state.logged_in:
        tenant = st.selectbox("Company", list(passwords.keys()))
        interviewer = st.selectbox("Interviewer", list(passwords[tenant].keys()))
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if pwd == passwords[tenant][interviewer]:
                st.session_state.logged_in = True
                st.session_state.user = interviewer
                st.session_state.tenant = tenant
                st.rerun()
            else:
                st.error("Wrong password")
    else:
        st.success(f"Logged in as {st.session_state.user}")

        res = requests.get(
            f"{BACKEND_URL}/interviewers/candidates",
            params={
                "tenant": st.session_state.tenant,
                "interviewer": st.session_state.user
            }
        )

        data = res.json()

        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

# ================= CANDIDATE =================
with tabs[2]:
    st.header("🧑 Candidate Tracking")

    tid = st.text_input("Tracking ID")

    if st.button("Check"):
        res = requests.get(f"{BACKEND_URL}/candidates/status/{tid}")

        if res.status_code == 200:
            st.json(res.json())
        else:
            st.error("Not found")