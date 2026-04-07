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
interviewer_passwords = {
    "Arjun – Backend Team": "arjun123",
    "Rohan – Frontend Team": "rohan123",
    "Vikram – DevOps Team": "vikram123",
    "Amit – Data Engineering": "amit123",
    "Neha – Analytics": "neha123",
    "Suresh – Program Management": "suresh123",
    "Priya – QA": "priya123",
    "Karan – Automation": "karan123",
    "Rahul – Security": "rahul123",
    "Ankit – Sales": "ankit123",
    "Pooja – HR": "pooja123",
    "Manish – Legal": "manish123",
    "Rajesh – Mechanical": "rajesh123",
    "Deepak – Electrical": "deepak123",
    "Sneha – Civil": "sneha123",
    "Meera – Creative": "meera123",
    "Kavya – Wellness": "kavya123",
    "Ananya – Lifestyle": "ananya123",
}

# ---------------- LOGIN ----------------
def login_page():
    st.title("🔐 Interviewer Login")

    tenants_data = requests.get(f"{BACKEND_URL}/tenants").json()["tenants"]

    tenant = st.selectbox("Select Company", list(tenants_data.keys()))
    interviewer = st.selectbox("Select Your Name", tenants_data[tenant]["recruiters"])
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if interviewer in interviewer_passwords and pwd == interviewer_passwords[interviewer]:
            st.session_state.logged_in = True
            st.session_state.user = interviewer
            st.session_state.tenant = tenant
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

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
    return re.sub(r"[^a-zA-Z ]", " ", text).lower()

def extract_pii(text):
    email = re.findall(r"\S+@\S+", text)
    phone = re.findall(r"\+?\d[\d -]{8,12}\d", text)
    name = text.split("\n")[0]
    return name, email[0] if email else "", phone[0] if phone else ""

def generate_hash(name, email, phone):
    return hashlib.sha256(f"{name}-{email}-{phone}".encode()).hexdigest()

# ---------------- MAIN ----------------
st.title("🚀 Privacy-Aware Resume Screening System")

tabs = st.tabs(["📂 Recruiter", "👨‍💼 Interviewer", "🧑 Candidate"])

# ================= RECRUITER =================
with tabs[0]:
    st.header("📂 Batch Resume Processing")

    tenants = list(requests.get(f"{BACKEND_URL}/tenants").json()["tenants"].keys())
    tenant = st.selectbox("🏢 Select Company", tenants)

    files = st.file_uploader("📄 Upload Resumes", accept_multiple_files=True)

    if st.button("⚡ Process Batch"):

        if not files:
            st.warning("Upload resumes first")
            st.stop()

        texts, raw_files, hashes = [], [], []
        seen = set()

        st.subheader("🔍 Duplicate Detection")

        for f in files:
            text = extract_text(f)
            name, email, phone = extract_pii(text)
            h = generate_hash(name, email, phone)

            if h in seen:
                st.error(f"❌ Exact duplicate removed → {f.name}")
                continue

            seen.add(h)
            texts.append(text)
            raw_files.append(f)
            hashes.append(h)

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(texts)
        sim = cosine_similarity(tfidf)

        keep, removed = [], set()

        for i in range(len(texts)):
            if i in removed:
                continue
            keep.append(i)

            for j in range(i+1, len(texts)):
                if sim[i][j] > 0.85:
                    removed.add(j)
                    st.warning(f"⚠️ Fuzzy duplicate removed → {raw_files[j].name}")

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

            if res.status_code != 200:
                st.error(f"Backend error → {f.name}")
                continue

            data = res.json()

            results.append({
                "tracking_id": data["tracking_id"],
                "UID": uid,
                "Role": data["predicted_role"],
                "Score": data["ranking_score"],
                "Eligible": data["eligible"],
                "Recruiter": data["assigned_recruiter"]
            })

        if results:
            df = pd.DataFrame(results).sort_values("Score", ascending=False)
            df["Rank"] = range(1, len(df)+1)

            st.dataframe(df[["Rank","UID","Role","Score","Eligible","Recruiter"]])

# ================= INTERVIEWER =================
with tabs[1]:

    if not st.session_state.logged_in:
        login_page()
    else:
        st.header(f"👨‍💼 {st.session_state.user}")

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

        res = requests.get(
            f"{BACKEND_URL}/interviewers/candidates",
            params={
                "tenant": st.session_state.tenant,
                "interviewer": st.session_state.user
            }
        )

        data = res.json()

        if not data:
            st.info("No candidates assigned")
        else:
            for c in data:
                st.subheader(c["tracking_id"])

                col1, col2, col3 = st.columns(3)

                with col1:
                    test = st.selectbox("Test", ["pending","completed"], key=f"t{c['tracking_id']}")
                with col2:
                    interview = st.selectbox("Interview", ["pending","completed"], key=f"i{c['tracking_id']}")
                with col3:
                    final = st.selectbox("Final", ["pending","selected","rejected"], key=f"f{c['tracking_id']}")

                if st.button(f"Update {c['tracking_id']}"):
                    requests.put(
                        f"{BACKEND_URL}/interview/update-status",
                        data={
                            "tracking_id": c["tracking_id"],
                            "test_status": test,
                            "interview_status": interview,
                            "final_status": final
                        }
                    )
                    st.success("Updated")

# ================= CANDIDATE =================
with tabs[2]:
    st.header("🧑 Candidate Tracking")

    tid = st.text_input("Enter Tracking ID")

    if st.button("Check Status"):
        res = requests.get(f"{BACKEND_URL}/candidates/status/{tid}")

        if res.status_code == 200:
            d = res.json()
            st.write("Role:", d.get("predicted_role"))
            st.write("Test:", d.get("test_status"))
            st.write("Interview:", d.get("interview_status"))
            st.write("Final:", d.get("final_status"))
        else:
            st.error("Not found")