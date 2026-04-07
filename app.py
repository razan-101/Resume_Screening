import io, re, hashlib, json, requests, streamlit as st, pdfplumber, docx, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BACKEND_URL = "https://resume-backend-i634.onrender.com"

st.set_page_config(page_title="Resume Screening", layout="wide")

# SESSION
if "logged_in" not in st.session_state:
    st.session_state.logged_in=False
if "user" not in st.session_state:
    st.session_state.user=None
if "tenant" not in st.session_state:
    st.session_state.tenant=None

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


# UTIL
def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file.getvalue())) as pdf:
            return "".join(p.extract_text() or "" for p in pdf.pages)
    elif file.name.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file.getvalue()))
        return "\n".join(p.text for p in doc.paragraphs)
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

st.title("🚀 Privacy-Aware Resume Screening System")
tabs = st.tabs(["📂 Recruiter","👨‍💼 Interviewer","🧑 Candidate"])

# ================= RECRUITER =================
with tabs[0]:
    res = requests.get(f"{BACKEND_URL}/tenants")
    tenants = list(res.json().get("tenants",{}).keys())

    tenant = st.selectbox("🏢 Company", tenants)
    batch_name = st.text_input("📦 Batch Name")

    files = st.file_uploader("Upload Resumes", accept_multiple_files=True)

    if st.button("⚡ Process Batch"):
        progress = st.progress(0)
        status = st.empty()

        results=[]
        for i,f in enumerate(files):
            progress.progress((i+1)/len(files))
            status.text(f"Processing {f.name}")

            text=extract_text(f)
            clean=clean_text(text)

            res=requests.post(
                f"{BACKEND_URL}/candidates/upload-resume",
                data={"tenant":tenant,"batch_name":batch_name,"cleaned_text":clean},
                files={"resume":(f.name,f.getvalue())}
            )

            results.append(res.json())

        st.success("Processing Done")
        st.dataframe(pd.DataFrame(results))

# ================= INTERVIEWER =================
with tabs[1]:
    if not st.session_state.logged_in:
        tenant = st.selectbox("Company", list(passwords.keys()))
        interviewer = st.selectbox("Interviewer", list(passwords[tenant].keys()))
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if pwd == passwords[tenant][interviewer]:
                st.session_state.logged_in=True
                st.session_state.user=interviewer
                st.session_state.tenant=tenant
                st.rerun()

    else:
        data = requests.get(
            f"{BACKEND_URL}/interviewers/candidates",
            params={"tenant":st.session_state.tenant,"interviewer":st.session_state.user}
        ).json()

        for c in data:
            st.write(c["tracking_id"])

            if c["test_status"]=="pending":
                if st.button("Test Done", key=c["tracking_id"]+"t"):
                    requests.put(f"{BACKEND_URL}/interview/update-status",
                        data={"tracking_id":c["tracking_id"],"test_status":"done","interview_status":"pending","final_status":"pending"})
                    st.rerun()

            elif c["interview_status"]=="pending":
                if st.button("Interview Done", key=c["tracking_id"]+"i"):
                    requests.put(f"{BACKEND_URL}/interview/update-status",
                        data={"tracking_id":c["tracking_id"],"test_status":"done","interview_status":"done","final_status":"pending"})
                    st.rerun()

            else:
                col1,col2=st.columns(2)
                if col1.button("Select", key=c["tracking_id"]+"s"):
                    requests.put(f"{BACKEND_URL}/interview/update-status",
                        data={"tracking_id":c["tracking_id"],"test_status":"done","interview_status":"done","final_status":"selected"})
                    st.rerun()
                if col2.button("Reject", key=c["tracking_id"]+"r"):
                    requests.put(f"{BACKEND_URL}/interview/update-status",
                        data={"tracking_id":c["tracking_id"],"test_status":"done","interview_status":"done","final_status":"rejected"})
                    st.rerun()

# ================= CANDIDATE =================
with tabs[2]:
    tid = st.text_input("Tracking ID")

    if st.button("Check"):
        res = requests.get(f"{BACKEND_URL}/candidates/status/{tid}")

        if res.status_code==200:
            d=res.json()

            st.write("Test:", d["test_status"])
            st.write("Interview:", d["interview_status"])
            st.write("Final:", d["final_status"])

            if d["final_status"]=="selected":
                st.success("Selected")
            elif d["final_status"]=="rejected":
                st.error("Rejected")
            else:
                st.info("In Progress")