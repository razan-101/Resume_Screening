import io,re,hashlib,json,requests,streamlit as st,pdfplumber,docx,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BACKEND_URL="https://resume-backend-i634.onrender.com"

st.set_page_config(page_title="Resume AI System",layout="wide")

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

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in=False
if "user" not in st.session_state:
    st.session_state.user=None
if "tenant" not in st.session_state:
    st.session_state.tenant=None

# ---------------- UTIL ----------------
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

def hash_id(text):
    return hashlib.md5(text.encode()).hexdigest()

# ---------------- UI ----------------
st.title("🚀 AI Resume Screening System")

tabs=st.tabs(["📂 Recruiter","👨‍💼 Interviewer","🧑 Candidate"])

# ================= RECRUITER =================
with tabs[0]:

    tenants=list(requests.get(f"{BACKEND_URL}/tenants").json()["tenants"].keys())

    tenant=st.selectbox("🏢 Company",tenants,key="recruiter_company")

    files=st.file_uploader("📄 Upload Resumes",accept_multiple_files=True)

    if st.button("⚡ Process Batch"):

        progress=st.progress(0)
        status=st.empty()

        texts=[]
        for i,f in enumerate(files):
            status.text(f"Processing {f.name}...")
            progress.progress((i+1)/len(files))
            texts.append(clean_text(extract_text(f)))

        tfidf=TfidfVectorizer().fit_transform(texts)
        cos=cosine_similarity(tfidf)

        results=[]
        for i,t in enumerate(texts):
            jac=len(set(t.split()))/len(t.split())
            score=(cos[i].mean()+jac)/2

            results.append({
                "tracking_id":hash_id(t)[:8],
                "Role":"Predicted",
                "Score":round(score,3),
                "Eligible":score>0.2,
                "Recruiter":"Auto"
            })

        df=pd.DataFrame(results).sort_values("Score",ascending=False)
        df["Rank"]=range(1,len(df)+1)

        st.success("✅ Processing Completed")
        st.dataframe(df,use_container_width=True)

        if st.button("🚀 Send Only Eligible to DB"):
            selected=df[df["Eligible"]==True].to_dict("records")

            requests.post(
                f"{BACKEND_URL}/save-final",
                data={"data":str(selected),"tenant":tenant}
            )

            st.success("🎯 Eligible Candidates Sent to DB")

# ================= INTERVIEWER =================
with tabs[1]:

    st.subheader("👨‍💼 Interview Dashboard")

    if not st.session_state.logged_in:

        tenant=st.selectbox("🏢 Company",list(passwords.keys()),key="login_company")
        user=st.selectbox("👤 Interviewer",list(passwords[tenant].keys()),key="login_user")
        pwd=st.text_input("🔑 Password",type="password")

        if st.button("Login"):
            if pwd==passwords[tenant][user]:
                st.session_state.logged_in=True
                st.session_state.user=user
                st.session_state.tenant=tenant
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Wrong password")

    else:
        st.success(f"Logged in as {st.session_state.user}")

        if st.button("Logout"):
            st.session_state.logged_in=False
            st.rerun()

        data=requests.get(
            f"{BACKEND_URL}/interviewers/candidates",
            params={
                "tenant":st.session_state.tenant,
                "interviewer":st.session_state.user
            }
        ).json()

        for c in data:
            with st.container():
                st.markdown(f"### 🧑 Candidate ID: `{c['tracking_id']}`")
                st.markdown("---")

                col1,col2,col3=st.columns(3)
                col1.metric("Test",c["test_status"])
                col2.metric("Interview",c["interview_status"])
                col3.metric("Final",c["final_status"])

                if c["test_status"]=="pending":
                    if st.button("🧪 Mark Test Done",key=c["tracking_id"]+"t"):
                        requests.put(f"{BACKEND_URL}/update-status",
                            data={"tracking_id":c["tracking_id"],"field":"test_status","value":"done"})
                        st.rerun()

                elif c["interview_started"]==0:
                    if st.button("▶ Start Interview",key=c["tracking_id"]+"start"):
                        requests.put(f"{BACKEND_URL}/update-status",
                            data={"tracking_id":c["tracking_id"],"field":"interview_started","value":"1"})
                        st.rerun()

                elif c["interview_status"]=="pending":
                    st.info("🎤 Interview in Progress")
                    if st.button("⏹ End Interview",key=c["tracking_id"]+"end"):
                        requests.put(f"{BACKEND_URL}/update-status",
                            data={"tracking_id":c["tracking_id"],"field":"interview_status","value":"done"})
                        st.rerun()

                else:
                    colA,colB=st.columns(2)
                    if colA.button("✅ Select",key=c["tracking_id"]+"s"):
                        requests.put(f"{BACKEND_URL}/update-status",
                            data={"tracking_id":c["tracking_id"],"field":"final_status","value":"selected"})
                        st.rerun()
                    if colB.button("❌ Reject",key=c["tracking_id"]+"r"):
                        requests.put(f"{BACKEND_URL}/update-status",
                            data={"tracking_id":c["tracking_id"],"field":"final_status","value":"rejected"})
                        st.rerun()

                st.markdown("------")

# ================= CANDIDATE =================
with tabs[2]:

    st.subheader("🧑 Candidate Tracking")

    tid=st.text_input("🔍 Enter Tracking ID")

    if st.button("Track Status"):
        res=requests.get(f"{BACKEND_URL}/status/{tid}")

        if res.status_code==200:
            d=res.json()

            progress = (
                1 if d["final_status"]!="pending" else
                0.66 if d["interview_status"]=="done" else
                0.33 if d["test_status"]=="done" else 0.1
            )

            st.progress(progress)

            st.markdown("### 📊 Status Overview")
            st.success(f"🧪 Test: {d['test_status']}")
            st.info(f"🎤 Interview: {d['interview_status']}")

            if d["final_status"]=="selected":
                st.success("🎉 FINAL RESULT: SELECTED")
            elif d["final_status"]=="rejected":
                st.error("❌ FINAL RESULT: REJECTED")
            else:
                st.warning("⏳ FINAL STATUS: IN PROCESS")

        else:
            st.error("Tracking ID not found")