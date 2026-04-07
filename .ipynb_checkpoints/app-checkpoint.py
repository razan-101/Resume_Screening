import io, re, hashlib, json, requests, streamlit as st, pdfplumber, docx, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BACKEND_URL = "https://resume-backend-i634.onrender.com"

st.set_page_config(page_title="Resume Screening", layout="wide")

st.title("🚀 Privacy-Aware Resume Screening System")

tabs = st.tabs(["📂 Recruiter", "👨‍💼 Interviewer", "🧑 Candidate"])

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

# ---------------- RECRUITER ----------------
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

        # -------- EXACT DUPLICATES --------
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

        # -------- FUZZY DUPLICATES --------
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

        # -------- PROCESS --------
        results = []

        st.subheader("⚙️ Processing Resumes")

        for i in keep:
            f = raw_files[i]
            uid = hashes[i][:8]

            clean = clean_text(texts[i])

            if len(clean) < 10:
                st.warning(f"⚠️ Weak resume skipped → {f.name}")
                continue

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
                "UID": uid,
                "Role": data.get("predicted_role"),
                "Score": data.get("ranking_score"),
                "Eligible": data.get("eligible"),
                "Recruiter": data.get("assigned_recruiter")
            })

        if results:
            df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
            df["Rank"] = df.index + 1
            df["Eligible"] = df["Eligible"].apply(lambda x: "✅ Yes" if x else "❌ No")

            df = df[["Rank","UID","Role","Score","Eligible","Recruiter"]]

            st.subheader("📊 Final Ranked Candidates")
            st.dataframe(df, use_container_width=True)

            st.download_button("⬇️ Download CSV", df.to_csv(index=False), "results.csv")

            # -------- FINAL BATCH --------
            st.subheader("📦 Finalize Batch")

            batch_name = st.text_input("Batch Name")

            if st.button("🚀 Send Final Eligible Candidates"):
                eligible = [r for r in results if r["Eligible"]]

                payload = []
                for r in eligible:
                    payload.append({
                        "tracking_id": "hidden",
                        "predicted_role": r["Role"],
                        "ranking_score": r["Score"],
                        "assigned_recruiter": r["Recruiter"]
                    })

                requests.post(
                    f"{BACKEND_URL}/final-batch",
                    data={"batch_name": batch_name, "tenant": tenant, "data": json.dumps(payload)}
                )

                st.success("✅ Batch Saved Successfully")

# ---------------- INTERVIEWER ----------------
with tabs[1]:
    st.header("👨‍💼 Interviewer Dashboard")

    res = requests.get(f"{BACKEND_URL}/interviewers/candidates")

    if res.status_code == 200:
        data = res.json()

        if not data:
            st.info("No candidates yet")
        else:
            st.dataframe(pd.DataFrame(data), use_container_width=True)
    else:
        st.error("Backend error")

# ---------------- CANDIDATE ----------------
with tabs[2]:
    st.header("🧑 Candidate Tracking")

    tid = st.text_input("Enter Tracking ID")

    if st.button("Check Status"):
        res = requests.get(f"{BACKEND_URL}/candidates/status/{tid}")

        if res.status_code == 200:
            st.json(res.json())
        else:
            st.error("Candidate not found")