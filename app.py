import io
import os
import re
import json
import hashlib
import docx
import matplotlib.pyplot as plt
import nltk
import pdfplumber
import requests
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BACKEND_URL = "https://resume-backend-i634.onrender.com"

def ensure_nltk_resource(resource_path, resource_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_name, quiet=True)

# ---------------- PASSWORDS ----------------
passwords = {
    "Tech Solutions Pvt Ltd": {"Aravind – Backend Team": "aravind1234", "Ishaan – Frontend Team": "ishaan1234", "Advait – DevOps Team": "advait1234"},
    "Data & Analytics Corp": {"Vihaan – Data Engineering": "vihaan1234", "Ananya – Analytics": "ananya1234", "Pranav – Program Management": "pranav1234"},
    "Quality & Security Systems Ltd": {"Kritika – QA": "kritika1234", "Rishi – Automation": "rishi1234", "Siddharth – Security": "siddharth1234"},
    "Enterprise & Sales Solutions": {"Varun – Sales": "varun1234", "Tanvi – HR": "tanvi1234", "Aditya – Legal": "aditya1234"},
    "Engineering & Manufacturing Group": {"Kartik – Mechanical": "kartik1234", "Arnav – Electrical": "arnav1234", "Saanvi – Civil": "saanvi1234"},
    "Creative & Wellness Services": {"Kiara – Creative": "kiara1234", "Zoya – Wellness": "zoya1234", "Vivaan – Lifestyle": "vivaan1234"}
}

def remove_pii(text):
    # Mask Emails
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", " ", text)
    # Mask Phone Numbers
    text = re.sub(r"(\+?\d[\d -]{8,12}\d)", " ", text)
    # Mask LinkedIn/URLs
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"linkedin\.com/\S+", " ", text)
    
    # Mask common location keywords 
    locations = ["odisha", "india", "bhubaneswar", "bangalore", "hyderabad", "pune", "mumbai", "delhi"]
    for loc in locations:
        text = re.sub(rf"\b{loc}\b", " ", text, flags=re.IGNORECASE)
        
    # Mask common resume header labels that clutter word clouds
    labels = ["name", "phone", "email", "linkedin", "location", "address", "mobile", "contact"]
    for label in labels:
        text = re.sub(rf"\b{label}\b", " ", text, flags=re.IGNORECASE)

    # Name removal heuristic: 
    lines = text.split('\n')
    if lines and len(lines[0].split()) < 5: # Heuristic for a name header
        lines[0] = " "
    
    return "\n".join(lines)

def clean_resume(text):
    text = remove_pii(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_bytes):
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)

def extract_resume_text(file_name, file_bytes):
    extension = file_name.split(".")[-1].lower()
    if extension == "txt":
        return file_bytes.decode("utf-8", errors="ignore")
    if extension == "pdf":
        return extract_text_from_pdf(file_bytes)
    if extension == "docx":
        return extract_text_from_docx(file_bytes)
    return ""

def get_backend_data(path, params=None):
    response = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()

def post_resume(tenant, uploaded_file, tracking_id=None):
    file_bytes = uploaded_file.getvalue()
    files = {"resume": (uploaded_file.name, file_bytes, uploaded_file.type or "application/octet-stream")}
    # Send dummy data as backend requires these fields but we don't want to use filename-based names
    data = {"full_name": "Candidate", "email": "candidate@privacy.com", "tenant": tenant}
    if tracking_id:
        data["tracking_id"] = tracking_id
    response = requests.post(f"{BACKEND_URL}/candidates/upload-resume", data=data, files=files, timeout=60)
    response.raise_for_status()
    return response.json()

def extract_structured_info(text):
    if not text:
        return {}
    
    text = text.lower()
    sections = {
        "Professional Summary": ["professional summary", "about me", "profile", "objective"],
        "Skills": ["skills", "technical skills", "coding languages", "general programmings", "database managements", "frameworks and libraries", "tools"],
        "Projects": ["projects", "personal projects", "academic projects"],
        "Internships": ["internship", "internships", "professional experience", "work experience"],
        "Education": ["education", "academic background", "qualification"],
        "Certifications": ["certifications", "achievements", "certifications completed"]
    }
    
    extracted = {k: "" for k in sections}
    
    found_headers = []
    for section_name, headers in sections.items():
        for h in headers:
            pattern = rf"\b{re.escape(h)}\b"
            for match in re.finditer(pattern, text):
                found_headers.append({
                    "start": match.start(),
                    "end": match.end(),
                    "section": section_name,
                    "header": h
                })
    
    if not found_headers:
        return {"Candidate Profile": text[:600] + "..." if len(text) > 600 else text}
        
    found_headers.sort(key=lambda x: x["start"])
    
    # Check if there is text before the first header
    if found_headers[0]["start"] > 20:
        extracted["Intro"] = text[:found_headers[0]["start"]].strip()
    
    for i in range(len(found_headers)):
        current = found_headers[i]
        start_pos = current["end"]
        end_pos = found_headers[i+1]["start"] if i + 1 < len(found_headers) else len(text)
        
        content = text[start_pos:end_pos].strip()
        if content:
            if current["section"] in extracted and extracted[current["section"]]:
                extracted[current["section"]] += " | " + content
            else:
                extracted[current["section"]] = content
                
    # Clean up and prioritize
    final_info = {}
    if "Intro" in extracted and extracted["Intro"]:
        final_info["Candidate Profile"] = extracted["Intro"]
        
    for k, v in extracted.items():
        if k != "Intro" and v:
            final_info[k] = v
            
    return final_info

def render_wordcloud(cleaned_text):
    if not cleaned_text or not cleaned_text.strip():
        st.warning("No professional text found to generate a word cloud.")
        return
    try:
        # Custom stopwords to exclude section headers from wordcloud
        custom_stopwords = set(stopwords.words("english"))
        header_terms = ["skills", "projects", "internship", "internships", "experience", "education", "achievements", "certifications", "professional", "summary", "objective", "about", "profile"]
        custom_stopwords.update(header_terms)
        
        wordcloud = WordCloud(width=900, height=400, background_color="white", stopwords=custom_stopwords).generate(cleaned_text)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not generate word cloud: {e}")

def extract_pii_for_dedup(text):
    # Extract Email
    emails = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    email = emails[0].lower().strip() if emails else "unknown@example.com"
    
    # Extract Phone
    phones = re.findall(r"(\+?\d[\d -]{8,12}\d)", text)
    phone = re.sub(r"[^\d]", "", phones[0]) if phones else "0000000000"
    
    # Extract Name (Heuristic: first non-empty line)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    name = lines[0].lower().strip() if lines else "unknown"
    
    return name, email, phone

def generate_fingerprint(name, email, phone):
    return hashlib.sha256(f"{name}-{email}-{phone}".encode()).hexdigest()

def render_resume_page(tenants):
    st.header("Batch Resume Processing")
    
    tenant = st.selectbox("Select Company", tenants)
    files = st.file_uploader("Upload Resumes (PDF/DOCX/TXT)", accept_multiple_files=True, type=["pdf", "docx", "txt"])
    
    if st.button("Process Batch"):
        if not files:
            st.warning("Please upload resumes first.")
            return
            
        results = []
        seen_fingerprints = set()
        processed_texts = [] # For fuzzy deduplication
        
        progress_bar = st.progress(0)
        
        with st.spinner("Analyzing resumes and identifying unique candidates..."):
            for idx, f in enumerate(files):
                file_bytes = f.getvalue()
                text = extract_resume_text(f.name, file_bytes)
                
                if not text.strip():
                    st.error(f"Empty file: {f.name}")
                    continue
                
                # 1. Identity-based Deduplication (Name, Email, Phone)
                name, email, phone = extract_pii_for_dedup(text)
                fingerprint = generate_fingerprint(name, email, phone)
                
                if fingerprint in seen_fingerprints:
                    st.warning(f"Skipping duplicate candidate (same identity in current batch): {f.name}")
                    continue
                
                # Check if candidate already exists in DB
                temp_tid = fingerprint[:8]
                try:
                    check_res = requests.get(f"{BACKEND_URL}/candidates/status/{temp_tid}")
                    if check_res.status_code == 200:
                        st.warning(f"Skipping duplicate candidate (already exists in database): {f.name}")
                        continue
                except:
                    pass
                
                # 2. Content-based Deduplication (Fuzzy Similarity)
                clean_text_content = clean_resume(remove_pii(text))
                
                is_fuzzy_duplicate = False
                if processed_texts:
                    # Compare with already processed resumes in this batch
                    tfidf_dedup = TfidfVectorizer().fit_transform([clean_text_content] + processed_texts)
                    sim_scores = cosine_similarity(tfidf_dedup[0:1], tfidf_dedup[1:])[0]
                    if any(score > 0.95 for score in sim_scores):
                        is_fuzzy_duplicate = True
                
                if is_fuzzy_duplicate:
                    st.warning(f"Skipping duplicate candidate (highly similar content): {f.name}")
                    continue

                # Add to unique set
                seen_fingerprints.add(fingerprint)
                processed_texts.append(clean_text_content)
                
                # Process unique candidate
                try:
                    res = post_resume(tenant, f, tracking_id=temp_tid)
                    results.append(res)
                except Exception as e:
                    st.error(f"Error processing {f.name}: {e}")
                
                progress_bar.progress((idx + 1) / len(files))
        
        if results:
            st.session_state.processing_results = results
            st.session_state.selected_tenant = tenant
            st.success(f"Processed {len(results)} unique candidates from {len(files)} files.")

    if "processing_results" in st.session_state and st.session_state.processing_results:
        st.subheader("Candidate Ranking & Assignment")
        
        df = pd.DataFrame(st.session_state.processing_results)
        df = df.sort_values("ranking_score", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1

        # Re-distribute recruiters evenly among the ranked candidates for fairness
        tenant_recruiters = list(passwords.get(st.session_state.selected_tenant, {}).keys())
        if tenant_recruiters:
            for i in range(len(df)):
                df.at[i, "assigned_recruiter"] = tenant_recruiters[i % len(tenant_recruiters)]
        
        # Update the session state results to match the fair assignment for final saving
        st.session_state.processing_results = df.to_dict('records')
        
        # Display the table with assigned interviewer/specialty
        df_display = df.copy()
        df_display["eligible"] = df_display["eligible"].apply(lambda x: "Yes" if x else "No")
        
        # Split recruiter name to show "Specialty" if needed, though here they are already named with roles
        # e.g., "Arjun – Backend Team"
        st.dataframe(df_display[["Rank", "tracking_id", "predicted_role", "ranking_score", "eligible", "assigned_recruiter"]], use_container_width=True)

        if st.button("Send Final Eligible Candidates to DB"):
            eligible_rows = [r for r in st.session_state.processing_results if r.get("eligible")]
            if not eligible_rows:
                st.warning("No eligible candidates to save.")
            else:
                try:
                    # st.write(f"Sending data to {BACKEND_URL}/final-batch...")
                    # st.write(eligible_rows)
                    res = requests.post(f"{BACKEND_URL}/final-batch", data={
                        "tenant": st.session_state.selected_tenant,
                        "data": json.dumps(eligible_rows)
                    })
                    if res.status_code == 200:
                        st.success("Candidates saved and assigned to interviewers successfully!")
                        del st.session_state.processing_results
                    else:
                        st.error(f"Backend error: {res.status_code} - {res.text}")
                except Exception as e:
                    st.error(f"Error saving batch: {e}")

def render_interviewer_page():
    st.header("Interviewer Dashboard")
    
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        tenant_list = list(passwords.keys())
        sel_tenant = st.selectbox("Company", tenant_list)
        sel_user = st.selectbox("Interviewer", list(passwords[sel_tenant].keys()))
        pwd = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if pwd == passwords[sel_tenant][sel_user]:
                st.session_state.logged_in = True
                st.session_state.user = sel_user
                st.session_state.tenant = sel_tenant
                st.rerun()
            else:
                st.error("Wrong password.")
    else:
        st.success(f"Welcome, {st.session_state.user} ({st.session_state.tenant})")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

        try:
            candidates = get_backend_data("/interviewers/candidates", params={
                "tenant": st.session_state.tenant,
                "interviewer": st.session_state.user
            })
            if candidates:
                df = pd.DataFrame(candidates)
                st.dataframe(df[["tracking_id", "predicted_role", "test_status", "interview_status", "final_status"]], use_container_width=True)
                
                st.divider()
                sel_tid = st.selectbox("Select Candidate to Evaluate", [c['tracking_id'] for c in candidates])
                c_details = get_backend_data(f"/candidates/details/{sel_tid}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📝 Process Updates")
                    test_s = st.selectbox("Test Status", ["pending", "attended", "passed", "failed"], index=["pending", "attended", "passed", "failed"].index(c_details.get("test_status", "pending")))
                    if st.button("Update Test"):
                        requests.put(f"{BACKEND_URL}/interview/update-status", data={"tracking_id": sel_tid, "test_status": test_s})
                        st.rerun()
                    
                    if test_s == "passed":
                        i_time = st.text_input("Interview Date/Time", value=c_details.get("interview_time") or "")
                        if st.button("Schedule Interview"):
                            requests.put(f"{BACKEND_URL}/interview/update-status", data={"tracking_id": sel_tid, "interview_status": "scheduled", "interview_time": i_time})
                            st.rerun()

                    if c_details.get("interview_status") == "scheduled":
                        if st.button("Complete Interview"):
                            requests.put(f"{BACKEND_URL}/interview/update-status", data={"tracking_id": sel_tid, "interview_status": "completed"})
                            st.rerun()

                    if c_details.get("interview_status") == "completed":
                        final_res = st.radio("Final Decision", ["selected", "rejected"])
                        if st.button("Finalize"):
                            requests.put(f"{BACKEND_URL}/interview/update-status", data={"tracking_id": sel_tid, "final_status": final_res})
                            st.rerun()
                
                with col2:
                    st.subheader("Word Cloud Insights")
                    render_wordcloud(c_details.get("cleaned_text", ""))
                    
                    st.divider()
                    st.subheader("Structured Insights")
                    structured_info = extract_structured_info(c_details.get("cleaned_text", ""))
                    if structured_info:
                        for section, content in structured_info.items():
                            with st.expander(f" {section}", expanded=(section in ["Skills", "Projects"])):
                                st.write(content.capitalize())
                    else:
                        st.info("No structured information could be extracted.")
            else:
                st.info("No candidates assigned.")
        except Exception as e:
            st.error(f"Error: {e}")

def render_candidate_page():
    st.header("Candidate Tracking")
    tracking_id = st.text_input("Enter Tracking ID")
    
    if st.button("Track Status"):
        if tracking_id:
            try:
                res = get_backend_data(f"/candidates/status/{tracking_id}")
                st.subheader(f"Status for Candidate {tracking_id}")
                
                # Progress Bar Logic
                prog = 0
                if res['test_status'] == "passed": prog += 33
                if res['interview_status'] == "completed": prog += 33
                if res['final_status'] in ["selected", "rejected"]: prog += 34
                
                st.progress(prog / 100)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Test", res['test_status'].capitalize())
                col2.metric("Interview", res['interview_status'].capitalize())
                col3.metric("Final", res['final_status'].capitalize())
                
                if res['interview_time']:
                    st.info(f" Interview Scheduled: {res['interview_time']}")
                
                if res['final_status'] == "selected": st.success(" Congratulations! You are selected.")
                elif res['final_status'] == "rejected": st.error(" Sorry, you were not selected.")
                
            except Exception:
                st.error("Candidate not found.")

def main():
    st.set_page_config(page_title="Resume Screening System", layout="wide")
    ensure_nltk_resource("corpora/stopwords", "stopwords")
    ensure_nltk_resource("tokenizers/punkt", "punkt")

    st.title(" Privacy-Aware AI Resume Screening")
    
    try:
        tenants_payload = get_backend_data("/tenants")
        tenants = list(tenants_payload.get("tenants", {}).keys())
    except Exception:
        st.error(f"Backend not reachable at {BACKEND_URL}.")
        st.stop()

    tabs = st.tabs([" Recruiter", " Interviewer", " Candidate"])
    with tabs[0]: render_resume_page(tenants)
    with tabs[1]: render_interviewer_page()
    with tabs[2]: render_candidate_page()

if __name__ == "__main__":
    main()
