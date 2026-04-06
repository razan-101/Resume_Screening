# Privacy-Aware Resume Screening Using NLP

This project implements an NLP-based resume screening system with a
privacy-aware edge–cloud architecture. Sensitive information is removed
at the edge layer before cloud-based classification.

## Features

- Resume upload (PDF/TXT)
- Privacy-aware preprocessing (email & phone anonymization)
- NLP-based resume classification
- Word cloud visualization
- Multi-tenant (company-wise) screening support

## Architecture

- Edge Layer: Resume upload + PII removal
- Cloud Layer: NLP vectorization & classification
- Multi-Tenant SaaS design

## How to Run

```bash
conda activate resume_nlp
streamlit run app.py

```
