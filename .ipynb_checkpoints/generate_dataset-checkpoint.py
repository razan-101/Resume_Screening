import os
import random
from faker import Faker
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker()

# ---------------- CONFIG ----------------
OUTPUT_DIR = "test_resumes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

roles = [
    "Python Developer", "Java Developer", "Data Science",
    "DevOps Engineer", "HR", "Mechanical Engineer",
    "Sales", "Testing", "Cyber Security"
]

skills_map = {
    "Python Developer": ["Python", "Django", "Flask"],
    "Java Developer": ["Java", "Spring Boot"],
    "Data Science": ["Python", "Pandas", "ML"],
    "DevOps Engineer": ["Docker", "AWS", "Kubernetes"],
    "HR": ["Recruitment", "Communication"],
    "Mechanical Engineer": ["AutoCAD", "SolidWorks"],
    "Sales": ["Marketing", "Client Handling"],
    "Testing": ["Selenium", "Manual Testing"],
    "Cyber Security": ["Ethical Hacking", "Firewalls"]
}

# ---------------- RESUME TEXT ----------------
def generate_resume_text(name, email, phone, role):
    skills = ", ".join(skills_map[role])
    return f"""
{name}
Email: {email}
Phone: {phone}

Role: {role}
Skills: {skills}
Experience: {random.randint(1,5)} years experience
"""

# ---------------- TXT ----------------
def create_txt(filename, text):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

# ---------------- DOCX ----------------
def create_docx(filename, text):
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(filename)

# ---------------- PDF ----------------
def create_pdf(filename, text):
    c = canvas.Canvas(filename, pagesize=letter)
    y = 750
    for line in text.split("\n"):
        c.drawString(50, y, line)
        y -= 15
    c.save()

# ---------------- GENERATE ----------------
records = []

for i in range(50):
    name = fake.name()
    email = fake.email()
    phone = fake.phone_number()
    role = random.choice(roles)

    text = generate_resume_text(name, email, phone, role)

    # Save original record (for duplicates later)
    records.append((name, email, phone, role, text))

    fmt = random.choice(["txt", "docx", "pdf"])
    filename = os.path.join(OUTPUT_DIR, f"resume_{i}.{fmt}")

    if fmt == "txt":
        create_txt(filename, text)
    elif fmt == "docx":
        create_docx(filename, text)
    else:
        create_pdf(filename, text)

# ---------------- HARD DUPLICATES ----------------
for i in range(5):
    name, email, phone, role, text = random.choice(records)

    # Slight variation (fuzzy duplicate)
    text = text.replace(role, role + " Engineer")

    filename = os.path.join(OUTPUT_DIR, f"duplicate_{i}.txt")
    create_txt(filename, text)

print("✅ Dataset generated in 'test_resumes/' folder")