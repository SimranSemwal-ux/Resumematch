"""
Sample Resume Generator for ResumeMatch
Generates fictional sample resumes in TXT, DOCX, and PDF formats.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
import fitz  # PyMuPDF can also create PDFs directly

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUME_DIR = os.path.join(BASE_DIR, "resumes")
os.makedirs(RESUME_DIR, exist_ok=True)

# Resume 1: Full-Stack Engineer (Alex Morgan)
RESUME_ALEX = """Alex Morgan
San Francisco, CA | alex.morgan.dev@email.com | github.com/alexmorgan-dev

PROFESSIONAL SUMMARY:
Results-driven Full-Stack Software Engineer with 4 years of experience building scalable cloud applications, REST APIs, and responsive web user interfaces. Proficient in Python, JavaScript, TypeScript, React, and PostgreSQL. Experienced with containerized deployments using Docker, Kubernetes, and AWS.

TECHNICAL SKILLS:
- Programming Languages: Python, JavaScript, TypeScript, C++, SQL, HTML, CSS
- Frameworks & Libraries: Django, FastAPI, React, Node.js, Express, Next.js
- Databases: PostgreSQL, MySQL, Redis, MongoDB
- Cloud & DevOps: AWS, Docker, Kubernetes, CI/CD, Git, GitHub, Linux
- Core CS: Data Structures & Algorithms (DSA), OOP, DBMS, REST API, System Design

WORK EXPERIENCE:
Senior Software Developer | CloudScale Solutions (2022 - Present)
- Designed and maintained 15+ microservices and REST APIs using Python, FastAPI, and Django.
- Developed modular frontend dashboard components using React and TypeScript, improving page load speed by 35%.
- Optimized database indexing and query execution plans in PostgreSQL, reducing query latency by 40%.
- Orchestrated container builds with Docker and automated staging deployments to AWS EKS using Kubernetes and CI/CD pipelines.

Software Engineer | NexaTech Systems (2020 - 2022)
- Built internal tooling and data endpoints using Python, Flask, and MySQL.
- Implemented responsive web UI views with JavaScript, HTML, and CSS.
- Maintained code repositories using Git and conducted peer reviews on GitHub.

PROJECTS:
- Distributed Task Queue: Developed asynchronous background worker system in Python using Redis and Docker.
- E-Commerce Microservices Platform: Architected full-stack shopping portal with React frontend and FastAPI REST API backend deployed on AWS.

EDUCATION:
Bachelor of Science in Computer Science | State University (Graduated 2020)
- Coursework: Algorithms, Data Structures, Operating Systems, Computer Networks, Database Management Systems (DBMS).
"""

# Resume 2: Data Scientist (Priya Sharma)
RESUME_PRIYA = """Priya Sharma
New York, NY | priya.sharma.ml@email.com | github.com/priyasharma-data

PROFESSIONAL SUMMARY:
Analytical Data Scientist and Machine Learning Engineer with 3+ years of expertise designing predictive models, deep learning architectures, and statistical analytics pipelines. Proficient in Python, PyTorch, TensorFlow, Pandas, NumPy, and SQL.

TECHNICAL SKILLS:
- Programming Languages: Python, SQL, R, Bash
- AI & Data Science: Machine Learning, Deep Learning, Natural Language Processing (NLP), Computer Vision, Data Science, Data Analysis
- Libraries & Frameworks: PyTorch, TensorFlow, Scikit-learn, Pandas, NumPy, OpenCV, Keras
- Databases & Tools: PostgreSQL, MongoDB, Power BI, Tableau, Excel
- Cloud & DevOps: Docker, AWS, Git, Linux

WORK EXPERIENCE:
Machine Learning Engineer | DataCore Insights (2022 - Present)
- Developed and trained deep learning NLP classifiers in PyTorch, achieving 94% F1-score on customer sentiment benchmarks.
- Built automated data analysis pipelines utilizing Pandas and NumPy to process 5M+ daily transactions.
- Designed enterprise business intelligence dashboards in Tableau and Power BI for executive leadership.
- Containerized model training workflows with Docker and deployed inference microservices on AWS EC2.

Data Analyst | Metrics Analytics Group (2021 - 2022)
- Wrote complex SQL queries across PostgreSQL databases to extract behavioral metrics.
- Constructed regression and clustering models using Scikit-learn for customer segmentation.
- Generated comprehensive exploratory data reports in Excel and Python.

PROJECTS:
- Medical Imaging Segmentation: Implemented convolutional neural network in TensorFlow and OpenCV for radiographic anomaly detection.
- Financial Forecasting Pipeline: Developed end-to-end time series predictor using Python, Pandas, and Scikit-learn.

EDUCATION:
Master of Science in Data Science | Global Tech Institute (2021)
Bachelor of Technology in Information Technology | National University (2019)
"""

# Resume 3: Frontend Developer (Jordan Lee)
RESUME_JORDAN = """Jordan Lee
Seattle, WA | jordan.lee.web@email.com | github.com/jordanlee-ui

PROFESSIONAL SUMMARY:
Creative and performance-oriented Frontend Engineer with 3 years of experience crafting accessible, responsive web interfaces using React, Next.js, JavaScript, TypeScript, HTML, and CSS.

TECHNICAL SKILLS:
- Programming Languages: JavaScript, TypeScript, HTML, CSS, Python
- Frameworks & Libraries: React, Next.js, Vue, Node.js, Express
- Web & Tools: REST API, Git, GitHub, Linux, Figma, Vite, Jest
- Core Competencies: UI/UX Design, Responsive Design, State Management, Web Performance

WORK EXPERIENCE:
Frontend Web Developer | PixelCraft Studio (2022 - Present)
- Engineered responsive client interfaces using React, Next.js, and TypeScript.
- Translated UI/UX designs into pixel-perfect components with modern CSS and HTML.
- Integrated REST API endpoints with state management for seamless asynchronous data rendering.
- Streamlined GitHub pull request workflows and automated linting via CI/CD.

Junior Web Developer | Apex Digital (2021 - 2022)
- Built interactive web pages using JavaScript, HTML, and CSS.
- Maintained legacy single page apps built with Vue and Node.js.
- Utilized Git for feature branching and release tracking.

PROJECTS:
- SaaS Analytics Dashboard: Built full-featured client portal using React and TypeScript.
- Design System Component Library: Developed accessible, reusable UI component library published as an npm package.

EDUCATION:
Bachelor of Science in Software Engineering | Metro University (2021)
"""

SAMPLES = [
    ("alex_morgan_software_engineer", RESUME_ALEX, "Alex Morgan - Full-Stack Software Engineer"),
    ("priya_sharma_data_scientist", RESUME_PRIYA, "Priya Sharma - Machine Learning & Data Scientist"),
    ("jordan_lee_frontend_developer", RESUME_JORDAN, "Jordan Lee - Frontend React Engineer"),
]


def create_txt(filename: str, text: str):
    path = os.path.join(RESUME_DIR, f"{filename}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip())
    print(f"Created {path}")


def create_docx(filename: str, text: str, title: str):
    path = os.path.join(RESUME_DIR, f"{filename}.docx")
    doc = Document()
    
    # Title
    heading = doc.add_heading(title, level=0)
    heading.style.font.color.rgb = RGBColor(30, 41, 59)
    
    for line in text.split("\n"):
        line_s = line.strip()
        if not line_s:
            continue
        if line_s.isupper() and len(line_s) < 30:
            h = doc.add_heading(line_s, level=1)
            h.style.font.color.rgb = RGBColor(79, 70, 229)
        elif line_s.startswith("- "):
            doc.add_paragraph(line_s[2:], style="List Bullet")
        else:
            doc.add_paragraph(line_s)
            
    doc.save(path)
    print(f"Created {path}")


def create_pdf(filename: str, text: str, title: str):
    path = os.path.join(RESUME_DIR, f"{filename}.pdf")
    # Generate clean PDF using PyMuPDF fitz
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Standard US Letter
    
    rect = fitz.Rect(50, 50, 562, 742)
    # Write text cleanly into rect
    page.insert_textbox(rect, text, fontsize=9.5, fontname="helv", color=(0.1, 0.1, 0.15))
    doc.save(path)
    doc.close()
    print(f"Created {path}")


def main():
    for name, content, title in SAMPLES:
        create_txt(name, content)
        create_docx(name, content, title)
        create_pdf(name, content, title)


if __name__ == "__main__":
    main()
