import os
import tempfile
import traceback
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import re
import logging
import fitz  # PyMuPDF
from docx import Document
from agents_module import create_resume_calibrator_agent, create_skills_agent, create_experience_agent
from tasks_local import create_calibration_task, create_skill_evaluation_task, create_experience_evaluation_task, log_run
from utils import extract_skills_section, extract_experience_section, skills_keywords
from datetime import datetime
import json
from crewai import Crew
from langchain_groq import ChatGroq
import platform
import transformers
import spacy
from spacy.matcher import PhraseMatcher
from langchain_core.messages import HumanMessage
from fuzzywuzzy import fuzz

# Streamlit UI setup
st.set_page_config(page_title='📝 Resume Cupid', page_icon="📝")

# Load environment variables
load_dotenv(find_dotenv())

# Initialize the LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)

# Initialize logging
logging.basicConfig(level=logging.INFO, filename='resume_calibrator.log')

# Log environment details
logging.info(f"Python Version: {platform.python_version()}")
logging.info(f"Transformers Version: {transformers.__version__}")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Feedback data file
FEEDBACK_FILE = os.getenv("FEEDBACK_FILE")
if not FEEDBACK_FILE:
    st.error("FEEDBACK_FILE environment variable is not set.")
    st.stop()

# Load feedback data from file
@st.cache_data
def load_feedback_data():
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as file:
                content = file.read().strip()
                if content:
                    return json.loads(content)
        except json.JSONDecodeError:
            logging.error("Error decoding JSON from feedback data file.")
    return []

feedback_data = load_feedback_data()

# Save feedback data to file
def save_feedback_data(feedback_data):
    try:
        with open(FEEDBACK_FILE, "w") as file:
            json.dump(feedback_data, file, indent=4)
    except IOError:
        logging.error("Error saving feedback data to file.")

# Main app function
def main_app():
    st.title("📝 Resume Cupid")
    st.header("Calibrate Resumes with AI")

    # Input job description
    job_description = st.text_area("Job Description", height=150)

    # Upload resume file
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

    # Select role
    role = st.selectbox("Select Role", ["Software Development Engineer", "Data Scientist"])

    # Calibration parameters
    parameters = st.text_input("Calibration Parameters", value="{}")
    weights = st.text_input("Weight Parameters", value="{}")

    # Process the uploaded file
    if uploaded_file is not None:
        resume_text = ""
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(uploaded_file)

        # Display the resume text
        st.text_area("Resume Text", resume_text, height=250)

        if st.button("Calibrate Resume"):
            with st.spinner("Calibrating..."):
                input_data = {
                    "job_description": job_description,
                    "resume": resume_text,
                    "role": role,
                    "parameters": parameters,
                    "weights": weights,
                }
                output_data = calibrate_resume(input_data)
                st.success("Calibration Complete!")
                st.json(output_data)

                log_run(input_data, output_data)

# Login form
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "tester" and password == "Fractal123":
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid username or password")

# Calibration function (placeholder, replace with actual logic)
def calibrate_resume(input_data):
    # Placeholder for actual calibration logic
    output_data = {
        "crew_result": "Sample result based on input data"
    }
    return output_data

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Run the main app or login
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login()
    else:
        main_app()
