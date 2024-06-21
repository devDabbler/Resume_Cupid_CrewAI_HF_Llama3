import os
import tempfile
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import re
import logging
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text
from agents_module import create_resume_calibrator_agent, create_skills_agent, create_experience_agent
from tasks import create_calibration_task, create_skill_evaluation_task, create_experience_evaluation_task, log_run
from utils import read_all_pdf_pages, extract_skills_section, extract_experience_section, skills_keywords
from datetime import datetime
import json
import time
import tensorflow as tf
from crewai import Crew
from crew import Crew
from langchain_groq import ChatGroq

# Streamlit UI setup
st.set_page_config(page_title='📝 Resume Cupid', page_icon="📝")

# Load environment variables
load_dotenv(find_dotenv())

# Initialize the LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)

# Load the BERT model and tokenizer from Hugging Face
@st.cache_resource
def load_model_and_tokenizer():
    model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

try:
    model, tokenizer = load_model_and_tokenizer()
except EnvironmentError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize logging
logging.basicConfig(filename='resume_calibrator.log', level=logging.ERROR)

# Feedback data file
FEEDBACK_FILE = "feedback_data.json"

# Load feedback data from file
@st.cache_data
def load_feedback_data():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as file:
            content = file.read().strip()
            if content:
                return json.loads(content)
    return []

feedback_data = load_feedback_data()

# Save feedback data to file
def save_feedback_data(feedback_data):
    with open(FEEDBACK_FILE, "w") as file:
        json.dump(feedback_data, file)

# Function to extract first name from resume text
def extract_first_name(resume_text):
    match = re.match(r"(\w+)", resume_text)
    if match:
        return match.group(1)
    return "Unknown"

# Streamlit UI setup
st.title("Resume Cupid")
st.markdown("Use this app to help you decide if a candidate is a good fit for a specific role.")

with st.form(key='resume_form'):
    job_description = st.text_area("Paste the Job Description here. Make sure to include key aspects of the role required.", placeholder="Job description. This field should have at least 100 characters.")
    resume_file = st.file_uploader("Upload your resume", type=['pdf', 'docx'])
    role = st.text_input("Type the role for which the candidate is being evaluated:", placeholder="Enter the role here")
    
    st.write("Enter the key parameters to evaluate the resume:")
    skill1 = st.text_input("Skill 1", placeholder="JavaScript")
    skill2 = st.text_input("Skill 2", placeholder="Python")
    skill3 = st.text_input("Skill 3", placeholder="React")
    skill4 = st.text_input("Skill 4", placeholder="ETL")
    skill5 = st.text_input("Skill 5", placeholder="Git")
    min_experience = st.number_input("Minimum years of experience", min_value=0, value=5)
    
    st.write("Rank the skills in order of importance (1 being the most important):")
    skill_rankings = []
    for i in range(1, 6):
        rank = st.number_input(f"Rank for Skill {i}", min_value=1, max_value=5, value=i, key=f"skill_rank_{i}")
        skill_rankings.append(rank)
    
    submitted = st.form_submit_button('Submit')

def calculate_weights(rankings):
    total_ranks = sum(rankings)
    weights = [rank / total_ranks for rank in rankings]
    return weights

def calculate_fitment_score(individual_scores, weights):
    total_score = sum(score * weight for score, weight in zip(individual_scores, weights))
    max_score = sum(weights)
    fitment_percentage = total_score * 100 / max_score
    return fitment_percentage

def display_results(result, position_titles):
    result_without_duplicate = re.sub(r'\*\*Experience Evaluation Report:\*\*', '', result)
    st.write(result_without_duplicate)
    
    st.subheader("Candidate's Position Titles:")
    for title in position_titles:
        st.write(f"- {title}")
    
    experience_report = re.search(r"\*\*Experience Evaluation Report\*\*(.*?)\n\n", result, re.DOTALL)
    if experience_report:
        st.subheader("Experience Evaluation Report:")
        st.write(experience_report.group(1).strip())
    
    skills_report = re.search(r"\*\*Technical Skills Evaluation Report\*\*(.*?)\*\*Recommendation\*\*", result, re.DOTALL)
    if skills_report:
        st.subheader("Technical Skills Evaluation Report:")
        st.write(skills_report.group().strip())

    fitment_score = re.search(r"\*\*Fitment Score: (\d+\.\d+)%", result)
    if fitment_score:
        fitment_percentage = float(fitment_score.group(1))
        st.subheader("Fitment Score:")
        st.write(f"{fitment_percentage:.2f}%")
        
        if fitment_percentage < 60:
            st.warning("The candidate's experience fitment is relatively low. Please review the detailed evaluation report for areas where the candidate's experience may not align with the job requirements.")
        elif fitment_percentage < 80:
            st.info("The candidate's experience fitment is moderate. Please review the detailed evaluation report to identify areas where the candidate's experience can be strengthened to better align with the job requirements.")
        else:
            st.success("The candidate's experience fitment is strong. Please review the detailed evaluation report to understand the candidate's relevant experience and skills.")

@st.cache_data
def analyze_skills(skills_section, job_description):
    required_skills = job_description.lower().split(',')
    candidate_skills = skills_section.lower().split(',')
    matched_skills = [skill for skill in required_skills if skill in candidate_skills]
    return matched_skills

@st.cache_data
def analyze_experience(experience_section, job_description):
    relevant_experience = []
    position_titles = []

    section_pattern = re.compile(r'^(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)$', re.MULTILINE)
    bullet_point_pattern = re.compile(r'^\s*•\s*(.+?)$', re.MULTILINE)

    sections = section_pattern.findall(experience_section)

    for title, company, duration in sections:
        position_titles.append(f"{title.strip()} at {company.strip()} ({duration.strip()})")

        section_start = experience_section.find(title)
        section_end = experience_section.find('\n\n', section_start)
        if section_end == -1:
            section_end = len(experience_section)

        section_text = experience_section[section_start:section_end]
        bullet_points = bullet_point_pattern.findall(section_text)

        if any(keyword in ' '.join(bullet_points).lower() for keyword in ['data scientist', 'machine learning', 'data analysis', 'python', 'sql', 'regression', 'unsupervised learning', 'time-series']):
            relevant_experience.append(section_text.strip())

    return relevant_experience, position_titles

@st.cache_data
def read_all_pdf_pages(pdf_path):
    text = ''
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        if text.strip():
            return text
    except Exception as e:
        print(f"PyMuPDF extraction failed: {e}")

    try:
        text = pdfminer_extract_text(pdf_path)
        if text.strip():
            return text
    except Exception as e:
        print(f"PDFMiner extraction failed: {e}")

    return ""

@st.cache_data
def extract_resume_sections(resume):
    resume_skills = extract_skills_section(resume, skills_keywords)
    resume_experience = extract_experience_section(resume)
    return resume_skills, resume_experience

def predict_fitment(job_description, resume_text):
    inputs = tokenizer(job_description + " " + resume_text, return_tensors="tf", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy().item()
    return predicted_class

resume_first_name = "Unknown"

if submitted and resume_file is not None and len(job_description) > 100:
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(resume_file.read())
            resume_file_path = tmp_file.name
        
        resume = read_all_pdf_pages(resume_file_path)
        os.unlink(resume_file_path)
        
        resume_first_name = extract_first_name(resume)

        resume_skills, resume_experience = extract_resume_sections(resume)

        matched_skills = analyze_skills(resume_skills, job_description)
        relevant_experience, position_titles = analyze_experience(resume_experience, job_description)

        resume_calibrator = create_resume_calibrator_agent(llm)
        skills_agent = create_skills_agent(llm)
        experience_agent = create_experience_agent(llm)

        parameters = [skill1, skill2, skill3, skill4, skill5, f"{min_experience} or more years of experience"]
        weights = calculate_weights(skill_rankings)
        
        calibration_task = create_calibration_task(job_description, resume, resume_calibrator, role, parameters)
        
        def retry_request(request_func):
            max_retries = 3
            retry_delay = 1

            for i in range(max_retries):
                try:
                    return request_func()
                except Exception as e:
                    logging.warning(f"Request failed. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2

            raise Exception("Request failed after maximum retries.")

        skill_evaluation_task = create_skill_evaluation_task(job_description, resume_skills, skills_agent, role, parameters, weights)
        experience_evaluation_task = create_experience_evaluation_task(job_description, resume_experience, experience_agent, role, position_titles)

        crew = Crew([calibration_task, skill_evaluation_task, experience_evaluation_task])
        result = retry_request(lambda: crew.kickoff())

        input_data = {
            "job_description": job_description,
            "resume": resume,
            "role": role,
            "parameters": parameters,
            "weights": weights
        }
        output_data = {"result": result}
        log_run(input_data, output_data)

        display_results(result, position_titles)
        
    except Exception as e:
        st.error(f"Failed to process the request: {str(e)}")
        logging.error(f"Failed to process the request: {str(e)}")
        logging.exception(e)
else:
    st.write("Awaiting input and file upload...")

with st.form(key='feedback_form'):
    st.subheader("Feedback")
    name = st.text_input("Name of Person Leaving Feedback")
    resume_first_name = st.text_input("Candidate or Resume Name", value=resume_first_name)
    role_input = st.text_input("Role", value=role, disabled=True)
    client = st.text_input("Client")
    accuracy_rating = st.select_slider("Accuracy of the evaluation:", options=[1, 2, 3, 4, 5])
    content_rating = st.select_slider("Quality of the report content:", options=[1, 2, 3, 4, 5])
    suggestions = st.text_area("Please provide any suggestions for improvement:")
    submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    submit_feedback = st.form_submit_button("Submit Feedback")

    if submit_feedback:
        feedback_entry = {
            "resume_id": resume_first_name,
            "job_role_id": role_input,
            "name": name,
            "accuracy_rating": accuracy_rating,
            "content_rating": content_rating,
            "suggestions": suggestions,
            "submitted_at": submitted_at,
            "client": client
        }
        feedback_data.append(feedback_entry)
        save_feedback_data(feedback_data)
        st.success("Thank you for your feedback!")
