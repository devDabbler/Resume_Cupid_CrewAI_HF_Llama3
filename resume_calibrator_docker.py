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

# Function to extract first name from resume text
def extract_first_name(resume_text):
    match = re.match(r"(\w+)", resume_text)
    if match:
        return match.group(1)
    return "Unknown"

# Function to parse resume
def parse_resume(file):
    if file.type == "application/pdf":
        return parse_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return parse_docx(file)
    else:
        raise ValueError("Unsupported file format")

def parse_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def parse_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Fetch login credentials from environment variables
LOGIN_USERNAME = os.getenv('LOGIN_USERNAME')
LOGIN_PASSWORD = os.getenv('LOGIN_PASSWORD')

# Login form
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid username or password")

def calculate_weights(rankings):
    total = sum(rankings)
    return [rank / total for rank in rankings]

def process_crew_result(result):
    if isinstance(result, list):
        return "\n\n".join(str(item) for item in result)
    elif isinstance(result, dict):
        return "\n\n".join(f"**{key}:** {value}" for key, value in result.items())
    else:
        return str(result)

def get_recommendation(fitment_score):
    if fitment_score >= 85:
        return "This candidate is an exceptional fit and should be strongly considered for an interview."
    elif fitment_score >= 75:
        return "This candidate is a strong fit and should be considered for an interview."
    elif fitment_score >= 65:
        return "This candidate shows promise but may need additional evaluation in some areas."
    elif fitment_score >= 55:
        return "This candidate meets some requirements but has significant gaps. Consider for a more junior role or different position."
    elif fitment_score >= 45:
        return "This candidate has relevant experience but may not be suitable for this specific role. Consider for other positions within the organization."
    else:
        return "This candidate does not meet most of the key requirements. Not recommended for this position."

def display_crew_results(crew_result):
    lines = crew_result.split('\n')
    
    # Extract fitment score
    fitment_score_line = next((line for line in lines if "Experience Fitment Score:" in line), None)
    if fitment_score_line:
        # Extract the score, removing any non-numeric characters
        score_text = ''.join(char for char in fitment_score_line.split(':')[1] if char.isdigit() or char == '.')
        try:
            fitment_score = float(score_text)
            st.subheader(f"Experience Fitment Score: {fitment_score:.1f}%")
            
            # Get and display recommendation
            recommendation = get_recommendation(fitment_score)
            st.subheader("Interview Recommendation")
            st.write(recommendation)
        except ValueError:
            st.error(f"Unable to parse fitment score: {score_text}")
    else:
        st.error("Fitment score not found in the evaluation result.")
    
    # Display the full report
    st.markdown(crew_result)

def main_app():
    st.title("Resume Cupid")
    st.markdown("Use this app to help you decide if a candidate is a good fit for a specific role.")

    resume_first_name = ""  # Initialize resume_first_name variable

    with st.form(key='resume_form'):
        job_description = st.text_area("Paste the Job Description here. Make sure to include key aspects of the role required.", placeholder="Job description. This field should have at least 100 characters.")
        resume_file = st.file_uploader("Upload your resume", type=['pdf', 'docx'])
        role = st.text_input("Type the role for which the candidate is being evaluated:", placeholder="Enter the role here")
        
        st.write("Enter the key parameters to evaluate the resume:")
        user_skills = [st.text_input(f"Skill {i+1}", placeholder=f"Skill {i+1}") for i in range(5)]
        min_experience = st.number_input("Minimum years of experience", min_value=0, value=5)
        
        st.write("Rank the skills in order of importance (1 being the most important):")
        skill_rankings = [st.number_input(f"Rank for Skill {i+1}", min_value=1, max_value=5, value=i+1, key=f"skill_rank_{i+1}") for i in range(5)]
        
        submitted = st.form_submit_button('Submit')

    if submitted and resume_file is not None and len(job_description) > 100:
        try:
            resume_text = parse_resume(resume_file)
            
            logging.info(f"Extracted resume text: {resume_text[:1000]}")

            resume_first_name = extract_first_name(resume_text)

            resume_calibrator = create_resume_calibrator_agent(llm)
            skills_agent = create_skills_agent(llm)
            experience_agent = create_experience_agent(llm)

            parameters = user_skills + [f"{min_experience} or more years of experience"]
            weights = calculate_weights(skill_rankings)
        
            weights = [str(weight) for weight in weights]

            calibration_task = create_calibration_task(job_description, resume_text, resume_calibrator, role, parameters)
            skill_evaluation_task = create_skill_evaluation_task(job_description, resume_text, skills_agent, role, weights, user_skills)
            experience_evaluation_task = create_experience_evaluation_task(job_description, resume_text, experience_agent, role)

            crew = Crew(
                agents=[resume_calibrator, skills_agent, experience_agent],
                tasks=[calibration_task, skill_evaluation_task, experience_evaluation_task],
                verbose=True
            )
        
            try:
                crew_result = crew.kickoff()
                logging.info(f"Raw result from crew.kickoff(): {crew_result}")
                if not crew_result:
                    raise ValueError("Crew.kickoff() returned an empty result")
                processed_result = process_crew_result(crew_result)
                logging.info(f"Processed result: {processed_result}")
                
                display_crew_results(processed_result)

                input_data = {
                    "job_description": job_description,
                    "resume": resume_text,
                    "role": role,
                    "parameters": parameters,
                    "weights": weights
                }
                output_data = {
                    "crew_result": processed_result
                }
                log_run(input_data, output_data)
            except Exception as e:
                logging.error(f"Error in crew.kickoff(): {e}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                st.error(f"Error: Unable to process the resume. {str(e)}")
        
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

def log_run(input_data, output_data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
    ===== Run Log: {timestamp} =====
    Input:
    Job Description: {input_data['job_description']}
    Resume: {input_data['resume']}
    Role: {input_data['role']}
    Parameters: {input_data['parameters']}
    Weights: {input_data['weights']}

    Output:
    Crew Result: {output_data['crew_result']}

    =============================
    """
    logging.info(log_entry)

if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login()
    else:
        main_app()
