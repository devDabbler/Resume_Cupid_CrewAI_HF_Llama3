import os
import time
import streamlit as st
import re
import logging
import PyPDF2
from docx import Document
from datetime import datetime
import json
from fuzzywuzzy import fuzz, process
from transformers import pipeline
import spacy
from threading import Lock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_groq import ChatGroq
from crewai import Agent, Crew, Task
from dotenv import load_dotenv, find_dotenv
from agents_module import create_resume_calibrator_agent, create_skills_agent, create_experience_agent
from tasks import create_calibration_task, create_skill_evaluation_task, create_experience_evaluation_task
from utils import (
    extract_skills_section,
    extract_experience_section,
    fuzzy_match_skills,
    analyze_experience,
    calculate_overall_fitment,
    extract_first_name,
    save_feedback_data,
    load_feedback_data
)

# Configure logging
log_file = 'resume_calibrator.log'
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to avoid debug messages
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize the Hugging Face NER pipeline
skills_extractor = pipeline("ner", model="dslim/bert-base-NER")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load environment variables
load_dotenv(find_dotenv())

# Fetch login credentials from environment variables
LOGIN_USERNAME = os.getenv('LOGIN_USERNAME')
LOGIN_PASSWORD = os.getenv('LOGIN_PASSWORD')

# Initialize the LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)

# Feedback data file
FEEDBACK_FILE = "feedback_data.json"

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

def parse_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logging.error(f"Error parsing PDF: {str(e)}")
        return ""

def parse_docx(file):
    try:
        doc = Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        logging.error(f"Error parsing DOCX: {str(e)}")
        return ""

def parse_resume(file):
    try:
        if file.type == "application/pdf":
            return parse_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return parse_docx(file)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        logging.error(f"Error parsing resume: {str(e)}")
        return ""

def extract_skills(resume_text):
    try:
        entities = skills_extractor(resume_text)
        skills = [entity['word'] for entity in entities if entity['entity'] in ['B-SKILL', 'I-SKILL']]
        return list(set(skills))
    except Exception as e:
        logging.error(f"Error extracting skills: {str(e)}")
        return []

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = Lock()

    def __call__(self, f):
        def wrapped(*args, **kwargs):
            with self.lock:
                now = time.time()
                self.calls = [c for c in self.calls if c > now - self.period]
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.calls[0] - (now - self.period)
                    time.sleep(sleep_time)
                self.calls.append(time.time())
            return f(*args, **kwargs)
        return wrapped

# Create a rate limiter instance
groq_rate_limiter = RateLimiter(max_calls=25, period=60)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type((ValueError, TypeError)))
@groq_rate_limiter
def make_groq_api_call(messages):
    try:
        response = llm(messages)
        return response
    except Exception as e:
        logging.error(f"Error making Groq API call: {str(e)}")
        raise

@st.cache_data(ttl=3600)
def cached_api_call(call_hash):
    try:
        return make_groq_api_call(call_hash)
    except Exception as e:
        logging.error(f"Error in cached_api_call: {str(e)}")
        raise

def extract_education_section(resume_text):
    education_keywords = ["Education", "Academic Background", "Qualifications"]
    education_section = ""
    
    for keyword in education_keywords:
        match = re.search(f"{keyword}.*", resume_text, re.IGNORECASE | re.DOTALL)
        if match:
            education_section = match.group()
            break
    
    return education_section

def parse_education_entries(education_section):
    education_entries = re.findall(r"([A-Za-z\s]*degree|Bachelors|Masters|PhD|Doctorate|Associate|Diploma)[^:]*(:?\d{4})?", education_section, re.IGNORECASE)
    return education_entries

def score_education(education_entries):
    score = 0
    max_score = 100

    for entry in education_entries:
        degree, year = entry
        degree_score = 0

        if re.search(r"PhD|Doctorate", degree, re.IGNORECASE):
            degree_score += 40
        elif re.search(r"Masters", degree, re.IGNORECASE):
            degree_score += 30
        elif re.search(r"Bachelors", degree, re.IGNORECASE):
            degree_score += 20
        elif re.search(r"Associate|Diploma", degree, re.IGNORECASE):
            degree_score += 10

        if year:
            current_year = datetime.now().year
            years_since_graduation = current_year - int(year)
            year_score = max(0, 10 - years_since_graduation)
            degree_score += year_score

        score += degree_score

    return min(score, max_score)

def extract_education_score(resume_text):
    education_section = extract_education_section(resume_text)
    education_entries = parse_education_entries(education_section)
    education_score = score_education(education_entries)
    return education_score

def extract_skill_score(skill_evaluation):
    overall_skill_score = skill_evaluation.get("overall_skill_score", 0)
    return overall_skill_score

def extract_experience_score(experience_evaluation):
    experience_fitment_score = experience_evaluation.get("experience_fitment_score", 0)
    return experience_fitment_score

def extract_industry_relevance(experience_evaluation):
    industry_relevance_score = experience_evaluation.get("industry_relevance", 0)
    return industry_relevance_score

def extract_skill_gaps(skill_evaluation):
    skill_gaps = skill_evaluation.get("skill_gaps", "No specific skill gaps identified.")
    return skill_gaps

def perform_evaluation(job_description, resume_text, role, user_skills, min_experience):
    try:
        # Create agents
        resume_calibrator = create_resume_calibrator_agent(llm)
        skills_agent = create_skills_agent(llm)
        experience_agent = create_experience_agent(llm)

        # Extract sections from resume
        skills_keywords = ["Skills", "Technical Skills", "Programming Languages", "Tools", "Technologies"]
        skills_section = extract_skills_section(resume_text, skills_keywords)
        experience_section = extract_experience_section(resume_text)

        # Create tasks
        calibration_task = create_calibration_task(
            job_description=job_description,
            resume=resume_text,
            resume_calibrator=resume_calibrator,
            role=role,
            parameters={"user_skills": user_skills, "min_experience": min_experience}
        )

        skill_evaluation_task = create_skill_evaluation_task(
            job_description=job_description,
            resume_skills=skills_section,
            skills_agent=skills_agent,
            role=role,
            weights=[1] * len(user_skills),
            required_skills=user_skills
        )

        experience_evaluation_task = create_experience_evaluation_task(
            job_description=job_description,
            resume_text=experience_section,
            experience_agent=experience_agent,
            role=role
        )

        # Create and kickoff crew
        crew = Crew(
            agents=[resume_calibrator, skills_agent, experience_agent],
            tasks=[calibration_task, skill_evaluation_task, experience_evaluation_task],
            verbose=False  # Ensure verbose is set to False to avoid debug messages
        )

        logger.info("Before Crew.kickoff()")
        results = crew.kickoff()
        logger.info("After Crew.kickoff()")

        # Process results
        if not results:
            logger.error("Crew.kickoff() returned an empty result")
            raise ValueError("Crew.kickoff() returned an empty result")

        if results and len(results) >= 3:
            calibration_result = results[0]
            skill_evaluation = results[1]
            experience_evaluation = json.loads(results[2])  # This one seems to be in JSON format
        else:
            raise ValueError("Insufficient results from Crew kickoff.")

        return experience_evaluation

    except Exception as e:
        logger.error(f"Error in perform_evaluation: {str(e)}")
        st.error(f"An error occurred during evaluation: {str(e)}")
        return {
            "experience_fitment_score": 0,
            "interview_recommendation": "Unable to determine due to error",
            "overall_assessment": "Error in processing",
            "relevant_experience": [],
            "gaps": [],
            "areas_of_improvement": [],
            "concluding_statement": "An error occurred during the evaluation process."
        }

# Replace st.experimental_rerun() with st.rerun()
def login():
    st.markdown("""
        <style>
        .reportview-container .main .block-container {
            max-width: 1000px;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .auth-form {
            width: 100%;
            max-width: 800px;
            margin: 1rem auto 0;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stTextInput > div > div > input {
            background-color: white;
            padding: 0.5rem;
            border: 1px solid #adb5bd;
        }
        .stButton > button {
            width: 100%;
        }
        .footer-text {
            text-align: center; 
            width: 100%;
            max-width: 800px;
            margin: 1rem auto;
        }
        h3 {
            margin-bottom: 1rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; font-size: 48px; margin-bottom: 0;'>💘</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; margin-top: 0;'>Welcome to Resume Cupid</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Login to access the resume evaluation tool</h3>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=False):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                username = st.session_state.username
                password = st.session_state.password
                if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
                    st.session_state["logged_in"] = True
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div class='footer-text'>Don't have an account? <a href='mailto:hello@resumecupid.ai'>Contact us</a> to get started!</div>", unsafe_allow_html=True)

def main_app():
    st.title("Resume Cupid")
    st.markdown("Use this app to help you decide if a candidate is a good fit for a specific role.")

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if "form_key" not in st.session_state:
        st.session_state["form_key"] = 0

    if not st.session_state["logged_in"]:
        login()
    else:
        # Increment the form key to ensure uniqueness
        st.session_state["form_key"] += 1
        form_key = f"resume_form_{st.session_state['form_key']}"

        with st.form(key=form_key):
            job_description = st.text_area("Paste the Job Description here.")
            resume_file = st.file_uploader("Upload your resume", type=['pdf', 'docx'])
            role = st.text_input("Type the role for which the candidate is being evaluated:", placeholder="Enter the role here")

            st.write("Enter the key parameters to evaluate the resume:")
            user_skills = [st.text_input(f"Skill {i+1}", key=f"skill_{i+1}_{st.session_state['form_key']}", placeholder=f"Skill {i+1}") for i in range(5)]
            min_experience = st.number_input("Minimum years of experience", min_value=0, value=5, key=f"min_exp_{st.session_state['form_key']}")

            st.write("Rank the skills in order of importance (1 being the most important):")
            skill_rankings = [st.number_input(f"Rank for Skill {i+1}", min_value=1, max_value=5, value=i+1, key=f"skill_rank_{i+1}_{st.session_state['form_key']}") for i in range(5)]

            submitted = st.form_submit_button('Submit')

        if submitted:
            if not job_description:
                st.error("Please provide a job description.")
            elif not resume_file:
                st.error("Please upload a resume.")
            elif not role:
                st.error("Please specify the role for evaluation.")
            elif not any(user_skills):
                st.error("Please provide at least one skill for evaluation.")
            else:
                with st.spinner('Evaluating resume... This may take a few minutes.'):
                    try:
                        resume_text = parse_resume(resume_file)
                        evaluation_result = perform_evaluation(job_description, resume_text, role, user_skills, min_experience)

                        st.subheader("Evaluation Results")
                        st.write(f"Fitment Score: {evaluation_result.get('experience_fitment_score', 'N/A')}%")
                        st.write(f"Interview Recommendation: {evaluation_result.get('interview_recommendation', 'N/A')}")

                        st.subheader("Detailed Analysis")
                        st.write("Overall Assessment:", evaluation_result.get('overall_assessment', 'N/A'))

                        st.write("Relevant Experience:")
                        for exp in evaluation_result.get('relevant_experience', []):
                            st.write(f"- {exp.get('experience', 'N/A')} (Relevance: {exp.get('relevance_score', 'N/A')}/10)")

                        st.write("Skill Gaps:")
                        for gap in evaluation_result.get('gaps', []):
                            st.write(f"- {gap.get('gap', 'N/A')} (Severity: {gap.get('severity', 'N/A')}/10)")

                        st.write("Areas to Improve:")
                        for area in evaluation_result.get('areas_of_improvement', []):
                            st.write(f"- {area.get('area', 'N/A')} (Priority: {area.get('priority', 'N/A')})")

                        st.write("Concluding Statement:", evaluation_result.get('concluding_statement', 'N/A'))

                        # Feedback form
                        st.subheader("Feedback")
                        feedback_form_key = f"feedback_form_{st.session_state['form_key']}"
                        with st.form(key=feedback_form_key):
                            name = st.text_input("Name of Person Leaving Feedback")
                            resume_first_name = st.text_input("Candidate or Resume Name", value=extract_first_name(resume_text))
                            role_input = st.text_input("Role", value=role, disabled=True)
                            client = st.text_input("Client")
                            accuracy_rating = st.select_slider("Accuracy of the evaluation:", options=[1, 2, 3, 4, 5])
                            content_rating = st.select_slider("Quality of the report content:", options=[1, 2, 3, 4, 5])
                            suggestions = st.text_area("Please provide any suggestions for improvement:")
                            submit_feedback = st.form_submit_button("Submit Feedback")

                            if submit_feedback:
                                feedback_entry = {
                                    "run_id": st.session_state.get("run_id", ""),
                                    "resume_id": resume_first_name,
                                    "job_role_id": role_input,
                                    "name": name,
                                    "accuracy_rating": accuracy_rating,
                                    "content_rating": content_rating,
                                    "suggestions": suggestions,
                                    "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "client": client
                                }
                                feedback_data = load_feedback_data()
                                feedback_data.append(feedback_entry)
                                save_feedback_data(feedback_data)
                                st.success("Thank you for your feedback!")

                    except Exception as e:
                        st.error(f"An error occurred during evaluation: {str(e)}")
                        logger.error(f"Error in main_app: {str(e)}")

if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login()
    else:
        main_app()
