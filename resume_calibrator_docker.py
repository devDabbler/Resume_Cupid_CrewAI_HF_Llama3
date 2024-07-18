import os
import time
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import re
import logging
import PyPDF2
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
from fuzzywuzzy import fuzz
import uuid
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import hashlib
from threading import Lock
import traceback
from langchain_core.messages import HumanMessage, AIMessage

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

# Improved parsing functions
def parse_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    # Use regex to extract structured information
    name = re.search(r"Name:?\s*([\w\s]+)", text, re.IGNORECASE)
    email = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    phone = re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    
    structured_data = {
        "full_text": text,
        "name": name.group(1) if name else "Not found",
        "email": email.group(0) if email else "Not found",
        "phone": phone.group(0) if phone else "Not found",
    }
    return structured_data

def parse_docx(file):
    doc = Document(file)
    full_text = ""
    structured_data = {
        "full_text": "",
        "sections": {}
    }
    current_section = "header"

    for para in doc.paragraphs:
        full_text += para.text + "\n"
        if para.style.name.startswith('Heading'):
            current_section = para.text
            structured_data["sections"][current_section] = ""
        else:
            structured_data["sections"][current_section] += para.text + "\n"

    structured_data["full_text"] = full_text
    return structured_data

def parse_resume(file):
    if file.type == "application/pdf":
        return parse_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return parse_docx(file)
    else:
        raise ValueError("Unsupported file format")

# Fetch login credentials from environment variables
LOGIN_USERNAME = os.getenv('LOGIN_USERNAME')
LOGIN_PASSWORD = os.getenv('LOGIN_PASSWORD')

# Login form
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
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div class='footer-text'>Don't have an account? <a href='mailto:hello@resumecupid.ai'>Contact us</a> to get started!</div>", unsafe_allow_html=True)

def calculate_weights(rankings):
    total = sum(rankings)
    return [rank / total for rank in rankings]

def process_crew_result(result):
    if isinstance(result, dict):
        return result
    elif isinstance(result, str):
        return parse_unstructured_result(result)
    elif hasattr(result, 'content'):
        return parse_unstructured_result(result.content)
    else:
        return str(result)

def parse_unstructured_result(content):
    sections = content.split('\n\n')
    result = {}
    current_section = None
    for section in sections:
        if ':' in section:
            key, value = section.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            result[key] = value.strip()
            current_section = key
        elif current_section:
            result[current_section] += '\n' + section.strip()
    return result

def display_crew_results(crew_result):
    if isinstance(crew_result, str):
        crew_result = process_crew_result(crew_result)
    
    if isinstance(crew_result, dict):
        if 'experience_fitment_score' in crew_result:
            st.subheader(f"Experience Fitment Score: {crew_result['experience_fitment_score']}")
        
        if 'overall_assessment' in crew_result:
            st.subheader("Overall Assessment")
            st.write(crew_result['overall_assessment'])
        
        for key in ['relevant_experience', 'irrelevant_experience', 'gaps', 'areas_of_improvement', 'concluding_statement']:
            if key in crew_result:
                st.subheader(key.replace('_', ' ').title())
                st.write(crew_result[key])
    else:
        st.write(crew_result)

def process_large_text(text, chunk_size=5000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunks))
    
    return combine_results(results)

def process_chunk(chunk):
    # Basic NLP processing
    doc = nlp(chunk)
    
    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Extract key phrases (simplified)
    key_phrases = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ']]
    
    # Count word frequency
    word_freq = {}
    for token in doc:
        if token.is_alpha and not token.is_stop:
            word_freq[token.text] = word_freq.get(token.text, 0) + 1
    
    return {
        "text": chunk,
        "entities": entities,
        "key_phrases": key_phrases,
        "word_freq": word_freq
    }

def combine_results(results):
    combined = {
        "text": "",
        "entities": [],
        "key_phrases": [],
        "word_freq": {}
    }
    
    for result in results:
        combined["text"] += result["text"]
        combined["entities"].extend(result["entities"])
        combined["key_phrases"].extend(result["key_phrases"])
        for word, freq in result["word_freq"].items():
            combined["word_freq"][word] = combined["word_freq"].get(word, 0) + freq
    
    return combined

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
groq_rate_limiter = RateLimiter(max_calls=25, period=60)  # 25 calls per minute to be safe

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ValueError, TypeError))
)
@groq_rate_limiter
def make_groq_api_call(messages):
    try:
        # Convert messages to the correct format if necessary
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, str):
                formatted_messages.append(HumanMessage(content=msg))
            elif isinstance(msg, dict):
                if msg.get('role') == 'human':
                    formatted_messages.append(HumanMessage(content=msg['content']))
                elif msg.get('role') == 'ai':
                    formatted_messages.append(AIMessage(content=msg['content']))
            else:
                formatted_messages.append(msg)
        
        response = llm(formatted_messages)
        
        # Try to parse the response as JSON
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = response.content

        return result
    except Exception as e:
        logging.error(f"Error in API call: {str(e)}")
        raise

# Caching function for API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_api_call(call_hash):
    try:
        return make_groq_api_call(call_hash)
    except Exception as e:
        logging.error(f"Error in cached_api_call: {str(e)}")
        raise

def main_app():
    st.title("Resume Cupid")
    st.markdown("Use this app to help you decide if a candidate is a good fit for a specific role.")

    # Generate a unique ID for each run if it does not exist
    if "run_id" not in st.session_state:
        st.session_state["run_id"] = str(uuid.uuid4())
    run_id = st.session_state["run_id"]

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
            resume_data = parse_resume(resume_file)
            resume_text = resume_data["full_text"]
            
            logging.info(f"Extracted resume text: {resume_text[:1000]}")

            resume_first_name = extract_first_name(resume_text)

            if len(resume_text) > 10000 or len(job_description) > 5000:
                resume_text = process_large_text(resume_text)
                job_description = process_large_text(job_description)

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
            
            # Enhanced progress bar
            status_text = st.empty()
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.05)  # Simulate some work
                progress_bar.progress(i + 1)
                if i < 33:
                    status_text.text("Starting the calibration...")
                elif i < 66:
                    status_text.text("Processing the evaluation...")
                else:
                    status_text.text("Finalizing the results...")
            
            crew_result = crew.kickoff()
            if not crew_result:
                raise ValueError("Crew.kickoff() returned an empty result")
            processed_result = process_crew_result(crew_result)
            display_crew_results(processed_result)
            st.success("Evaluation Complete!")

            input_data = {
                "run_id": run_id,
                "job_description": job_description,
                "resume": resume_text,
                "role": role,
                "parameters": parameters,
                "weights": weights
            }
            output_data = {
                "run_id": run_id,
                "crew_result": processed_result
            }
            
            log_run(input_data, output_data)
        
        except Exception as e:
            logging.error(f"Error in processing: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            st.error(f"Error: Unable to process the resume. {str(e)}")
    
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
                "run_id": run_id,
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
    log_entry = {
        "timestamp": timestamp,
        "input_data": input_data,
        "output_data": output_data
    }
    
    log_file = 'run_logs.json'
    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as file:
                logs = json.load(file)
        else:
            logs = []

        logs.append(log_entry)
        
        with open(log_file, "w") as file:
            json.dump(logs, file, indent=4)
        logging.info(f"Logged run at {timestamp}")
    except Exception as e:
        logging.error(f"Error in log_run: {str(e)}")
        st.error(f"Error in logging run: {str(e)}")

if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login()
    else:
        main_app()
