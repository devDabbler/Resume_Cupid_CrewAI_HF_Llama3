import os
import uuid
import sys
import traceback
from typing import List, Dict, Any
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json
import sqlite3
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from utils import (
    extract_skills_contextually, parse_resume, extract_first_name,
    extract_experience, calculate_experience_score, calculate_education_score,
    calculate_relevant_experience_score, evaluate_project_complexity,
    extract_text_from_pdf, extract_text_from_docx, extract_skills_nlp,
    JOB_ROLES, calculate_robust_fitment, extract_education_from_resume
)
from agents_module import (
    create_resume_calibrator_agent, create_skills_agent, 
    create_experience_agent, create_education_agent, 
    create_project_complexity_agent
)

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging with UTF-8 encoding
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")

class RunIdFilter(logging.Filter):
    def filter(self, record):
        record.run_id = getattr(st.session_state, 'run_id', 'N/A')
        return True

handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - run_id: %(run_id)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.addFilter(RunIdFilter())

# Feedback database setup
db_path = os.path.join(BASE_DIR, "feedback.db")

def init_db():
    logger.info("Initializing database", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS feedback
                         (id TEXT PRIMARY KEY, name TEXT, role_input TEXT, client TEXT, 
                          accuracy_rating INTEGER, content_rating INTEGER, 
                          suggestions TEXT, submitted_at TEXT)''')
        conn.commit()
        conn.close()
        logger.info(f"Database initialized successfully at {db_path}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        raise

# Initialize run_id
if 'run_id' not in st.session_state:
    st.session_state.run_id = str(uuid.uuid4())

logger.info("Application started")

def save_feedback(feedback_entry):
    logger.info("Attempting to save feedback", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''INSERT OR REPLACE INTO feedback VALUES 
                         (:id, :name, :role_input, :client, :accuracy_rating, 
                          :content_rating, :suggestions, :submitted_at)''', 
                      feedback_entry)
        conn.commit()
        conn.close()
        logger.info(f"Feedback saved successfully: {feedback_entry['id']}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    except sqlite3.Error as e:
        logger.error(f"Database error when saving feedback: {e}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    except Exception as e:
        logger.error(f"Unexpected error when saving feedback: {e}", extra={'run_id': st.session_state.get('run_id', 'N/A')})

# Load environment variables
load_dotenv(find_dotenv())
logger.info("Environment variables loaded", extra={'run_id': st.session_state.get('run_id', 'N/A')})

# Fetch login credentials and API key from environment variables
LOGIN_USERNAME = os.getenv('LOGIN_USERNAME')
LOGIN_PASSWORD = os.getenv('LOGIN_PASSWORD')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not all([LOGIN_USERNAME, LOGIN_PASSWORD, GROQ_API_KEY]):
    logger.error("Missing required environment variables", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    st.error("Missing required environment variables. Please check your .env file.")
    st.stop()

@st.cache_resource
def initialize_llm():
    """Initialize and return the LLM (Language Model) instance."""
    logger.info("Initializing LLM", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    try:
        llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.1,
            groq_api_key=GROQ_API_KEY,
            max_tokens=1024,
            top_p=1,
            streaming=True,
            callbacks=None
        )
        test_response = llm.invoke("Hello, can you hear me?")
        logger.info(f"LLM test response: {test_response}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        raise

def login():
    logger.info("Displaying login page", extra={'run_id': st.session_state.get('run_id', 'N/A')})
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
            submit_button = st.form_submit_button("Login", on_click=handle_login)

    st.markdown("<div class='footer-text'>Don't have an account? <a href='mailto:hello@resumecupid.ai'>Contact us</a> to get started!</div>", unsafe_allow_html=True)

def handle_login():
    username = st.session_state.username
    password = st.session_state.password
    logger.info(f"Login attempt for user: {username}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
        st.session_state["logged_in"] = True
        logger.info("Login successful", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        st.success("Login successful!")
    else:
        logger.warning("Login failed: Invalid credentials", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        st.error("Invalid username or password")

def parse_resume(file, role: str):
    logger.info(f"Starting to parse resume: {file.name}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    
    try:
        if file.type == "application/pdf":
            logger.info("Parsing PDF file", extra={'run_id': st.session_state.get('run_id', 'N/A')})
            file_content = file.read()
            logger.info(f"PDF file size: {len(file_content)} bytes", extra={'run_id': st.session_state.get('run_id', 'N/A')})
            resume_text = extract_text_from_pdf(file_content)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            logger.info("Parsing DOCX file", extra={'run_id': st.session_state.get('run_id', 'N/A')})
            file_content = file.read()
            logger.info(f"DOCX file size: {len(file_content)} bytes", extra={'run_id': st.session_state.get('run_id', 'N/A')})
            resume_text = extract_text_from_docx(file_content)
        else:
            logger.error(f"Unsupported file format: {file.type}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
            raise ValueError(f"Unsupported file format: {file.type}")
        
        if not resume_text:
            logger.error("Extracted resume text is empty", extra={'run_id': st.session_state.get('run_id', 'N/A')})
            raise ValueError("Failed to extract text from resume: Extracted text is empty")
        
        logger.info(f"Successfully extracted text from resume. Length: {len(resume_text)}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        logger.debug(f"Resume text preview: {resume_text[:500]}...", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        
        experiences = extract_experience(resume_text, role)
        logger.info(f"Extracted {len(experiences)} experiences", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        
        skills = extract_skills_nlp(resume_text)
        logger.info(f"Extracted {len(skills)} skills", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        
        education = extract_education_from_resume(resume_text)
        logger.info(f"Extracted education: {education}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        
        return resume_text, experiences, list(skills), education
    
    except Exception as e:
        logger.error(f"Error parsing resume: {str(e)}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        logger.error(traceback.format_exc(), extra={'run_id': st.session_state.get('run_id', 'N/A')})
        raise ValueError(f"Failed to extract text from resume: {str(e)}")

def evaluate_candidate(resume_text: str, job_description: str, role: str, required_years: int, skills: List[str], weights: List[float]) -> Dict[str, Any]:
    logger.info("Starting candidate evaluation", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    try:
        # Initialize LLM
        llm = initialize_llm()

        # Create agents
        resume_calibrator_agent = create_resume_calibrator_agent(llm)
        skills_agent = create_skills_agent(llm)
        experience_agent = create_experience_agent(llm)
        education_agent = create_education_agent(llm)
        project_complexity_agent = create_project_complexity_agent(llm)

        # Skill evaluation
        skill_score, matched_skills, missing_skills = skills_agent.evaluate_skills(resume_text, job_description, role, skills, weights)
        logger.info(f"Skill evaluation complete. Score: {skill_score}, Matched: {matched_skills}, Missing: {missing_skills}", extra={'run_id': st.session_state.get('run_id', 'N/A')})

        # Experience evaluation
        experience_evaluation = experience_agent.evaluate_experience(resume_text, job_description, role)
        experience_score = experience_evaluation['experience_score']
        relevant_experience_score = experience_evaluation['relevant_experience_score']
        experiences = experience_evaluation['experiences']
        logger.info(f"Experience evaluation complete. Score: {experience_score}, Relevant Score: {relevant_experience_score}", extra={'run_id': st.session_state.get('run_id', 'N/A')})

        # Education evaluation
        education_evaluation = education_agent.evaluate_education(resume_text)
        education_score = education_evaluation['education_score']
        education_level = education_evaluation['education_level']
        degrees = education_evaluation['degrees']
        logger.info(f"Education evaluation complete. Score: {education_score}, Level: {education_level}, Degrees: {degrees}", extra={'run_id': st.session_state.get('run_id', 'N/A')})

        # Project complexity evaluation
        project_complexity_evaluation = project_complexity_agent.evaluate_project_complexity(resume_text, job_description, role)
        project_complexity_score = project_complexity_evaluation['project_complexity_score']
        project_details = project_complexity_evaluation['project_details']
        logger.info(f"Project complexity evaluation complete. Score: {project_complexity_score}, Details: {project_details}", extra={'run_id': st.session_state.get('run_id', 'N/A')})

        # Overall fitment evaluation
        resume_fitment_score = resume_calibrator_agent.calibrate(resume_text, job_description, role)
        logger.info(f"Resume calibrator fitment score: {resume_fitment_score}", extra={'run_id': st.session_state.get('run_id', 'N/A')})

        # Calculate overall fitment
        overall_fitment = calculate_robust_fitment(
            experience_score, skill_score * 100, education_score, project_complexity_score, role
        )
        logger.info(f"Overall fitment calculated: {overall_fitment}", extra={'run_id': st.session_state.get('run_id', 'N/A')})

        strengths = []
        weaknesses = []

        if skill_score >= 0.7:
            strengths.append(f"Candidate has a good skill match ({skill_score:.2f}) for the job requirements.")
        else:
            weaknesses.append(f"Candidate's skills ({skill_score:.2f}) could be improved to better match the job requirements.")

        if experience_score >= 70:
            strengths.append(f"Candidate has significant relevant experience ({experience_score:.2f}%).")
        else:
            weaknesses.append(f"Candidate's relevant experience ({experience_score:.2f}%) could be stronger for this role.")

        if education_score >= 60:
            strengths.append(f"Candidate has a strong educational background ({education_score:.2f}%).")
        else:
            weaknesses.append(f"Candidate's educational background ({education_score:.2f}%) might need improvement for this role.")

        if project_complexity_score >= 70:
            strengths.append(f"Candidate has handled complex projects effectively ({project_complexity_score:.2f}%).")
        else:
            weaknesses.append(f"Candidate's experience with complex projects ({project_complexity_score:.2f}%) could be improved.")

        final_eval = {
            "overall_fitment": overall_fitment,
            "resume_fitment_score": resume_fitment_score,
            "skill_match": skill_score * 100,
            "experience_score": experience_score,
            "relevant_experience_score": relevant_experience_score,
            "education_score": education_score,
            "education_level": education_level,
            "degrees": degrees,
            "project_complexity_score": project_complexity_score,
            "project_details": project_details,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "experiences": experiences,
            "target_skills": skills,
            "skill_weights": weights,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "interview_recommendation": ""
        }

        # Adjust interview recommendation based on overall fitment and resume fitment score
        if final_eval["overall_fitment"] >= 80 and final_eval["resume_fitment_score"] >= 80:
            final_eval["interview_recommendation"] = "Strongly recommend for interview"
        elif final_eval["overall_fitment"] >= 70 or final_eval["resume_fitment_score"] >= 70:
            final_eval["interview_recommendation"] = "Recommend for interview"
        elif final_eval["overall_fitment"] >= 60 or final_eval["resume_fitment_score"] >= 60:
            final_eval["interview_recommendation"] = "Consider for interview with reservations"
        else:
            final_eval["interview_recommendation"] = "May not be suitable for this role"

        logger.info("Candidate evaluation completed", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        return final_eval

    except Exception as e:
        logger.error(f"Error in evaluate_candidate: {str(e)}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        logger.error(traceback.format_exc(), extra={'run_id': st.session_state.get('run_id', 'N/A')})
        raise

def calculate_weights(skill_rankings: List[int]) -> List[float]:
    logger.info("Calculating skill weights", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    if not skill_rankings:
        logger.warning("Empty skill rankings provided", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        return []

    total = sum(skill_rankings)
    if total == 0:
        logger.warning("All skill rankings are zero", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        return [1.0 / len(skill_rankings)] * len(skill_rankings)

    weights = [rank / total for rank in skill_rankings]

    softening_factor = 0.5
    softened_weights = [
        w * softening_factor + (1 - softening_factor) / len(weights)
        for w in weights
    ]

    normalized_weights = [w / sum(softened_weights) for w in softened_weights]

    logger.info(f"Calculated weights: {normalized_weights}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    return normalized_weights

def get_recommendation(overall_score):
    logger.info(f"Getting recommendation for overall score: {overall_score}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    if overall_score >= 75:
        return "This candidate is a strong fit and should be considered for an interview."
    elif overall_score >= 60:
        return "This candidate shows potential and may be worth considering for an interview, with additional screening."
    elif overall_score >= 50:
        return "This candidate has some relevant skills but may not be a strong fit. Consider only if the candidate pool is limited."
    else:
        return "This candidate does not appear to be a good fit for the job based on the initial assessment."

def display_results(evaluation_result):
    logger.info("Displaying evaluation results", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    
    # Overall Fitment and Recommendation
    st.header("Candidate Evaluation Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Fitment Score", f"{evaluation_result['overall_fitment']:.1f}%")
    with col2:
        st.info(f"**Recommendation:** {evaluation_result['interview_recommendation']}")
    
    # Fitment Analysis
    st.subheader("Fitment Analysis")
    fitment_analysis = generate_fitment_analysis(evaluation_result)
    st.write(fitment_analysis)
    
    # Gap Analysis
    st.subheader("Gap Analysis and Discussion Points")
    gap_analysis = generate_gap_analysis(evaluation_result)
    st.write(gap_analysis)
    
    # Detailed Scores
    with st.expander("View Detailed Scores"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Skills Score", f"{evaluation_result['skill_match']:.1f}%")
        with col2:
            st.metric("Experience Score", f"{evaluation_result['experience_score']:.1f}%")
        with col3:
            st.metric("Education Score", f"{evaluation_result['education_score']:.1f}%")
        with col4:
            st.metric("Project Complexity", f"{evaluation_result['project_complexity_score']:.1f}%")

def generate_fitment_analysis(evaluation_result):
    analysis = []
    
    if evaluation_result['overall_fitment'] >= 80:
        analysis.append("The candidate demonstrates an excellent overall fit for the position.")
    elif evaluation_result['overall_fitment'] >= 70:
        analysis.append("The candidate shows a good fit for the position with some areas for potential growth.")
    else:
        analysis.append("The candidate may need further development to fully meet the position requirements.")
    
    if evaluation_result['skill_match'] >= 80:
        analysis.append("Their skill set aligns well with the job requirements.")
    elif evaluation_result['skill_match'] >= 70:
        analysis.append("Their skills match many of the job requirements, with some areas for potential improvement.")
    else:
        analysis.append("There may be a gap between the candidate's current skills and the job requirements.")
    
    if evaluation_result['experience_score'] >= 80:
        analysis.append("The candidate's experience is highly relevant to the role.")
    elif evaluation_result['experience_score'] >= 70:
        analysis.append("The candidate has relevant experience, though some aspects of the role may be new to them.")
    else:
        analysis.append("The candidate may need additional experience in key areas related to this role.")
    
    if evaluation_result['education_score'] >= 80:
        analysis.append(f"Their {evaluation_result['education_level']} provides a strong educational foundation for this position.")
    else:
        analysis.append("The candidate's educational background may need to be supplemented with additional training or certifications.")
    
    return " ".join(analysis)

def generate_gap_analysis(evaluation_result):
    analysis = []
    
    if evaluation_result['missing_skills']:
        analysis.append("Key skills to discuss:")
        for skill in evaluation_result['missing_skills'][:3]:  # Top 3 missing skills
            analysis.append(f"- Proficiency and experience with {skill}")
    
    if evaluation_result['experience_score'] < 80:
        analysis.append("Experience areas to explore:")
        analysis.append("- Specific projects or roles that align closely with our current needs")
        analysis.append("- How they've handled challenges similar to what we're facing")
    
    if evaluation_result['project_complexity_score'] < 80:
        analysis.append("Project complexity:")
        analysis.append("- Examples of the most complex projects they've worked on")
        analysis.append("- How they've managed large-scale or multi-faceted projects")
    
    if not analysis:
        analysis.append("The candidate appears to be a strong fit across all areas. Consider discussing:")
        analysis.append("- Their most impactful projects or achievements")
        analysis.append("- How they see themselves growing in this role")
        analysis.append("- Any unique perspectives or innovations they could bring to the team")
    
    return "\n".join(analysis)

def main_app():
    logger.info("Starting main application", extra={'run_id': st.session_state.get('run_id', 'N/A')})
    st.title("Resume Cupid")
    st.markdown("Use this app to help you decide if a candidate is a good fit for a specific role.")

    # Initialize role in session state
    if 'role' not in st.session_state:
        st.session_state.role = ""

    # Helper function to format roles
    def format_role(role):
        return role.replace('_', ' ').title()

    # Helper function to reverse format roles
    def reverse_format_role(role):
        return role.replace(' ', '_').lower()

    with st.form(key='resume_form_1'):
        job_description = st.text_area("Paste the Job Description here. Make sure to include key aspects of the role required.", placeholder="Job description. This field should have at least 100 characters.")
        resume_file = st.file_uploader("Upload your resume", type=['pdf', 'docx'])
        role_options = list(JOB_ROLES.keys()) + ["Other"]
        formatted_roles = [format_role(role) for role in role_options]
        selected_role = st.selectbox("Select the job role", options=formatted_roles, key='selected_role')
        
        st.caption("If 'Other' is selected above, please specify the custom job title below:")
        custom_role = st.text_input("Custom Job Title (required if 'Other' is selected)", key='custom_role')
        
        st.write("Enter the key parameters to evaluate the resume content.")
        skills = []
        skill_rankings = []
        for i in range(5):
            col1, col2 = st.columns(2)
            with col1:
                skill = st.text_input(f"Skill {i+1}", key=f"skill_{i}")
                skills.append(skill)
            with col2:
                rank = st.number_input(f"Rank for Skill {i+1}", value=i+1, key=f"skill_rank_{i}")
                skill_rankings.append(rank)

        min_experience = st.number_input("Minimum years of experience", min_value=0, value=2)

        submitted = st.form_submit_button('Submit')

    if submitted:
        logger.info("Form submitted", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        if selected_role == "Other":
            if custom_role.strip():
                st.session_state.role = custom_role
                logger.info(f"Custom role selected: {custom_role}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
            else:
                logger.warning("'Other' selected but no custom role provided", extra={'run_id': st.session_state.get('run_id', 'N/A')})
                st.error("Please enter a custom job title when selecting 'Other'.")
                st.stop()
        else:
            st.session_state.role = reverse_format_role(selected_role)
            logger.info(f"Role selected: {st.session_state.role}", extra={'run_id': st.session_state.get('run_id', 'N/A')})

    if submitted:
        logger.info("Form submitted", extra={'run_id': st.session_state.get('run_id', 'N/A')})
        if resume_file and len(job_description) > 100:
            logger.info("Processing resume and job description", extra={'run_id': st.session_state.get('run_id', 'N/A')})
            try:
                resume_text, experiences, extracted_skills, education = parse_resume(resume_file, st.session_state.role)
            
                if resume_text is None:
                    logger.error("Failed to extract text from resume", extra={'run_id': st.session_state.get('run_id', 'N/A')})
                    raise ValueError("Failed to extract text from resume")

                weights = calculate_weights(skill_rankings)

                skills = [skill for skill in skills if skill.strip()]
                weights = weights[:len(skills)]

                evaluation_result = evaluate_candidate(resume_text, job_description, st.session_state.role, min_experience, skills, weights)
                
                display_results(evaluation_result)

                with st.form(key='feedback_form_1'):
                    st.subheader("Feedback")
                    name = st.text_input("Name of Person Leaving Feedback")
                    role_input = st.text_input("Candidate or Resume Name", value=extract_first_name(resume_text))
                    client = st.text_input("Client")
                    accuracy_rating = st.select_slider("Accuracy of the evaluation:", options=[1, 2, 3, 4, 5])
                    content_rating = st.select_slider("Quality of the report content:", options=[1, 2, 3, 4, 5])
                    suggestions = st.text_area("Please provide any suggestions for improvement:")
                    submitted_at = datetime.now().isoformat()
                    submit_feedback = st.form_submit_button("Submit Feedback")

                if submit_feedback:
                    logger.info("Feedback form submitted", extra={'run_id': st.session_state.get('run_id', 'N/A')})
                    feedback_entry = {
                        "id": str(uuid.uuid4()),
                        "name": name,
                        "role_input": role_input,
                        "client": client,
                        "accuracy_rating": accuracy_rating,
                        "content_rating": content_rating,
                        "suggestions": suggestions,
                        "submitted_at": submitted_at
                    }
                    logger.info(f"Feedback entry to be saved: {feedback_entry}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
                    save_feedback(feedback_entry)
                    st.success("Thank you for your feedback!")

            except Exception as e:
                logger.error(f"Error processing resume: {str(e)}", extra={'run_id': st.session_state.get('run_id', 'N/A')})
                st.error(f"An error occurred while processing the resume: {str(e)}")
                logger.error(traceback.format_exc(), extra={'run_id': st.session_state.get('run_id', 'N/A')})
        else:
            logger.warning("Form submitted but missing required fields", extra={'run_id': st.session_state.get('run_id', 'N/A')})
            st.error("Please upload a resume and ensure the job description has at least 100 characters.")

if __name__ == "__main__":
    if 'run_id' not in st.session_state:
        st.session_state.run_id = str(uuid.uuid4())
    logger.info(f"Starting new session with run_id: {st.session_state.run_id}")

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    init_db()

    if not st.session_state["logged_in"]:
        login()
    else:
        main_app()

    logger.info("Application execution completed", extra={'run_id': st.session_state.run_id})
