import os
import traceback
from typing import List, Dict, Any
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import sys
import logging
from datetime import datetime
import json
from langchain_groq import ChatGroq
from utils import (
    extract_skills_contextually, parse_resume, extract_first_name,
    load_feedback_data, save_feedback_data, extract_experience,
    calculate_experience_score, calculate_education_score, calculate_relevant_experience_score, 
    evaluate_project_complexity, extract_text_from_pdf, extract_text_from_docx, extract_skills_nlp, JOB_ROLES, calculate_robust_fitment,
    extract_education_from_resume
)
from agents_module import (
    create_resume_calibrator_agent, create_skills_agent, 
    create_experience_agent, create_education_agent, 
    create_project_complexity_agent
)

# Configure logging
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "app.log")
logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode="w", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add console logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levellevel)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Streamlit UI setup
st.set_page_config(page_title='📝 Resume Cupid', page_icon="📝")

# Load environment variables
load_dotenv(find_dotenv())

# Feedback data file
FEEDBACK_FILE = os.getenv("FEEDBACK_FILE")
if not FEEDBACK_FILE:
    st.error("FEEDBACK_FILE environment variable is not set.")
    st.stop()

# Fetch login credentials and API key from environment variables
LOGIN_USERNAME = os.getenv('LOGIN_USERNAME')
LOGIN_PASSWORD = os.getenv('LOGIN_PASSWORD')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not all([LOGIN_USERNAME, LOGIN_PASSWORD, GROQ_API_KEY]):
    st.error("Missing required environment variables. Please check your .env file.")
    st.stop()

# Load feedback data from file
@st.cache_data
def load_feedback_data_cached():
    return load_feedback_data(FEEDBACK_FILE)

feedback_data = load_feedback_data_cached()

def initialize_llm():
    """
    Initialize and return the LLM (Language Model) instance.
    """
    logger.info("Initializing LLM")
    try:
        return ChatGroq(model="llama3-8b-8192", temperature=0.2, api_key=GROQ_API_KEY)
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise

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
            submit_button = st.form_submit_button("Login", on_click=handle_login)

    st.markdown("<div class='footer-text'>Don't have an account? <a href='mailto:hello@resumecupid.ai'>Contact us</a> to get started!</div>", unsafe_allow_html=True)

def handle_login():
    username = st.session_state.username
    password = st.session_state.password
    if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
        st.session_state["logged_in"] = True
        st.success("Login successful!")
    else:
        st.error("Invalid username or password")

def parse_resume(file, role: str):
    logger.info(f"Starting to parse resume: {file.name}")
    
    try:
        if file.type == "application/pdf":
            logger.info("Parsing PDF file")
            file_content = file.read()
            logger.info(f"PDF file size: {len(file_content)} bytes")
            resume_text = extract_text_from_pdf(file_content)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            logger.info("Parsing DOCX file")
            file_content = file.read()
            logger.info(f"DOCX file size: {len(file_content)} bytes")
            resume_text = extract_text_from_docx(file_content)
        else:
            logger.error(f"Unsupported file format: {file.type}")
            raise ValueError(f"Unsupported file format: {file.type}")
        
        if not resume_text:
            logger.error("Extracted resume text is empty")
            raise ValueError("Failed to extract text from resume: Extracted text is empty")
        
        logger.info(f"Successfully extracted text from resume. Length: {len(resume_text)}")
        logger.debug(f"Resume text preview: {resume_text[:500]}...")
        
        experiences = extract_experience(resume_text, role)
        logger.info(f"Extracted {len(experiences)} experiences")
        
        skills = extract_skills_nlp(resume_text)
        logger.info(f"Extracted {len(skills)} skills")
        
        education = extract_education_from_resume(resume_text)
        logger.info(f"Extracted education: {education}")
        
        return resume_text, experiences, list(skills), education
    
    except Exception as e:
        logger.error(f"Error parsing resume: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to extract text from resume: {str(e)}")

def evaluate_candidate(resume_text: str, job_description: str, role: str, required_years: int, skills: List[str], weights: List[float]) -> Dict[str, Any]:
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
        logger.info("Starting skill evaluation")
        skill_evaluation = extract_skills_contextually(resume_text, job_description, role, skills, weights)
        if skill_evaluation is None:
            logger.error("extract_skills_contextually returned None")
            skill_score, matched_skills, missing_skills = 0.0, [], []
        else:
            skill_score, matched_skills, missing_skills = skill_evaluation
        logger.info(f"Skill evaluation complete. Score: {skill_score}, Matched: {matched_skills}, Missing: {missing_skills}")

        # Experience evaluation
        logger.info("Starting experience evaluation")
        experiences = extract_experience(resume_text, role)
        experience_score = calculate_experience_score(experiences, required_years)
        relevant_experience_score = calculate_relevant_experience_score(experiences, job_description, role)
        logger.info(f"Experience evaluation complete. Score: {experience_score}, Relevant Score: {relevant_experience_score}")

        # Education evaluation
        logger.info("Starting education evaluation")
        education_score, education_level = calculate_education_score(resume_text)
        logger.info(f"Education evaluation complete. Score: {education_score}, Level: {education_level}")

        # Project complexity evaluation
        project_complexity_score = evaluate_project_complexity(resume_text, job_description, role)
        logger.info(f"Project complexity score: {project_complexity_score}")

        # Calculate overall fitment
        overall_fitment = calculate_robust_fitment(
            experience_score, skill_score * 100, education_score, project_complexity_score * 100, role
        )

        strengths = []
        weaknesses = []

        # Determine strengths and weaknesses based on scores
        if skill_score >= 0.6:
            strengths.append(f"Candidate has a good skill match ({skill_score:.2f}) for the job requirements.")
        else:
            weaknesses.append(f"Candidate's skills ({skill_score:.2f}) could be improved to better match the job requirements.")

        if experience_score >= 65:
            strengths.append(f"Candidate has significant relevant experience ({experience_score:.2f}%).")
        else:
            weaknesses.append(f"Candidate's relevant experience ({experience_score:.2f}%) could be stronger for this role.")

        if education_score >= 60:
            strengths.append(f"Candidate has a strong educational background ({education_score:.2f}%).")
        else:
            weaknesses.append(f"Candidate's educational background ({education_score:.2f}%) might need improvement for this role.")

        if project_complexity_score >= 0.6:
            strengths.append(f"Candidate has handled complex projects effectively ({project_complexity_score:.2f}).")
        else:
            weaknesses.append(f"Candidate's experience with complex projects ({project_complexity_score:.2f}) could be improved.")

        final_eval = {
            "overall_fitment": overall_fitment,
            "skill_match": skill_score * 100,
            "experience_score": experience_score,
            "relevant_experience_score": relevant_experience_score,
            "education_score": education_score,
            "education_level": education_level,
            "project_complexity_score": project_complexity_score * 100,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "experiences": experiences,
            "target_skills": skills,
            "skill_weights": weights,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "interview_recommendation": ""
        }

        # Adjust interview recommendation based on overall fitment
        if final_eval["overall_fitment"] >= 75:
            final_eval["interview_recommendation"] = "Strongly recommend for interview"
        elif final_eval["overall_fitment"] >= 60:
            final_eval["interview_recommendation"] = "Recommend for interview"
        elif final_eval["overall_fitment"] >= 50:
            final_eval["interview_recommendation"] = "Consider for interview with reservations"
        else:
            final_eval["interview_recommendation"] = "May not be suitable for this role"

        return final_eval

    except Exception as e:
        logger.error(f"Error in evaluate_candidate: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def calculate_weights(skill_rankings: List[int]) -> List[float]:
    if not skill_rankings:
        return []

    total = sum(skill_rankings)
    if total == 0:
        return [1.0 / len(skill_rankings)] * len(skill_rankings)

    weights = [rank / total for rank in skill_rankings]

    softening_factor = 0.5
    softened_weights = [
        w * softening_factor + (1 - softening_factor) / len(weights)
        for w in weights
    ]

    normalized_weights = [w / sum(softened_weights) for w in softened_weights]

    return normalized_weights

def get_recommendation(overall_score):
    if overall_score >= 75:
        return "This candidate is a strong fit and should be considered for an interview."
    elif overall_score >= 60:
        return "This candidate shows potential and may be worth considering for an interview, with additional screening."
    elif overall_score >= 50:
        return "This candidate has some relevant skills but may not be a strong fit. Consider only if the candidate pool is limited."
    else:
        return "This candidate does not appear to be a good fit for the job based on the initial assessment."

def display_results(evaluation_result):
    st.subheader("Evaluation Report")

    overall_score = evaluation_result["overall_fitment"]
    st.write(f"Overall Fitment Score: {overall_score:.2f}%")

    recommendation = get_recommendation(overall_score)
    st.markdown(f"**Interview Recommendation:** {recommendation}")

    st.subheader("Key Skills Analysis")
    st.write(f"Skills Relevance Score: {evaluation_result['skill_match']:.2f}% (Target: 70%+)")
    st.write("Top Matched Skills/Phrases:")
    for skill in evaluation_result['matched_skills'][:5]:  # Display top 5
        st.write(f"- {skill}")
    st.write("Top Missing Skills/Phrases:")
    for skill in evaluation_result['missing_skills'][:5]:  # Display top 5
        st.write(f"- {skill}")

    st.subheader("Experience Analysis")
    st.write(f"Experience Score: {evaluation_result['experience_score']:.2f}% (Target: 65%+)")
    st.write(f"Relevant Experience Score: {evaluation_result['relevant_experience_score']:.2f}%")
    st.write("Relevant Experiences:")
    for exp in evaluation_result['experiences']:
        st.write(f"- {exp['title']} at {exp['company']} ({exp['start_date']} to {exp['end_date']})")

    st.subheader("Education Analysis")
    st.write(f"Education Score: {evaluation_result['education_score']:.2f}% (Target: 60%+)")
    st.write(f"Highest Education Level: {evaluation_result['education_level']}")

    st.subheader("Project Complexity Analysis")
    st.write(f"Project Complexity Score: {evaluation_result['project_complexity_score']:.2f}% (Target: 60%+)")

    st.subheader("Strengths and Areas for Improvement")
    st.markdown("### Strengths")
    for strength in evaluation_result["strengths"]:
        st.markdown(f"- {strength}")

    st.markdown("### Areas for Improvement")
    for weakness in evaluation_result["weaknesses"]:
        st.markdown(f"- {weakness}")

    with st.expander("See full evaluation details"):
        st.json(evaluation_result)

def main_app():
    print("Starting main_app function")  # Direct console output
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

    with st.form(key='resume_form'):
        job_description = st.text_area("Paste the Job Description here. Make sure to include key aspects of the role required.", placeholder="Job description. This field should have at least 100 characters.")
        resume_file = st.file_uploader("Upload your resume", type=['pdf', 'docx'])
        role_options = list(JOB_ROLES.keys()) + ["Other"]
        formatted_roles = [format_role(role) for role in role_options]
        selected_role = st.selectbox("Select the job role", options=formatted_roles, key='selected_role')
        
        # Always show the custom role input but instruct users to fill it only if 'Other' is selected
        st.caption("If 'Other' is selected above, please specify the custom job title below:")
        custom_role = st.text_input("Custom Job Title (required if 'Other' is selected)", key='custom_role')
        
        st.write("Enter the key parameters to evaluate the resume:")
        skills = []
        skill_rankings = []
        for i in range(5):
            col1, col2 = st.columns(2)
            with col1:
                skill = st.text_input(f"Skill {i+1}", placeholder=f"Skill {i+1}", key=f"skill_{i}")
                skills.append(skill)
            with col2:
                rank = st.number_input(f"Rank for Skill {i+1}", min_value=1, max_value=5, value=i+1, key=f"skill_rank_{i}")
                skill_rankings.append(rank)

        min_experience = st.number_input("Minimum years of experience", min_value=0, value=2)

        submitted = st.form_submit_button('Submit')

    # Logic to update the role based on the selection
    if submitted:
        if selected_role == "Other":
            if custom_role.strip():  # Ensure custom_role is not empty if 'Other' is selected
                st.session_state.role = custom_role
            else:
                st.error("Please enter a custom job title when selecting 'Other'.")
                st.stop()  # Stop execution if 'Other' is selected but no custom role is entered
        else:
            st.session_state.role = reverse_format_role(selected_role)

    print(f"Form submitted: {submitted}")  # Direct console output
    print(f"Resume file: {resume_file.name if resume_file else 'None'}")  # Direct console output
    print(f"Job description length: {len(job_description)}")  # Direct console output
    print(f"Selected role: {st.session_state.role}")  # Direct console output

    if submitted and resume_file and len(job_description) > 100:
        print("Conditions met, processing resume...")
        try:
            print(f"Resume file: {resume_file.name}, type: {resume_file.type}")  # Direct console output
            resume_text, experiences, extracted_skills, education = parse_resume(resume_file, st.session_state.role)
        
            if resume_text is None:
                raise ValueError("Failed to extract text from resume")

            print(f"Extracted resume text length: {resume_text[:500]}...")
        
            # Calculate weights for skills
            weights = calculate_weights(skill_rankings)

            # Filter out empty skills
            skills = [skill for skill in skills if skill.strip()]
            weights = weights[:len(skills)]

            evaluation_result = evaluate_candidate(resume_text, job_description, st.session_state.role, min_experience, skills, weights)
            
            display_results(evaluation_result)

            # Feedback form
            with st.form(key='feedback_form'):
                st.subheader("Feedback")
                name = st.text_input("Name of Person Leaving Feedback")
                role_input = st.text_input("Candidate or Resume Name", value=extract_first_name(resume_text))
                client = st.text_input("Client")
                accuracy_rating = st.select_slider("Accuracy of the evaluation:", options=[1, 2, 3, 4, 5])
                content_rating = st.select_slider("Quality of the report content:", options=[1, 2, 3, 4, 5])
                suggestions = st.text_area("Please provide any suggestions for improvement:")
                submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                submit_feedback = st.form_submit_button("Submit Feedback")

                if submit_feedback:
                    feedback_entry = {
                        "name": name,
                        "role_input": role_input,
                        "client": client,
                        "accuracy_rating": accuracy_rating,
                        "content_rating": content_rating,
                        "suggestions": suggestions,
                        "submitted_at": submitted_at
                    }
                    feedback_data.append(feedback_entry)
                    save_feedback_data(FEEDBACK_FILE, feedback_data)
                    st.success("Thank you for your feedback!")
                    st.session_state["feedback_submitted"] = True

        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")
            st.error(f"An error occurred while processing the resume: {str(e)}")
            logger.error(traceback.format_exc())
    
    elif submitted:
        st.error("Please upload a resume and ensure the job description has at least 100 characters.")

if __name__ == "__main__":
    print("Starting the application")  # Direct console output

    # Initialize the 'logged_in' state if it doesn't exist
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # Initialize the 'feedback_submitted' state if it doesn't exist
    if "feedback_submitted" not in st.session_state:
        st.session_state["feedback_submitted"] = False

    # Check login status and decide which page to show
    if not st.session_state["logged_in"]:
        login()
    else:
        main_app()

    print("Application execution completed")  # Direct console output
