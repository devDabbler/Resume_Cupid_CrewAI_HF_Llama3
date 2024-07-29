import os
import json
import logging
import asyncio
import traceback
from typing import Dict, Any
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from crewai import Crew

from agents_module import create_resume_calibrator_agent, create_skills_agent, create_experience_agent
from tasks import create_calibration_task, create_skill_evaluation_task, create_experience_evaluation_task
from utils import (
    parse_resume, extract_first_name, load_feedback_data, save_feedback_data,
    evaluate_resume, evaluate_education, evaluate_skills, evaluate_experience,
    evaluate_projects, evaluate_soft_skills, calculate_weights
)

# Load environment variables
load_dotenv(find_dotenv())

# Initialize logging
logging.basicConfig(level=logging.INFO, filename='resume_calibrator.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit UI setup
st.set_page_config(page_title='📝 Resume Cupid', page_icon="📝", layout="wide")

# Environment variables
FEEDBACK_FILE = os.getenv("FEEDBACK_FILE")
LOGIN_USERNAME = os.getenv('LOGIN_USERNAME')
LOGIN_PASSWORD = os.getenv('LOGIN_PASSWORD')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Validate environment variables
if not all([FEEDBACK_FILE, LOGIN_USERNAME, LOGIN_PASSWORD, GROQ_API_KEY]):
    st.error("Missing required environment variables. Please check your .env file.")
    st.stop()

# Load feedback data
feedback_data = load_feedback_data(FEEDBACK_FILE)

@st.cache_resource
def initialize_llm():
    return ChatGroq(model="llama3-8b-8192", temperature=0.1, api_key=GROQ_API_KEY)

def login_page():
    st.markdown("""
        <style>
        /* Your CSS styles here */
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; font-size: 48px; margin-bottom: 0;'>💘</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; margin-top: 0;'>Welcome to Resume Cupid</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Login to access the resume evaluation tool</h3>", unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
                st.session_state.logged_in = True
                st.success("Login successful! Redirecting...")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
    
    st.markdown("<div class='footer-text'>Don't have an account? <a href='mailto:hello@resumecupid.ai'>Contact us</a> to get started!</div>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def parse_resume_cached(file):
    return parse_resume(file)

def optimize_prompt(prompt: str, max_length: int = 2000) -> str:
    """
    Optimize the prompt by truncating it if it exceeds the maximum length.
    """
    if len(prompt) <= max_length:
        return prompt
    
    # Truncate the prompt and add an ellipsis
    return prompt[:max_length-3] + "..."

async def ai_model_call(llm, prompt: str):
    optimized_prompt = optimize_prompt(prompt)
    response = await asyncio.to_thread(llm, optimized_prompt)
    return response

async def process_resume_async(job_description, resume_file, role, user_skills, min_experience, skill_rankings):
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress, status):
        progress_bar.progress(progress)
        status_text.text(status)

    try:
        update_progress(0.1, "Initializing...")
        llm = initialize_llm()

        update_progress(0.2, "Parsing resume...")
        resume_text = parse_resume_cached(resume_file)
        resume_first_name = extract_first_name(resume_text)

        update_progress(0.3, "Creating agents...")
        resume_calibrator = create_resume_calibrator_agent(llm)
        skills_agent = create_skills_agent(llm)
        experience_agent = create_experience_agent(llm)

        update_progress(0.4, "Preparing evaluation parameters...")
        parameters = user_skills + [f"{min_experience} or more years of experience"]
        weights = [rank / sum(skill_rankings) for rank in skill_rankings]

        update_progress(0.5, "Creating tasks...")
        calibration_task = create_calibration_task(job_description, resume_text, resume_calibrator, role, parameters)
        skill_evaluation_task = create_skill_evaluation_task(job_description, resume_text, skills_agent, role, weights, user_skills)
        experience_evaluation_task = create_experience_evaluation_task(job_description, resume_text, experience_agent, role)

        update_progress(0.6, "Starting evaluation...")
        crew = Crew(
            agents=[resume_calibrator, skills_agent, experience_agent],
            tasks=[calibration_task, skill_evaluation_task, experience_evaluation_task]
        )

        update_progress(0.7, "Processing results...")
        result = await asyncio.to_thread(crew.kickoff)

        update_progress(0.8, "Finalizing evaluation...")
        try:
            processed_result = json.loads(result) if isinstance(result, str) else result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from crew result: {result}")
            processed_result = {"error": "Failed to process the resume. Please try again.", "raw_output": result}

        update_progress(0.9, "Calculating additional scores...")
        our_evaluations = {
            "education_score": evaluate_education(resume_text),
            "skills_score": evaluate_skills(resume_text, job_description),
            "experience_score": evaluate_experience(resume_text, job_description),
            "projects_score": evaluate_projects(resume_text, job_description),
            "soft_skills_score": evaluate_soft_skills(resume_text)
        }
        processed_result.update(our_evaluations)

        update_progress(1.0, "Evaluation complete!")
        return processed_result, resume_first_name

    except Exception as e:
        logger.error(f"Error in process_resume: {str(e)}", exc_info=True)
        update_progress(1.0, "Error occurred during evaluation")
        return {"error": f"An error occurred during the resume evaluation process: {str(e)}"}, None

def display_results(result: Dict[str, Any]):
    if "error" in result:
        st.error(result["error"])
        st.write("Raw output for debugging:")
        st.code(result.get("raw_output", "No raw output available"))
        return

    st.subheader("Resume Evaluation Summary")
    
    # Overall Fitment Score
    fitment_score = result.get('fitment_score', 0)
    st.metric("Overall Fitment Score", f"{fitment_score}%")
    
    # Recommendation
    if fitment_score >= 80:
        recommendation = "Highly Recommended for Interview"
    elif fitment_score >= 60:
        recommendation = "Recommended for Interview"
    else:
        recommendation = "Not Recommended for Interview"
    
    st.info(f"Recommendation: {recommendation}")

    # Key Strengths
    st.subheader("Key Strengths")
    strengths = [
        f"Skills: {result.get('skills_score', 0):.2f}/30",
        f"Experience: {result.get('experience_score', 0):.2f}/30",
        f"Education: {result.get('education_score', 0):.2f}/20",
    ]
    for strength in strengths:
        st.write(f"• {strength}")

    # Identified Gaps
    if 'gaps' in result and result['gaps']:
        st.subheader("Areas for Improvement")
        for gap in result['gaps']:
            st.write(f"• {gap['gap']}")
            st.write(f"  Suggestion: {gap['improvement_suggestion']}")

    # Relevant Experience
    if 'relevant_experience' in result and result['relevant_experience']:
        st.subheader("Most Relevant Experience")
        top_experiences = sorted(result['relevant_experience'], key=lambda x: x['relevance_score'], reverse=True)[:3]
        for exp in top_experiences:
            st.write(f"• {exp['experience']}")

    # Detailed Report Expander
    with st.expander("View Detailed Report"):
        st.write(result)

    # Interview Decision
    st.subheader("Interview Decision")
    decision = st.radio("Select your decision:", ["Invite for Interview", "Reject Application", "Need More Information"])
    if decision == "Need More Information":
        st.text_area("Specify additional information needed:")

    if st.button("Submit Decision"):
        st.success(f"Decision '{decision}' submitted successfully!")

async def main_app():
    st.title("Resume Cupid")
    st.markdown("Use this app to help you decide if a candidate is a good fit for a specific role.")

    with st.form(key='resume_form'):
        job_description = st.text_area("Paste the Job Description here:", placeholder="Job description (min 100 characters)")
        resume_file = st.file_uploader("Upload your resume", type=['pdf', 'docx'])
        role = st.text_input("Enter the role for evaluation:", placeholder="Enter the role here")
        user_skills = [st.text_input(f"Skill {i+1}", placeholder=f"Skill {i+1}") for i in range(5)]
        min_experience = st.number_input("Minimum years of experience", min_value=0, value=5)
        skill_rankings = [st.number_input(f"Rank for Skill {i+1}", min_value=1, max_value=5, value=i+1) for i in range(5)]
        
        submitted = st.form_submit_button('Submit')

    if submitted and resume_file and len(job_description) > 100:
        with st.spinner("Evaluating resume..."):
            try:
                result, resume_first_name = await process_resume_async(job_description, resume_file, role, user_skills, min_experience, skill_rankings)
                display_results(result)
                st.success("Evaluation Complete!")
            except Exception as e:
                logger.error(f"Error in processing: {str(e)}", exc_info=True)
                st.error(f"Error: Unable to process the resume. {str(e)}")
                st.expander("Debug Information").code(traceback.format_exc())
    else:
        st.write("Awaiting input and file upload...")

    # Feedback Form
    with st.form(key='feedback_form'):
        st.subheader("Feedback")
        name = st.text_input("Name of Person Leaving Feedback")
        resume_first_name = st.text_input("Candidate or Resume Name", value=resume_first_name if 'resume_first_name' in locals() else "")
        role_input = st.text_input("Role", value=role if 'role' in locals() else "", disabled=True)
        client = st.text_input("Client")
        accuracy_rating = st.select_slider("Accuracy of the evaluation:", options=[1, 2, 3, 4, 5])
        content_rating = st.select_slider("Quality of the report content:", options=[1, 2, 3, 4, 5])
        suggestions = st.text_area("Please provide any suggestions for improvement:")
        submit_feedback = st.form_submit_button("Submit Feedback")

        if submit_feedback:
            feedback_entry = {
                "resume_id": resume_first_name,
                "job_role_id": role_input,
                "name": name,
                "accuracy_rating": accuracy_rating,
                "content_rating": content_rating,
                "suggestions": suggestions,
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "client": client
            }
            feedback_data.append(feedback_entry)
            save_feedback_data(feedback_data, FEEDBACK_FILE)
            st.success("Thank you for your feedback!")

def app():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
    else:
        asyncio.run(main_app())

if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try again later or contact support.")
        st.expander("Debug Information").code(traceback.format_exc())
    finally:
        logger.info("Application session ended")