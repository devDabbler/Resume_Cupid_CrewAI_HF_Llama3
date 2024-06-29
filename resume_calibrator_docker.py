import os
import tempfile
import streamlit as st
import re
import logging
import fitz
from pdfminer.high_level import extract_text as pdfminer_extract_text
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
import json
import torch
import yaml
from tasks import classify_job_title, create_calibration_task, create_skill_evaluation_task, create_experience_evaluation_task, log_run
from utils import extract_experience_section, extract_skills_section
import streamlit_authenticator as stauth
from langchain_groq import ChatGroq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import platform
import transformers
from crewai import Crew, Task, Process
from agents_module import create_resume_calibrator_agent, create_skills_agent, create_experience_agent

# Set up logging
logging.basicConfig(level=logging.INFO, filename='resume_calibrator.log')

# Log environment details
logging.info(f"Python Version: {platform.python_version()}")
logging.info(f"PyTorch Version: {torch.__version__}")
logging.info(f"Transformers Version: {transformers.__version__}")

# Streamlit UI setup
st.set_page_config(page_title='📝 Resume Cupid', page_icon="📝")

# Load environment variables
load_dotenv(find_dotenv())

# Load the configuration file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

# Create an instance of the Authenticate class
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

# Render the login form
name, authentication_status, username = authenticator.login("main")

# Initialize the LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)

FEEDBACK_FILE = r"/app/feedback_data.json"

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

def save_feedback_data(feedback_data):
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "w") as file:
        json.dump(feedback_data, file)

def extract_first_name(resume_text):
    match = re.match(r"(\w+)", resume_text)
    if match:
        return match.group(1)
    return "Unknown"

def calculate_weights(rankings):
    total_ranks = sum(rankings)
    weights = [rank / total_ranks for rank in rankings]
    return weights

def calculate_fitment_score(individual_scores, weights):
    total_score = sum(score * weight for score, weight in zip(individual_scores, weights))
    max_score = sum(weights)
    fitment_percentage = total_score * 100 / max_score
    return fitment_percentage

def display_results(result, job_description=""):
    st.subheader("Evaluation Report")

    if result.startswith("Error:"):
        st.error(result)
        return

    # Display the entire result first
    st.write(result)
    
    # Try to extract and display an overall fitment score
    fitment_score_match = re.search(r"(?:Fitment Score|Experience Fitment Score):\s*(\d+)%", result, re.IGNORECASE)
    if fitment_score_match:
        fitment_score = int(fitment_score_match.group(1))
        st.write(f"Overall Fitment Score: {fitment_score}%")
    else:
        st.write("Overall Fitment Score not explicitly stated in the report.")

    # Try to extract skill-based evaluation
    st.subheader("Skill-based Evaluation:")
    skills_evaluation = re.findall(r"(\w+):\s*(.+?)\s*Score:\s*(\d+)%\s*Justification:\s*(.+?)(?=\n\w+:|$)", result, re.DOTALL)
    if skills_evaluation:
        for skill, description, score, justification in skills_evaluation:
            st.write(f"**{skill}**: {score}%")
            st.write(f"Description: {description.strip()}")
            st.write(f"Justification: {justification.strip()}")
            st.write("")
    else:
        st.write("No detailed skill-based evaluation found in the report.")

    # Try to extract general job description fitment
    st.subheader("General Job Description Fitment:")
    general_fitment = re.search(r"Overall:(.+?)(?=Areas for Improvement:|$)", result, re.DOTALL)
    if general_fitment:
        st.write(general_fitment.group(1).strip())
    else:
        st.write("No general job description fitment assessment found in the report.")

    # Try to extract areas for improvement
    st.subheader("Areas for Improvement:")
    improvements = re.search(r"Areas for Improvement:(.+?)(?=Note:|$)", result, re.DOTALL)
    if improvements:
        improvement_points = improvements.group(1).strip().split('\n')
        for point in improvement_points:
            st.write(f"- {point.strip('* ')}")
    else:
        st.write("No specific areas for improvement found in the report.")

def analyze_skills(skills_section, job_description):
    required_skills = [skill.strip().lower() for skill in job_description.split(',')]
    candidate_skills = [skill.strip().lower() for skill in skills_section.split(',')]

    matched_skills = {}
    unmatched_skills = []
    for req_skill in required_skills:
        if req_skill in candidate_skills:
            matched_skills[req_skill] = 1.0
        else:
            best_match = max(candidate_skills, key=lambda x: calculate_semantic_similarity(x, req_skill))
            similarity = calculate_semantic_similarity(best_match, req_skill)
            if similarity > 0.6:  # Adjust threshold as needed
                matched_skills[req_skill] = similarity
            else:
                unmatched_skills.append(req_skill)

    return matched_skills, unmatched_skills

def analyze_experience(experience_section, job_description):
    relevant_experience = {}

    section_pattern = re.compile(r'^(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)$', re.MULTILINE)
    bullet_point_pattern = re.compile(r'^\s*•\s*(.+?)$', re.MULTILINE)

    sections = section_pattern.findall(experience_section)

    for title, company, duration in sections:
        section_start = experience_section.find(title)
        section_end = experience_section.find('\n\n', section_start)
        if section_end == -1:
            section_end = len(experience_section) 

        section_text = experience_section[section_start:section_end]
        bullet_points = bullet_point_pattern.findall(section_text)

        for bullet_point in bullet_points:
            similarity = calculate_semantic_similarity(bullet_point, job_description)
            if similarity > 0.5:  # Adjust threshold as needed
                relevant_experience[bullet_point] = similarity

    return relevant_experience

def calculate_semantic_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

def read_all_pdf_pages(pdf_path):
    text = ''
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        if text.strip():
            return text
    except Exception as e:
        logging.error(f"PyMuPDF extraction failed: {e}")

    try:
        text = pdfminer_extract_text(pdf_path)
        if text.strip():
            return text
    except Exception as e:
        logging.error(f"PDFMiner extraction failed: {e}")

    return ""

skills_keywords = ["skills", "technical skills", "professional skills", "key skills", "core competencies", "technical proficiencies", "technical competencies", "skills summary", "skills & competencies", "skills and competencies", "skills & proficiencies", "skills and proficiencies", "skills & strengths", "skills and strengths", "skills & abilities", "skills and abilities", "skills & qualifications", "skills and qualifications", "skills & experience", "skills and experience", "skills & knowledge", "skills and knowledge", "skills & expertise", "skills and expertise"]

def extract_resume_sections(resume):
    resume_skills = extract_skills_section(resume, skills_keywords)
    resume_experience = extract_experience_section(resume)
    return resume_skills, resume_experience

# Check the authentication status
if authentication_status == False:
    st.error("Username/password is incorrect")
elif authentication_status == None:
    st.warning("Please enter your username and password")
elif authentication_status:
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

if submitted and resume_file is not None and len(job_description) > 100:
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(resume_file.read())
            resume_file_path = tmp_file.name
        
        resume = read_all_pdf_pages(resume_file_path)
        os.unlink(resume_file_path)
        
        resume_first_name = extract_first_name(resume)

        resume_skills, resume_experience = extract_resume_sections(resume)

        resume_calibrator = create_resume_calibrator_agent(llm)
        skills_agent = create_skills_agent(llm)
        experience_agent = create_experience_agent(llm)

        parameters = [skill1, skill2, skill3, skill4, skill5, f"{min_experience} or more years of experience"]
        weights = calculate_weights(skill_rankings)
        
        calibration_task = create_calibration_task(job_description, resume, resume_calibrator, role, parameters)
        skill_evaluation_task = create_skill_evaluation_task(job_description, resume_skills, skills_agent, role, parameters, weights)
        experience_evaluation_task = create_experience_evaluation_task(job_description, resume_experience, experience_agent, role)

        crew = Crew(
            agents=[resume_calibrator, skills_agent, experience_agent],
            tasks=[calibration_task, skill_evaluation_task, experience_evaluation_task],
            verbose=True
        )
        
        result = crew.kickoff()
        
        # Process the results
        processed_result = str(result)
        
        # Display results
        display_results(processed_result, job_description)

        # Log the run
        input_data = {
            "job_description": job_description,
            "resume": resume,
            "role": role,
            "parameters": parameters,
            "weights": weights
        }
        output_data = {"result": processed_result}
        log_run(input_data, output_data)

    except Exception as e:
        st.error(f"Failed to process the request: {str(e)}")
        logging.error(f"Failed to process the request: {str(e)}")
        logging.exception(e)
else:
    st.write("Awaiting input and file upload...")

    # Adding a separate feedback form outside the main resume form
    st.subheader("Feedback")
    with st.form(key='feedback_form'):
        name = st.text_input("Name of Person Leaving Feedback")
        resume_first_name = st.text_input("Candidate or Resume Name", value=resume_first_name) # type: ignore
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