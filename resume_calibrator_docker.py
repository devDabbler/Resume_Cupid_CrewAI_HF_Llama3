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
from tasks import classify_job_title
from utils import extract_experience_section, extract_skills_section
import streamlit_authenticator as stauth
from safetensors import safe_open
from langchain_groq import ChatGroq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import platform
import transformers
import onnxruntime as ort

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

# Check the authentication status
if authentication_status == False:
    st.error("Username/password is incorrect")
elif authentication_status == None:
    st.warning("Please enter your username and password")
elif authentication_status:
    st.title("Resume Cupid")
    st.markdown("Use this app to help you decide if a candidate is a good fit for a specific role.")

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
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)  # Ensure directory exists
        with open(FEEDBACK_FILE, "w") as file:
            json.dump(feedback_data, file)

    def extract_first_name(resume_text):
        match = re.match(r"(\w+)", resume_text)
        if match:
            return match.group(1)
        return "Unknown"

    # Initialize the LLM
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)

    def calculate_weights(rankings):
        total_ranks = sum(rankings)
        weights = [rank / total_ranks for rank in rankings]
        return weights

    def calculate_fitment_score(predicted_class):
        # Map the predicted class to a fitment score
        if predicted_class == 0:
            return 33.33  # Low fitment
        elif predicted_class == 1:
            return 66.67  # Medium fitment
        else:
            return 100.0  # High fitment

    def display_results(fitment_score, matched_skills, unmatched_skills, relevant_experience):
        st.subheader("Fitment Score:")
        st.write(f"{fitment_score:.2f}%")

        if fitment_score < 60:
            st.warning("The candidate's experience fitment is relatively low. Please review the detailed evaluation report for areas where the candidate's experience may not align with the job requirements.")
        elif fitment_score < 80:
            st.info("The candidate's experience fitment is moderate. Please review the detailed evaluation report to identify areas where the candidate's experience can be strengthened to better align with the job requirements.")
        else:
            st.success("The candidate's experience fitment is strong. Please review the detailed evaluation report to understand the candidate's relevant experience and skills.")

        st.subheader("Matched Skills:")
        if matched_skills:
            for skill, score in matched_skills.items():
                st.write(f"- {skill}: {score:.2f}")
        else:
            st.write("No matched skills found.")

        st.subheader("Unmatched Skills:")
        if unmatched_skills:
            for skill in unmatched_skills:
                st.write(f"- {skill}")
        else:
            st.write("All required skills are matched.")

        st.subheader("Relevant Experience:")
        if relevant_experience:
            for experience, score in relevant_experience.items():
                st.write(f"- {experience}: {score:.2f}")
        else:
            st.write("No relevant experience found.")

    def analyze_skills(skills_section, job_description):
        required_skills = [skill.strip().lower() for skill in job_description.split(',')]
        candidate_skills = [skill.strip().lower() for skill in skills_section.split(',')]

        matched_skills = {}
        for skill in candidate_skills:
            if skill in required_skills:
                matched_skills[skill] = 1.0
            else:
                for req_skill in required_skills:
                    similarity = calculate_semantic_similarity(skill, req_skill)
                    if similarity > 0.6:  # Adjust the threshold as needed
                        matched_skills[skill] = similarity
                        break

        unmatched_skills = [skill for skill in required_skills if skill not in matched_skills]

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
                if similarity > 0.5:  # Adjust the threshold as needed
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
        resume_skills = extract_skills_section(resume, skills_keywords)  # type: ignore
        resume_experience = extract_experience_section(resume)
        return resume_skills, resume_experience

    def predict_fitment(job_description, resume_text):
        logging.info(f"Job Description: {job_description}")
        logging.info(f"Resume Text: {resume_text}")
    
        predicted_class = classify_job_title(job_description, resume_text)
    
        logging.info(f"Predicted Class: {predicted_class}")
    
        return predicted_class

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

        if submitted:
            if resume_file is not None:
                # Read the resume file content
                resume_text = resume_file.read()
                # Call the classify_job_title function (modify as needed to process resume_text properly)
                predicted_class = classify_job_title(job_description, resume_text)
                st.write(f'The predicted class for the given job description and resume is: {predicted_class}')
            else:
                st.error("Please upload a resume file.")

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

                    matched_skills, unmatched_skills = analyze_skills(resume_skills, job_description)
                    relevant_experience = analyze_experience(resume_experience, job_description)

                    parameters = [skill1, skill2, skill3, skill4, skill5, f"{min_experience} or more years of experience"]
                    weights = calculate_weights(skill_rankings)

                    predicted_class = predict_fitment(job_description, resume)
                    fitment_score = calculate_fitment_score(predicted_class)

                    # Use the LLM for additional processing if needed
                    llm_response = llm.predict(fitment_score)  # Example usage

                    input_data = {
                        "job_description": job_description,
                        "resume": resume,
                        "role": role,
                        "parameters": parameters,
                        "weights": weights
                    }
                    output_data = {"result": fitment_score}

                    print("Result:", fitment_score)
                    display_results(fitment_score, matched_skills, unmatched_skills, relevant_experience)

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
