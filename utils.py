import json
import os
import PyPDF2
import re
from docx import Document
from transformers import pipeline
import numpy as np
import logging
from scipy import stats
from fuzzywuzzy import fuzz, process
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import spacy
import warnings
from opentelemetry import trace

# Suppress specific warning messages
warnings.filterwarnings("ignore", message="Overriding of current TracerProvider is not allowed")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the Hugging Face NER pipeline
logger.info("Initializing the Hugging Face NER pipeline.")
skills_extractor = pipeline("ner", model="dslim/bert-base-NER")

# Load spaCy model
logger.info("Loading spaCy model.")
nlp = spacy.load("en_core_web_sm")

skills_keywords = ["Skills", "Technical Skills", "Programming Languages", "Tools", "Technologies"]

def extract_skills_with_huggingface(resume_text):
    logger.info("Extracting skills using Hugging Face NER pipeline.")
    entities = skills_extractor(resume_text)
    skills = [entity['word'] for entity in entities if entity['entity'] in ['B-SKILL', 'I-SKILL']]
    logger.info(f"Extracted skills: {skills}")
    return list(set(skills))  # Remove duplicates

def read_all_pdf_pages(pdf_path):
    logger.info(f"Reading all PDF pages from {pdf_path}.")
    text = ''
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
    return text.strip()

def parse_resume(file):
    logger.info("Parsing resume file.")
    if file.type == "application/pdf":
        logger.info("Resume file is a PDF.")
        return parse_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        logger.info("Resume file is a DOCX.")
        return parse_docx(file)
    else:
        logger.error("Unsupported file format.")
        raise ValueError("Unsupported file format")

def parse_pdf(file):
    logger.info("Parsing PDF resume.")
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    logger.info("Parsed PDF resume successfully.")
    return text

def parse_docx(file):
    logger.info("Parsing DOCX resume.")
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    logger.info("Parsed DOCX resume successfully.")
    return text

# Define the path for the feedback data file
FEEDBACK_FILE = "feedback_data.json"

def save_feedback_data(feedback_data):
    """
    Save feedback data to a JSON file.
    """
    logger.info("Saving feedback data.")
    try:
        with open(FEEDBACK_FILE, "w") as file:
            json.dump(feedback_data, file, indent=4)
        logger.info("Feedback data saved successfully.")
    except IOError as e:
        logger.error(f"Error saving feedback data to file: {str(e)}")

def load_feedback_data():
    """
    Load feedback data from a JSON file.
    """
    logger.info("Loading feedback data.")
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as file:
                content = file.read().strip()
                if content:
                    logger.info("Feedback data loaded successfully.")
                    return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from feedback data file: {str(e)}")
        except IOError as e:
            logger.error(f"Error reading feedback data file: {str(e)}")
    logger.info("No existing feedback data found. Starting with an empty list.")
    return []

def extract_first_name(resume_text):
    logger.info("Extracting first name from resume text.")
    match = re.match(r"(\w+)", resume_text)
    if match:
        return match.group(1)
    return "Unknown"

def extract_skills_section(resume_text, skills_keywords):
    logger.info("Extracting skills section from resume.")
    skills_section = ""
    resume = re.sub(r'\s+', ' ', resume_text)  # Normalize spacing
    for keyword in skills_keywords:
        pattern = re.compile(rf"\b{keyword}\b.*?(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE)
        match = pattern.search(resume)
        if match:
            skills_section = match.group().strip()
            break
    logger.info(f"Extracted skills section: {skills_section}")

    if skills_section:
        skills = extract_skills_with_huggingface(skills_section)
        logger.info(f"Extracted skills using Hugging Face: {skills}")
        return ", ".join(skills)
    else:
        logger.info("No skills section found.")
        return ""

def extract_experience_section(resume_text):
    logger.info("Extracting experience section from resume.")
    experience_section = ""
    experience_keywords = ["Experience", "Work Experience", "Employment History", "Professional Experience"]
    resume = re.sub(r'\s+', ' ', resume_text)  # Normalize spacing
    for keyword in experience_keywords:
        pattern = re.compile(rf"\b{keyword}\b.*?(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE)
        match = pattern.search(resume)
        if match:
            experience_section = match.group().strip()
            break
    logger.info(f"Extracted experience section: {experience_section}")
    return experience_section

def fuzzy_match_skills(candidate_skills, required_skills, threshold=80):
    logger.info("Performing fuzzy matching of skills.")
    matched_skills = []
    for req_skill in required_skills:
        matches = process.extract(req_skill, candidate_skills, limit=1, scorer=fuzz.token_sort_ratio)
        if matches and matches[0][1] >= threshold:
            matched_skills.append((req_skill, matches[0][0], matches[0][1]))
    logger.info(f"Matched skills: {matched_skills}")
    return matched_skills

def analyze_skills(skills_section, job_description, threshold=80):
    logger.info("Analyzing skills.")
    required_skills = job_description.lower().split(',')
    candidate_skills = skills_section.lower().split(',')
    matched_skills = []
    for req_skill in required_skills:
        best_match = max(candidate_skills, key=lambda x: fuzz.ratio(req_skill, x))
        if fuzz.ratio(req_skill, best_match) >= threshold:
            matched_skills.append((req_skill, best_match))
    logger.info(f"Analyzed matched skills: {matched_skills}")
    return matched_skills

def analyze_skill_relevance(skill, job_description):
    logger.info("Analyzing skill relevance.")
    skill_doc = nlp(skill.lower())
    job_doc = nlp(job_description.lower())
    
    skill_vector = skill_doc.vector
    job_vector = job_doc.vector
    
    similarity = skill_vector.dot(job_vector) / (skill_vector.norm() * job_vector.norm())
    logger.info(f"Skill relevance: {similarity}")
    return max(0, min(1, similarity))  # Ensure the result is between 0 and 1

def analyze_experience(experience_section, job_description):
    logger.info("Analyzing experience.")
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

        relevance_score = calculate_experience_relevance(section_text, job_description)
        if relevance_score > 0.5:  # Adjust this threshold as needed
            relevant_experience.append({
                'title': title.strip(),
                'company': company.strip(),
                'duration': duration.strip(),
                'description': section_text.strip(),
                'relevance_score': relevance_score
            })

    experience_depth = calculate_experience_depth(relevant_experience)
    industry_relevance = calculate_industry_relevance(relevant_experience, job_description)
    
    logger.info(f"Relevant experience: {relevant_experience}")
    logger.info(f"Position titles: {position_titles}")
    logger.info(f"Experience depth: {experience_depth}")
    logger.info(f"Industry relevance: {industry_relevance}")

    return relevant_experience, position_titles, experience_depth, industry_relevance

def calculate_experience_relevance(experience_text, job_description):
    logger.info("Calculating experience relevance.")
    exp_doc = nlp(experience_text.lower())
    job_doc = nlp(job_description.lower())
    
    exp_vector = exp_doc.vector
    job_vector = job_doc.vector
    
    similarity = exp_vector.dot(job_vector) / (exp_vector.norm() * job_vector.norm())
    logger.info(f"Experience relevance: {similarity}")
    return max(0, min(1, similarity))  # Ensure the result is between 0 and 1

def calculate_experience_depth(relevant_experience):
    logger.info("Calculating experience depth.")
    total_years = 0
    for exp in relevant_experience:
        years = extract_years_from_duration(exp['duration'])
        total_years += years * exp['relevance_score']
    logger.info(f"Total experience years: {total_years}")
    return min(10, total_years)  # Cap at 10 years

def extract_years_from_duration(duration):
    logger.info(f"Extracting years from duration: {duration}")
    years = re.findall(r'(\d+)\s*years?', duration.lower())
    logger.info(f"Extracted years: {years}")
    return int(years[0]) if years else 0

def calculate_industry_relevance(relevant_experience, job_description):
    logger.info("Calculating industry relevance.")
    industry_keywords = extract_industry_keywords(job_description)
    relevance_scores = []
    
    for exp in relevant_experience:
        exp_text = f"{exp['title']} {exp['company']} {exp['description']}"
        relevance = sum(keyword in exp_text.lower() for keyword in industry_keywords)
        relevance_scores.append(relevance)
    
    logger.info(f"Relevance scores: {relevance_scores}")
    return min(10, max(relevance_scores)) if relevance_scores else 0

def extract_industry_keywords(job_description):
    logger.info("Extracting industry keywords.")
    doc = nlp(job_description.lower())
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop]
    logger.info(f"Extracted industry keywords: {keywords}")
    return keywords

def normalize_score(score, min_score, max_score):
    logger.info(f"Normalizing score: {score}, min_score: {min_score}, max_score: {max_score}")
    normalized = 100 * (score - min_score) / (max_score - min_score)
    logger.info(f"Normalized score: {normalized}")
    return normalized

def calculate_confidence_interval(scores, confidence=0.95):
    logger.info("Calculating confidence interval.")
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    logger.info(f"Confidence interval: mean={mean}, lower={mean-h}, upper={mean+h}")
    return mean, mean-h, mean+h

def calculate_overall_fitment(scores):
    logger.info("Calculating overall fitment.")
    weights = {
        'skills': 0.4,
        'experience': 0.4,
        'education': 0.1,
        'industry_relevance': 0.1
    }
    
    weighted_scores = [score * weights[category] for category, score in scores.items()]
    overall_fitment = sum(weighted_scores)
    logger.info(f"Overall fitment: {overall_fitment}")
    return overall_fitment

def log_evaluation_step(step_name, input_data, output_data, scores):
    logger.info(f"Logging evaluation step: {step_name}")
    logger.info(f"Input data: {input_data}")
    logger.info(f"Output data: {output_data}")
    logger.info(f"Scores: {scores}")

def log_inconsistency(expected, actual, threshold):
    if abs(expected - actual) > threshold:
        logger.warning(f"Inconsistency detected: Expected {expected}, Actual {actual}")
        logger.warning(f"Difference exceeds threshold of {threshold}")

def extract_evaluation_info(evaluation_report):
    logger.info("Extracting evaluation info.")
    info = {}
    sections = re.split(r'Experience Evaluation Report - ', evaluation_report)[1:]
    for section in sections:
        aspect, content = section.split('\n', 1)
        info[aspect] = {}
        subsections = re.split(r'\n(?=\w+:)', content)
        for subsection in subsections:
            if ':' in subsection:
                title, details = subsection.split(':', 1)
                info[aspect][title.strip()] = details.strip()
    logger.info(f"Extracted evaluation info: {info}")
    return info

def evaluate_model_performance(model_predictions, human_evaluations):
    logger.info("Evaluating model performance.")
    mse = mean_squared_error(human_evaluations, model_predictions)
    mae = mean_absolute_error(human_evaluations)
    r2 = r2_score(human_evaluations, model_predictions)
    correlation = np.corrcoef(model_predictions, human_evaluations)[0, 1]
    
    logger.info(f"Model Performance Evaluation:")
    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"Mean Absolute Error: {mae}")
    logger.info(f"R-squared: {r2}")
    logger.info(f"Correlation with Human Evaluations: {correlation}")

    return mse, mae, r2, correlation

# Add this line at the end of the file to log when the module is imported
logger.info("utils.py module loaded")
