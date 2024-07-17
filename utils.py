import PyPDF2
import re
from transformers import pipeline
import numpy as np
import logging
from scipy import stats
from fuzzywuzzy import fuzz
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the Hugging Face NER pipeline
skills_extractor = pipeline("ner", model="dslim/bert-base-NER")

skills_keywords = ["Skills", "Technical Skills", "Programming Languages", "Tools", "Technologies"]

def extract_skills_with_huggingface(resume_text):
    # Extract skills using the Hugging Face NER pipeline
    entities = skills_extractor(resume_text)
    skills = [entity['word'] for entity in entities if entity['entity'] == 'B-SKILL']
    return skills

def read_all_pdf_pages(pdf_path):
    text = ''
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
    return text.strip()

def extract_skills_section(resume, skills_keywords):
    skills_section = ""
    resume = re.sub(r'\s+', ' ', resume)  # Normalize spacing
    for keyword in skills_keywords:
        pattern = re.compile(rf"\b{keyword}\b.*?(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE)
        match = pattern.search(resume)
        if match:
            skills_section = match.group().strip()
            break

    if skills_section:
        skills = extract_skills_with_huggingface(skills_section)
        return ", ".join(skills)
    else:
        return ""

def extract_experience_section(resume):
    experience_section = ""
    experience_keywords = ["Experience", "Work Experience", "Employment History", "Professional Experience"]
    resume = re.sub(r'\s+', ' ', resume)  # Normalize spacing
    for keyword in experience_keywords:
        pattern = re.compile(rf"\b{keyword}\b.*?(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE)
        match = pattern.search(resume)
        if match:
            experience_section = match.group().strip()
            break
    return experience_section

def analyze_skills(skills_section, job_description, threshold=80):
    required_skills = job_description.lower().split(',')
    candidate_skills = skills_section.lower().split(',')
    matched_skills = []
    for req_skill in required_skills:
        best_match = max(candidate_skills, key=lambda x: fuzz.ratio(req_skill, x))
        if fuzz.ratio(req_skill, best_match) >= threshold:
            matched_skills.append((req_skill, best_match))
    return matched_skills

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

def normalize_score(score, min_score, max_score):
    """
    Normalize a score to a 0-100 scale
    """
    return 100 * (score - min_score) / (max_score - min_score)

def calculate_confidence_interval(scores, confidence=0.95):
    """
    Calculate the confidence interval for a list of scores
    """
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean-h, mean+h

def calculate_overall_fitment(scores):
    """
    Calculate the overall fitment score based on multiple evaluation scores
    """
    return np.mean(scores) * 10  # Convert 0-10 scale to percentage

def log_evaluation_step(step_name, input_data, output_data, scores):
    """
    Log detailed information about each evaluation step
    """
    logger.info(f"Evaluation Step: {step_name}")
    logger.info(f"Input: {input_data}")
    logger.info(f"Output: {output_data}")
    logger.info(f"Scores: {scores}")
    logger.info("---")

def log_inconsistency(expected, actual, threshold):
    """
    Log when an inconsistency is detected
    """
    if abs(expected - actual) > threshold:
        logger.warning(f"Inconsistency detected: Expected {expected}, Actual {actual}")
        logger.warning(f"Difference exceeds threshold of {threshold}")
        logger.warning("---")

def extract_evaluation_info(evaluation_report):
    """
    Extract key information from the evaluation report
    """
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
    return info

def evaluate_model_performance(model_predictions, human_evaluations):
    """
    Evaluate model performance against human evaluations
    """
    mse = mean_squared_error(human_evaluations, model_predictions)
    mae = mean_absolute_error(human_evaluations, model_predictions)
    r2 = r2_score(human_evaluations, model_predictions)
    correlation = np.corrcoef(model_predictions, human_evaluations)[0, 1]
    
    logger.info(f"Model Performance Evaluation:")
    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"Mean Absolute Error: {mae}")
    logger.info(f"R-squared: {r2}")
    logger.info(f"Correlation with Human Evaluations: {correlation}")
    logger.info("---")

    return mse, mae, r2, correlation

# Add this line at the end of the file to log when the module is imported
logger.info("utils.py module loaded")