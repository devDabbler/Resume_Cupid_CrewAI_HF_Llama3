import PyPDF2
import re
from transformers import pipeline
import numpy as np
import logging
from scipy import stats
from fuzzywuzzy import fuzz
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the Hugging Face NER pipeline
skills_extractor = pipeline("ner", model="dslim/bert-base-NER")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

skills_keywords = ["Skills", "Technical Skills", "Programming Languages", "Tools", "Technologies"]

def extract_skills_with_huggingface(resume_text):
    # Extract skills using the Hugging Face NER pipeline
    entities = skills_extractor(resume_text)
    skills = [entity['word'] for entity in entities if entity['entity'] in ['B-SKILL', 'I-SKILL']]
    return list(set(skills))  # Remove duplicates

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

def analyze_skill_relevance(skill, job_description):
    skill_doc = nlp(skill.lower())
    job_doc = nlp(job_description.lower())
    
    skill_vector = skill_doc.vector
    job_vector = job_doc.vector
    
    similarity = skill_vector.dot(job_vector) / (skill_vector.norm() * job_vector.norm())
    return max(0, min(1, similarity))  # Ensure the result is between 0 and 1

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
    
    return relevant_experience, position_titles, experience_depth, industry_relevance

def calculate_experience_relevance(experience_text, job_description):
    exp_doc = nlp(experience_text.lower())
    job_doc = nlp(job_description.lower())
    
    exp_vector = exp_doc.vector
    job_vector = job_doc.vector
    
    similarity = exp_vector.dot(job_vector) / (exp_vector.norm() * job_vector.norm())
    return max(0, min(1, similarity))  # Ensure the result is between 0 and 1

def calculate_experience_depth(relevant_experience):
    total_years = 0
    for exp in relevant_experience:
        years = extract_years_from_duration(exp['duration'])
        total_years += years * exp['relevance_score']
    return min(10, total_years)  # Cap at 10 years

def extract_years_from_duration(duration):
    # This is a simple implementation. You might want to make it more robust.
    years = re.findall(r'(\d+)\s*years?', duration.lower())
    return int(years[0]) if years else 0

def calculate_industry_relevance(relevant_experience, job_description):
    industry_keywords = extract_industry_keywords(job_description)
    relevance_scores = []
    
    for exp in relevant_experience:
        exp_text = f"{exp['title']} {exp['company']} {exp['description']}"
        relevance = sum(keyword in exp_text.lower() for keyword in industry_keywords)
        relevance_scores.append(relevance)
    
    return min(10, max(relevance_scores)) if relevance_scores else 0

def extract_industry_keywords(job_description):
    # This is a simple keyword extraction. You might want to use more advanced NLP techniques.
    doc = nlp(job_description.lower())
    return [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop]

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
    weights = {
        'skills': 0.4,
        'experience': 0.4,
        'education': 0.1,
        'industry_relevance': 0.1
    }
    
    weighted_scores = [score * weights[category] for category, score in scores.items()]
    return sum(weighted_scores)

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