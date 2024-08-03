import re
import json
import os
import logging
import traceback
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime
import fitz  # PyMuPDF
from docx import Document
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from io import BytesIO
from fuzzywuzzy import process
import yaml

# Initialize logging
logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

def load_job_roles():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, 'job_roles.yaml')
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

JOB_ROLES = load_job_roles()

def load_job_role_config(role: str) -> Dict[str, Any]:
    if role not in JOB_ROLES:
        raise ValueError(f"Unsupported job role: {role}")
    return JOB_ROLES[role]

def extract_skills_nlp(text: str) -> Set[str]:
    doc = nlp(text.lower())
    
    skills = set()
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3 and len(chunk.text) > 2:
            skills.add(chunk.text)
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG", "GPE"] and len(ent.text) > 2:
            skills.add(ent.text)
    
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) > 2:
            skills.add(token.text)
    
    return skills

def extract_skills_contextually(resume_text: str, job_description: str, role: str, target_skills: List[str], weights: List[float]) -> Tuple[float, List[str], List[str]]:
    if not resume_text or not job_description:
        return 0.0, [], target_skills

    role_config = load_job_role_config(role)
    role_specific_skills = set(role_config["skills"])

    resume_skills = extract_skills_nlp(resume_text.lower())
    job_skills = extract_skills_nlp(job_description.lower())
    
    job_skills.update(skill.lower() for skill in target_skills if skill)
    job_skills.update(role_specific_skills)
    
    matched_skills = []
    for skill in job_skills:
        match = process.extractOne(skill, resume_skills)
        if match and isinstance(match, tuple) and len(match) >= 2 and match[1] >= 80:
            matched_skills.append(match[0])

    missing_skills = job_skills - set(matched_skills)

    skill_score = sum(weight for skill, weight in zip(target_skills, weights) 
                      if process.extractOne(skill, matched_skills)[1] >= 80)
    total_weight = sum(weights) if weights else 1
    skill_score /= total_weight
    
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        cosine_sim = 0
    
    final_skill_score = 0.7 * skill_score + 0.3 * cosine_sim
    
    return final_skill_score, list(matched_skills), list(missing_skills)

def extract_education_from_resume(text):
    education = []
    education_keywords = [
        'Bsc', 'B. Pharmacy', 'B Pharmacy', 'Msc', 'M. Pharmacy', 'Ph.D', 'Bachelor', 'Master',
        'BE', 'B.E.', 'B.E', 'BS', 'B.S', 'C.A.', 'B.Com', 'M.Com', 'ME', 'M.E', 'MS', 'M.S',
        'BTECH', 'B.TECH', 'M.TECH', 'MTECH', 'PHD', 'MBA', 'graduate', 'post-graduate',
        '5 year integrated masters', 'masters', 'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII'
    ]

    pattern = r"(?i)(?:{})\s(?:\w+\s)*\w+".format("|".join(re.escape(keyword) for keyword in education_keywords))
    matches = re.findall(pattern, text)
    for match in matches:
        education.append({
            'degree': match.strip(),
            'score': calculate_degree_score(match.strip())
        })

    return education

def calculate_degree_score(degree):
    degree_scores = {
        'Ph.D': 100, 'Master': 80, 'Bachelor': 60, 'Bsc': 60, 'Msc': 80,
        'BE': 60, 'B.E.': 60, 'BS': 60, 'B.S': 60, 'ME': 80, 'M.E': 80,
        'MS': 80, 'M.S': 80, 'BTECH': 60, 'B.TECH': 60, 'M.TECH': 80,
        'MTECH': 80, 'MBA': 80, 'PHD': 100, 'graduate': 50, 'post-graduate': 70,
        'SSC': 40, 'HSC': 50, 'CBSE': 50, 'ICSE': 50, 'X': 30, 'XII': 40
    }
    return degree_scores.get(degree, 50)

def calculate_education_score(resume_text):
    education_entries = extract_education_from_resume(resume_text)
    logger.debug(f"Education entries: {education_entries}")
    if not education_entries:
        return 0, "None"

    highest_education = max(education_entries, key=lambda x: x['score'])
    return highest_education['score'], highest_education['degree']

def extract_experience(resume_text: str, role: str) -> List[Dict[str, Any]]:
    logger.info("Extracting experience from resume text")
    doc = nlp(resume_text)
    experiences = []
    
    role_config = load_job_role_config(role)
    job_titles = role_config["experience_keywords"]

    for sent in doc.sents:
        logger.debug(f"Processing sentence: {sent.text}")
        date_range = None
        job_title = None
        company = None
        
        date_range_match = re.search(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\w\s,.-]+\d{4})\s*[-–—to]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\w\s,.-]+\d{4}|Present)', sent.text)
        if date_range_match:
            date_range = date_range_match.group()
            logger.debug(f"Found date range: {date_range}")
        
        for ent in sent.ents:
            if ent.label_ == "ORG" and not company:
                company = ent.text
            elif ent.label_ == "PERSON" or any(process.extractOne(title, [ent.text.lower()])[1] >= 80 for title in job_titles):
                job_title = ent.text
        
        title_indicators = ["position of", "worked as", "role of", "titled"]
        for indicator in title_indicators:
            if indicator in sent.text.lower():
                title_start = sent.text.lower().index(indicator) + len(indicator)
                potential_title = sent.text[title_start:].split(',')[0].strip()
                if any(process.extractOne(title, [potential_title.lower()])[1] >= 80 for title in job_titles):
                    job_title = potential_title
                    break
        
        if date_range:
            start_date, end_date = parse_date_range(date_range)
            
            if start_date:
                end_date = end_date or datetime.now()
                duration = relativedelta(end_date, start_date).years + (relativedelta(end_date, start_date).months / 12)
                
                experiences.append({
                    "title": job_title or "Unknown Position",
                    "company": company or "Unknown Company",
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d") if end_date != datetime.now() else "Present",
                    "duration": duration
                })
    
    logger.info(f"Extracted experiences: {experiences}")
    return experiences

def parse_date_range(date_str):
    parts = re.split(r'\s*[-–—]\s*|\s+to\s+', date_str)
    dates = []
    for part in parts:
        try:
            dates.append(parse(part, fuzzy=True))
        except ValueError:
            year_match = re.search(r'\d{4}', part)
            if year_match:
                dates.append(parse(year_match.group(), fuzzy=True))
    
    if len(dates) == 2:
        return dates[0], dates[1]
    elif len(dates) == 1:
        return dates[0], None
    else:
        return None, None

def calculate_relevant_experience_score(experiences: List[Dict[str, Any]], job_description: str, role: str) -> float:
    job_desc_doc = nlp(job_description.lower())
    relevance_score = 0
    
    role_config = load_job_role_config(role)
    role_keywords = role_config["experience_keywords"]
    
    for exp in experiences:
        exp_text = f"{exp['title']} {exp['company']}"
        exp_doc = nlp(exp_text.lower())
        
        similarity = exp_doc.similarity(job_desc_doc)
        keyword_bonus = sum(0.1 for keyword in role_keywords if keyword.lower() in exp_text.lower())
        similarity += keyword_bonus
        
        recency_weight = 1 + (0.1 * (datetime.now().year - int(exp['start_date'][:4])))
        
        duration = exp.get('duration', 0)
        relevance_score += similarity * min(duration, 5) * recency_weight
    
    return min(relevance_score * 10, 100)

def evaluate_project_complexity(resume_text: str, job_description: str, role: str) -> float:
    role_config = load_job_role_config(role)
    project_keywords = role_config["project_keywords"]
    
    doc = nlp(resume_text.lower())
    job_doc = nlp(job_description.lower())
    
    complexity_score = sum(2 for phrase in project_keywords if phrase.lower() in resume_text.lower())
    
    for term in project_keywords:
        if term.lower() in resume_text.lower():
            surrounding_text = resume_text[max(0, resume_text.lower().index(term.lower()) - 50):min(len(resume_text), resume_text.lower().index(term.lower()) + 50)]
            if any(keyword in surrounding_text.lower() for keyword in ['implement', 'develop', 'design', 'architect']):
                complexity_score += 1
    
    project_sentences = sum(1 for sent in doc.sents if 'project' in sent.text.lower())
    
    relevance = doc.similarity(job_doc)
    
    combined_score = (0.4 * complexity_score + 0.3 * len(project_keywords) + 0.3 * project_sentences) * relevance
    
    return min(combined_score / 15, 1)

def calculate_experience_score(experiences: List[Dict[str, Any]], required_years: float) -> float:
    total_duration = sum(exp.get('duration', 0) for exp in experiences)
    score = (total_duration / required_years) * 100
    return min(score, 100)

def calculate_robust_fitment(experience_score: float, skill_score: float, education_score: float, project_complexity_score: float, role: str) -> float:
    try:
        experience_score = float(experience_score)
        skill_score = float(skill_score)
        education_score = float(education_score)
        project_complexity_score = float(project_complexity_score)
    except ValueError as e:
        raise TypeError(f"All scores must be convertible to floats. Error: {str(e)}")
    
    experience_score = min(max(experience_score, 0), 100)
    skill_score = min(max(skill_score, 0), 100)
    education_score = min(max(education_score, 0), 100)
    project_complexity_score = min(max(project_complexity_score, 0), 100)

    if role == "data_scientist":
        weights = [0.3, 0.3, 0.2, 0.2]
    elif role == "software_engineer":
        weights = [0.3, 0.35, 0.15, 0.2]
    elif role == "ui_designer":
        weights = [0.25, 0.4, 0.15, 0.2]
    elif role == "full_stack_engineer":
        weights = [0.3, 0.35, 0.15, 0.2]
    else:
        weights = [0.25, 0.25, 0.25, 0.25]

    overall_score = (
        weights[0] * experience_score +
        weights[1] * skill_score +
        weights[2] * education_score +
        weights[3] * project_complexity_score
    )

    if overall_score >= 80:
        overall_score = min(overall_score * 1.05, 100)
    elif overall_score >= 70:
        overall_score = min(overall_score * 1.03, 100)

    return round(overall_score, 2)

def extract_text_from_pdf(file_content):
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        if not text:
            logger.warning("Extracted PDF text is empty")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
def extract_text_from_docx(file_content):
    try:
        doc = Document(BytesIO(file_content))
        text = "\n".join([para.text for para in doc.paragraphs])
        if not text:
            logger.warning("Extracted DOCX text is empty")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")

def extract_first_name(resume_text: str) -> str:
    doc = nlp(resume_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.split()[0]
    return "Unknown"

def parse_resume(file, role: str):
    logger.info(f"Starting to parse resume: {file.name}")
    
    try:
        if file.type == "application/pdf":
            logger.info("Parsing PDF file")
            resume_text = extract_text_from_pdf(file.read())
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            logger.info("Parsing DOCX file")
            resume_text = extract_text_from_docx(file.read())
        else:
            logger.error(f"Unsupported file format: {file.type}")
            raise ValueError("Unsupported file format")
        
        logger.info(f"Successfully extracted text from resume. Length: {len(resume_text)}")
        logger.debug(f"Resume text preview: {resume_text[:500]}...")
        
        experiences = extract_experience(resume_text, role)
        logger.info(f"Extracted {len(experiences)} experiences")
        
        skills = extract_skills_nlp(resume_text)
        logger.info(f"Extracted {len(skills)} skills")
        
        return resume_text, experiences, list(skills)
    
    except Exception as e:
        logger.error(f"Error parsing resume: {str(e)}")
        logger.error(traceback.format_exc())
        return None, [], []

def get_recommendation(overall_score: float) -> str:
    if overall_score >= 80:
        return "Strongly recommend for interview. The candidate's qualifications closely match the job requirements."
    elif overall_score >= 70:
        return "Recommend for interview. The candidate shows strong potential for the role."
    elif overall_score >= 60:
        return "Consider for interview. The candidate meets many of the job requirements but may need additional screening."
    elif overall_score >= 50:
        return "Potentially consider for interview, but only if the candidate pool is limited. The candidate meets some job requirements but may not be the best fit."
    else:
        return "Not recommended for interview at this time. The candidate's qualifications do not closely align with the job requirements."

def calculate_skill_relevance(skills: List[str], job_description: str) -> Dict[str, float]:
    job_doc = nlp(job_description)
    return {skill: max(job_doc.similarity(nlp(skill)), 0) for skill in skills}