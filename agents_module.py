from crewai import Agent, Task
import logging
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fuzzy_match_skills(candidate_skills, required_skills, threshold=80):
    """
    Perform fuzzy matching of skills
    """
    matched_skills = []
    for req_skill in required_skills:
        matches = process.extract(req_skill, candidate_skills, limit=1, scorer=fuzz.token_sort_ratio)
        if matches and matches[0][1] >= threshold:
            matched_skills.append((req_skill, matches[0][0], matches[0][1]))
    return matched_skills

def weighted_scoring(criteria, experience_years=0, weight=1):
    """
    Scoring Guidelines with weighted scoring:
    - Expert (5 points): Extensive hands-on experience (5+ years)
    - Advanced (4 points): Strong hands-on experience (3-5 years)
    - Intermediate (3 points): Moderate hands-on experience (1-3 years)
    - Beginner (2 points): Limited hands-on experience (less than 1 year)
    - No Experience (0 points): No explicit mention of experience
    """
    score = 0
    if criteria == 'expert':
        score = 5
    elif criteria == 'advanced':
        score = 4
    elif criteria == 'intermediate':
        score = 3
    elif criteria == 'beginner':
        score = 2
    else:
        score = 0

    # Additional weight based on experience years
    if experience_years >= 10:
        score += 1
    elif experience_years >= 5:
        score += 0.5
    
    # Apply the weight factor
    score *= weight
    
    return min(score, 5 * weight)  # Ensure the score doesn't exceed the maximum value

def create_skills_agent(llm):
    class SkillsEvaluator(Agent):
        def fuzzy_match(self, candidate_skills, required_skills):
            return fuzzy_match_skills(candidate_skills, required_skills)

    skills_agent = SkillsEvaluator(
        role='Skills Evaluator',
        goal='Critically evaluate the candidate\'s technical skills against the required skills based on the provided scoring guidelines. Use fuzzy matching for skills comparison. Look for concrete examples, hands-on experience, and proven results for each skill. Provide a score for each skill and a detailed justification. Be objective and avoid overstating the candidate\'s qualifications. Do not assess soft skills.',
        backstory='An expert in assessing candidate technical skills based on their resume. Focuses on concrete examples, hands-on experience, and proven results. Follows a standardized scoring system and provides objective and critical evaluations. Uses fuzzy matching to compare skills.',
        verbose=True,
        llm=llm,
        max_iters=1,
        allow_delegation=False,
        scoring=weighted_scoring
    )
    return skills_agent

def create_resume_calibrator_agent(llm):
    class ResumeCalibrator(Agent):
        def fuzzy_match(self, candidate_skills, required_skills):
            return fuzzy_match_skills(candidate_skills, required_skills)

    resume_calibrator = ResumeCalibrator(
        role='Resume Calibrator',
        goal='Critically assess the fitment of the given resume against the job requirements. Provide a fitment percentage, detailed reasons for each score assigned, and identify areas where the candidate\'s qualifications may not meet the requirements. Be objective and avoid overstating the candidate\'s fitment. Use fuzzy matching for skills comparison.',
        backstory='An expert in evaluating resumes and determining their alignment with job requirements. Uses a consistent scoring methodology, provides a fitment percentage, and identifies areas of improvement. Provides objective and critical evaluations. Uses fuzzy matching to compare skills.',
        verbose=True,
        llm=llm,
        max_iters=1,
        allow_delegation=False
    )
    return resume_calibrator

def create_experience_agent(llm):
    class ExperienceEvaluator(Agent):
        def fuzzy_match(self, candidate_skills, required_skills):
            return fuzzy_match_skills(candidate_skills, required_skills)

    experience_agent = ExperienceEvaluator(
        role='Experience Evaluator',
        goal='Critically evaluate the candidate\'s work experience and history based on the provided scoring guidelines. Focus on relevant work experience, job titles, responsibilities, and their direct relevance to the required skills. Use fuzzy matching for skills comparison. Provide a detailed report with scores, reasons for each relevant experience, and identify any gaps or areas of improvement. Be objective and avoid overstating the candidate\'s experience.',
        backstory='An expert in assessing candidate work experience based on their resume. Emphasizes relevant work experience, clear job titles and responsibilities, and their direct relevance to the required skills. Follows a standardized scoring system and provides objective and critical evaluations. Uses fuzzy matching to compare skills.',
        verbose=True,
        llm=llm,
        max_iters=1,
        allow_delegation=False,
        scoring=weighted_scoring
    )
    return experience_agent

def log_run(task, result):
    logging.info(f'Task: {task.role}, Result: {result}')

# Add this line at the end of the file to log when the module is imported
logging.info("agents_module.py loaded")