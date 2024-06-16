from crewai import Task
from transformers import pipeline
import logging
from datetime import datetime
import json
import os

# Initialize Hugging Face pipelines
skills_extractor = pipeline("ner", model="dslim/bert-base-NER")
job_title_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def log_run(input_data, output_data, log_file="run_logs.json"):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "output": output_data
    }
    
    try:
        with open(log_file, "r") as file:
            logs = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)
    
    with open(log_file, "w") as file:
        json.dump(logs, file, indent=4)

def create_calibration_task(job_description, resume, resume_calibrator, role, parameters):
    """
    Creates a calibration task for evaluating the resume against the job description.
    
    Args:
        job_description (str): The job description.
        resume (str): The content of the resume.
        resume_calibrator (Agent): The agent responsible for calibrating the resume.
        role (str): The role for which the evaluation is being done.
        parameters (list): Parameters to guide the evaluation.
    
    Returns:
        Task: A CrewAI Task object configured for resume calibration.
    """
    try:
        calibration_task = Task(
            name="Calibrate Resume",
            description=(
                f"Job Requirements:\n{job_description}\n\nResume:\n{resume}\n\n"
                f"Critically evaluate the fitment of the provided resume against the job requirements "
                f"for the role of {role}. Use the given parameters and scoring guidelines:\n\n"
                f"Parameters:\n{parameters}\n\nScoring Guidelines:\n- Evaluate the candidate's "
                f"skills, experience, and qualifications against the job requirements.\n- Assign scores based on "
                f"depth of experience, skill relevance, and direct applicability to the role.\n- Identify any gaps or areas where the candidate's qualifications may not meet the requirements.\n"
                f"- Provide a fitment score as a percentage value between 0 and 100, along with detailed reasons for the score.\n"
                f"- Be objective and critical in your evaluation, and avoid overstating the candidate's fitment."
            ),
            agent=resume_calibrator,
            expected_output="Fitment score as a percentage value between 0 and 100, with detailed reasons for the score and identified gaps or areas of improvement."
        )
        
        input_data = {
            "job_description": job_description,
            "resume": resume,
            "role": role,
            "parameters": parameters
        }
        output_data = {"task": "calibration task created"}
        log_run(input_data, output_data)
        
        return calibration_task
    except Exception as e:
        logging.error(f"Failed to create calibration task: {str(e)}")
        raise

def create_skill_evaluation_task(job_description, resume_skills, skills_agent, role, required_skills, weights):
    instructions = [
        f"Look for explicit mentions of hands-on experience, proven results, and direct relevance to the skill '{skill}' and its related techniques, algorithms, frameworks, or libraries."
        for skill in required_skills
    ]
    instructions_str = "\n".join(instructions)

    skill_evaluation_task = Task(
        name="Evaluate Technical Skills",
        description=(
            f"Critically evaluate the candidate's technical skills against the required skills for the "
            f"role of {role}. Do not assess soft skills. Use the following guidelines:\n\n"
            f"Job Requirements:\n{job_description}\n\nResume Skills:\n{resume_skills}\n\n"
            f"Required Skills:\n{', '.join(required_skills)}\n\nSkill Importance Weights:\n{weights}\n\n"
            f"Instructions:\n{instructions_str}\n\n"
            f"Scoring Guidelines:\n- Expert (5 points): Extensive hands-on experience (5+ years), deep understanding, and proven track record of delivering significant results\n"
            f"- Advanced (4 points): Strong hands-on experience (3-5 years), solid understanding, and demonstrated ability to apply effectively with measurable results\n"
            f"- Intermediate (3 points): Moderate hands-on experience (1-3 years), reasonable understanding, and ability to apply with some guidance and tangible outcomes\n"
            f"- Beginner (2 points): Limited hands-on experience (less than 1 year), basic understanding, and requires substantial guidance to apply with minimal results\n"
            f"- No Experience (0 points): No explicit mention of hands-on experience, understanding, or results\n\n"
            f"Provide a score for each skill based on the candidate's demonstrated expertise level, years of experience, and proven results, "
            f"along with a detailed justification. Be critical and objective in your evaluation, and avoid overstating the candidate's qualifications."
        ),
        agent=skills_agent,
        expected_output="Scores for each skill based on the candidate's demonstrated expertise level, years of experience, and proven results, with detailed justifications."
    )
    
    input_data = {
        "job_description": job_description,
        "resume_skills": resume_skills,
        "role": role,
        "required_skills": required_skills,
        "weights": weights
    }
    output_data = {"task": "skill evaluation task created"}
    log_run(input_data, output_data)
    
    return skill_evaluation_task

def create_experience_evaluation_task(job_description, resume_experience, experience_agent, role, position_titles):
    """
    Creates an experience evaluation task for evaluating the candidate's experience against the job requirements.
    
    Args:
        job_description (str): The job description.
        resume_experience (str): The experience extracted from the resume.
        experience_agent (Agent): The agent responsible for evaluating experience.
        role (str): The role for which the evaluation is being done.
        position_titles (list): List of the candidate's position titles.
    
    Returns:
        Task: A CrewAI Task object configured for experience evaluation.
    """
    try:
        experience_evaluation_task = Task(
            name="Evaluate Experience",
            description=(
                f"Critically evaluate the candidate's work experience and history based on the following job requirements "
                f"for the role of {role}. Focus on directly relevant work experience, job titles, responsibilities, and their "
                f"applicability to the required skills. Look for explicit mentions of hands-on experience, proven results, and "
                f"direct relevance to the required skills.\n\n"
                f"Job Requirements:\n{job_description}\n\nResume Experience:\n{resume_experience}\n\n"
                f"Candidate's Position Titles:\n{', '.join(position_titles)}\n\nScoring Guidelines:\n"
                f"- Expert (5 points): Extensive hands-on experience (5+ years), proven results, and direct relevance to the required skills.\n"
                f"- Advanced (4 points): Strong hands-on experience (3-5 years), measurable results, and clear relevance to the required skills.\n"
                f"- Intermediate (3 points): Moderate hands-on experience (1-3 years), tangible outcomes, and reasonable relevance to the required skills.\n"
                f"- Beginner (2 points): Limited hands-on experience (less than 1 year), minimal results, and partial relevance to the required skills.\n"
                f"- No Experience (0 points): No explicit mention of hands-on experience, results, or relevance to the required skills.\n\n"
                f"Provide an experience fitment score as a percentage value between 0 and 100, along with a detailed justification. "
                f"Emphasize the importance of hands-on experience, proven results, and direct relevance to the job requirements. "
                f"Identify any gaps or areas where the candidate's experience may not meet the requirements. Be critical and objective "
                f"in your evaluation, and avoid overstating the candidate's experience or skills."
            ),
            agent=experience_agent,
            expected_output="Experience fitment score as a percentage value between 0 and 100, with detailed justification, identified gaps, and areas of improvement."
        )
        
        input_data = {
            "job_description": job_description,
            "resume_experience": resume_experience,
            "role": role,
            "position_titles": position_titles
        }
        output_data = {"task": "experience evaluation task created"}
        log_run(input_data, output_data)
        
        return experience_evaluation_task

    except Exception as e:
        logging.error(f"Failed to create experience evaluation task: {str(e)}")
        raise