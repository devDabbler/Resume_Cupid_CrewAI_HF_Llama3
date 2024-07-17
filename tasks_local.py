from datetime import datetime
from crewai import Task
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_calibration_task(job_description, resume, resume_calibrator, role, parameters):
    try:
        calibration_task = Task(
            name="Calibrate Resume",
            description=f"""
            Evaluate the fitment of the provided resume against the job requirements for the role of {role}.
            Provide a detailed evaluation report with the following structure:

            Evaluation Report
            Experience Fitment Score: [Score]%

            [Overall assessment of the candidate's experience]

            • Relevant Experience: [List relevant experience or state 'None' if not applicable]
            • Irrelevant Experience: [List irrelevant experience and explain why]
            • Gaps: [Identify gaps in the candidate's experience]
            • Areas of Improvement: [Suggest areas where the candidate should focus on improving]

            [Concluding statement about the candidate's fitment for the role]

            Use the given parameters and job description to inform your evaluation.
            Parameters: {parameters}
            Job Description: {job_description}
            Resume: {resume}
            """,
            agent=resume_calibrator,
            expected_output="A detailed calibration report with scores and justifications."
        )
        logging.info(f"Created calibration task for role: {role}")
        return calibration_task
    except Exception as e:
        logging.error(f"Failed to create calibration task: {str(e)}")
        raise

def create_skill_evaluation_task(job_description, resume_skills, skills_agent, role, required_skills, weights):
    try:
        skill_evaluation_task = Task(
            name="Evaluate Skills",
            description=f"""
            Evaluate the candidate's skills against the required skills for the role of {role}.
            Provide a detailed evaluation report with the following structure:

            Skill Evaluation Report
            Skills Fitment Score: [Score]%

            [Overall assessment of the candidate's skills]

            Skill-by-Skill Analysis:
            {', '.join([f"• {skill}: [Score]% - [Brief justification]" for skill in required_skills])}

            Strengths: [List the candidate's strongest skills]
            Areas for Improvement: [List skills the candidate needs to develop]

            [Concluding statement about the candidate's skill fitment for the role]

            Use the following information to inform your evaluation:
            Job Requirements: {job_description}
            Resume Skills: {resume_skills}
            Required Skills: {', '.join(required_skills)}
            Skill Importance Weights: {weights}

            Scoring Guidelines:
            - expert: 10 points
            - advanced: 8 points
            - intermediate: 5 points
            - beginner: 2 points
            - no experience: 0 points
            """,
            agent=skills_agent,
            expected_output="A detailed skills evaluation report with scores and justifications for each skill."
        )
        logging.info(f"Created skills evaluation task for role: {role}")
        return skill_evaluation_task
    except Exception as e:
        logging.error(f"Failed to create skills evaluation task: {str(e)}")
        raise

def create_experience_evaluation_task(job_description, resume_experience, experience_agent, role):
    try:
        experience_evaluation_task = Task(
            name="Evaluate Experience",
            description=f"""
            Evaluate the candidate's work experience and history based on the job requirements for the role of {role}.
            Provide a detailed evaluation report with the following structure:

            Experience Evaluation Report
            Experience Fitment Score: [Score]%

            [Overall assessment of the candidate's experience]

            • Relevant Experience: [List relevant experience or state 'None' if not applicable]
            • Irrelevant Experience: [List irrelevant experience and explain why]
            • Gaps: [Identify gaps in the candidate's experience]
            • Areas of Improvement: [Suggest areas where the candidate should focus on improving]

            [Concluding statement about the candidate's experience fitment for the role]

            Use the following information to inform your evaluation:
            Job Requirements: {job_description}
            Resume Experience: {resume_experience}

            Scoring Guidelines:
            - Assess the relevance and depth of work experience.
            - Consider the alignment of past roles with the current job requirements.
            - Evaluate the progression and growth in the candidate's career.
            """,
            agent=experience_agent,
            expected_output="A detailed experience evaluation report with scores, relevant experiences, and improvement areas."
        )
        logging.info(f"Created experience evaluation task for role: {role}")
        return experience_evaluation_task
    except Exception as e:
        logging.error(f"Failed to create experience evaluation task: {str(e)}")
        raise

def log_run(input_data, output_data):
    """
    Logs the input and output data of a run.
    
    Args:
        input_data (dict): The input data for the run.
        output_data (dict): The output data from the run.
    """
    try:
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_data": input_data,
            "output_data": output_data
        }
        with open("run_log.json", "a") as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")
        logging.info("Run logged successfully.")
    except Exception as e:
        logging.error(f"Failed to log run: {str(e)}")

# Add this line at the end of the file to log when the module is imported
logging.info("tasks_local.py module loaded")