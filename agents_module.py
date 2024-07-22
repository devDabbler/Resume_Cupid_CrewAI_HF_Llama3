from datetime import datetime
import logging
from crewai import Agent

logger = logging.getLogger(__name__)

def create_resume_calibrator_agent(llm):
    return Agent(
        role="resume calibrator",
        goal="Critically assess the fitment of the given resume against the job requirements.",
        backstory="An expert in evaluating resumes and determining their alignment with job requirements.",
        llm=llm
    )

def create_skills_agent(llm):
    return Agent(
        role="skills evaluator",
        goal="Evaluate the candidate's skills based on the job requirements and resume.",
        backstory="An expert in assessing technical and soft skills for various roles.",
        llm=llm
    )

def create_experience_agent(llm):
    return Agent(
        role="experience evaluator",
        goal="Evaluate the candidate's work experience and history based on the job requirements.",
        backstory="An expert in assessing work experience and determining its relevance to the role.",
        llm=llm
    )

# Optional: Commenting out the detailed run logging if it's not necessary
# def log_run(input_data, output_data):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     log_entry = f"""
#     ===== Run Log: {timestamp} =====
#     Input:
#     Job Description: {input_data['job_description'][:100]}...
#     Resume: {input_data['resume'][:100]}...
#     Role: {input_data['role']}
#     Parameters: {input_data['parameters']}
#     Weights: {input_data['weights']}

#     Output:
#     Fitment Score: {output_data['fitment_score']}
#     Skill Score: {output_data['skill_score']}
#     Experience Score: {output_data['experience_score']}
#     Recommendation: {output_data['recommendation']}
#     Detailed Report: {output_data['detailed_report'][:100]}...

#     =============================
#     """
#     logger.info(log_entry)

logger.info("agents_module.py module loaded")
