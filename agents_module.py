import logging
from crewai import Agent

logger = logging.getLogger(__name__)

def create_agent(llm, role, goal, backstory):
    """
    Create an agent with the given parameters.
    """
    logger.info(f"Creating agent with role: {role}")
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=True  # Added for better debugging
    )

def create_resume_calibrator_agent(llm):
    logger.info("Creating resume calibrator agent")
    return create_agent(
        llm,
        role="resume calibrator",
        goal="Critically assess the fitment of the given resume against the job requirements.",
        backstory="An expert in evaluating resumes and determining their alignment with job requirements, with a keen eye for detail and understanding of various industries."
    )

def create_skills_agent(llm):
    logger.info("Creating skills evaluator agent")
    return create_agent(
        llm,
        role="skills evaluator",
        goal="Evaluate the candidate's skills based on the job requirements and resume, providing a detailed analysis of matches and gaps.",
        backstory="An expert in assessing technical and soft skills for various roles, with up-to-date knowledge of industry trends and requirements."
    )

def create_experience_agent(llm):
    logger.info("Creating experience evaluator agent")
    return create_agent(
        llm,
        role="experience evaluator",
        goal="Evaluate the candidate's work experience and history based on the job requirements, focusing on relevance and impact.",
        backstory="An expert in assessing work experience and determining its relevance to the role, with a deep understanding of career progression and industry-specific achievements."
    )

def create_education_agent(llm):
    logger.info("Creating education evaluator agent")
    return create_agent(
        llm,
        role="education evaluator",
        goal="Evaluate the candidate's educational background based on the job requirements, considering both formal education and continuous learning.",
        backstory="An expert in assessing educational qualifications and their relevance to the role, with knowledge of various educational systems and the value of ongoing professional development."
    )

def create_project_complexity_agent(llm):
    logger.info("Creating project complexity evaluator agent")
    return create_agent(
        llm,
        role="project complexity evaluator",
        goal="Assess the complexity and scale of projects the candidate has worked on, evaluating their relevance to the job requirements.",
        backstory="An expert in analyzing project descriptions and determining their complexity, scale, and relevance to various industries and roles."
    )

logger.info("agents_module.py module loaded")