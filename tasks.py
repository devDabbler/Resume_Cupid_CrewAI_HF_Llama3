from datetime import datetime
import re
from crewai import Task
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_calibration_task(job_description, resume, resume_calibrator, role, parameters):
    return Task(
        description=f"""
        Evaluate the fitment of the provided resume against the job requirements for the role of {role}.
        Provide a detailed evaluation report with the following structure:

        Calibration Report:
        1. Overall Assessment:
           - Provide a brief summary of the candidate's overall fitment.
           - Assign an initial calibration score (0-100).

        2. Key Requirements Analysis:
           - List each key requirement from the job description.
           - For each requirement, provide:
             a) Whether the candidate meets it (Yes/Partially/No)
             b) Brief justification for the assessment

        3. Skills Gap Analysis:
           - Identify any missing or underdeveloped skills.
           - Suggest potential areas for improvement or additional training.

        4. Experience Alignment:
           - Evaluate how well the candidate's experience aligns with the role.
           - Highlight any particularly relevant or impressive experiences.

        5. Calibration Adjustments:
           - Based on the detailed analysis, adjust the initial calibration score if necessary.
           - Explain any significant adjustments made.

        Use the given parameters and job description to inform your evaluation.
        Parameters: {parameters}
        Job Description: {job_description}
        """,
        agent=resume_calibrator
    )

def create_skill_evaluation_task(job_description, resume_skills, skills_agent, role, weights, required_skills):
    individual_skill_analysis = ' '.join([
        f'''
        - {skill} (Weight: {weight:.2f}):
          a) Proficiency level (Beginner/Intermediate/Advanced/Expert)
          b) Evidence from resume
          c) Alignment with job requirements
          d) Score (0-10)
        ''' for skill, weight in zip(required_skills, weights)
    ])

    return Task(
        description=f"""
        Evaluate the candidate's skills for the {role} role based on the job description and resume.
        Provide a detailed evaluation report with the following structure:

        Skill Evaluation Report:
        1. Overall Skill Assessment:
           - Provide a summary of the candidate's skill set.
           - Assign an overall skill score (0-100).

        2. Individual Skill Analysis:
           {individual_skill_analysis}

        3. Skill Gap Analysis:
           - Identify any missing or underdeveloped skills.
           - Suggest potential areas for improvement or additional training.

        4. Impact on Role Performance:
           - Explain how the candidate's skill set contributes to success in the role.
           - Assess potential limitations based on the candidate's current skill levels.

        5. Recommendations:
           - Suggest any additional training or experience that would benefit the candidate.

        Use the following information to inform your evaluation:
        Job Requirements: {job_description}
        Resume Skills: {resume_skills}

        Note: Be objective and avoid overstating the candidate's qualifications.
        """,
        agent=skills_agent
    )

def create_experience_evaluation_task(job_description, resume_text, experience_agent, role):
    return Task(
        description=f"""
        Evaluate the candidate's work experience and history based on the job requirements for the role of {role}.
        Provide a detailed evaluation report with the following structure:

        Experience Evaluation Report
        Experience Fitment Score: [Score as a number between 0 and 100]%

        Be very critical and precise in your assessment. Ensure that your score accurately reflects the candidate's fit for this specific role.
        Consider the following factors when determining the score:
        1. Relevance of experience to the role
        2. Years of applicable experience
        3. Demonstrated skills matching the job requirements
        4. Achievements and impact in previous roles
        5. Progression and growth in career

        Justify your score with specific examples from the resume.

        Overall Assessment: [Provide a brief overall assessment]

        Relevant Experience:
        - [List relevant experiences with brief explanations of their importance]

        Irrelevant Experience:
        - [List irrelevant experiences and explain why they don't apply to this role]

        Gaps:
        - [Identify specific gaps in the candidate's experience relative to the job requirements]

        Areas of Improvement:
        - [Suggest concrete areas where the candidate should focus on improving]

        Concluding Statement: [Provide a concluding statement about the candidate's experience fitment for the role]

        Interview Recommendation: [Provide a clear recommendation on whether to interview the candidate or not, based on their experience fitment]

        Use the following information to inform your evaluation:
        Job Requirements: {job_description}
        Resume Experience: {resume_text}

        Remember, your evaluation should be unique to this specific candidate and role. Avoid generic assessments.
        """,
        agent=experience_agent
    )
    
def log_run(input_data, output_data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
    ===== Run Log: {timestamp} =====
    Input:
    Job Description: {input_data['job_description']}
    Resume: {input_data['resume']}
    Role: {input_data['role']}
    Parameters: {input_data['parameters']}
    Weights: {input_data['weights']}

    Output:
    {output_data['fitment_score']}
    Recommendation: {output_data['recommendation']}
    Detailed Report: {output_data['detailed_report']}

    =============================
    """
    logging.info(log_entry)

# Add this line at the end of the file to log when the module is imported
logging.info("tasks.py module loaded")