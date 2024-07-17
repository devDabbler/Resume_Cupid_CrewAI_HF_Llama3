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
             c) Relevance score (0-10) of this requirement to the role

        3. Skills Gap Analysis:
           - Identify any missing or underdeveloped skills.
           - Suggest potential areas for improvement or additional training.
           - Provide a gap score (0-10) for each identified gap.

        4. Experience Alignment:
           - Evaluate how well the candidate's experience aligns with the role.
           - Highlight any particularly relevant or impressive experiences.
           - Provide an experience alignment score (0-10).

        5. Industry Relevance:
           - Assess the candidate's industry experience relevance to the role.
           - Provide an industry relevance score (0-10).

        6. Calibration Adjustments:
           - Based on the detailed analysis, adjust the initial calibration score if necessary.
           - Explain any significant adjustments made.

        7. Overall Fitment Score:
           - Calculate the final fitment score (0-100) based on all the above factors.
           - Provide a brief explanation of how this score was determined.

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
          e) Relevance to the role (0-10)
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
           - Provide a gap score (0-10) for each identified gap.

        4. Skill Relevance Analysis:
           - For each skill, analyze its relevance to the specific role and industry.
           - Provide a relevance score (0-10) for each skill.

        5. Skill Application:
           - Evaluate how the candidate has applied these skills in their past experiences.
           - Provide examples of effective skill application from the resume.
           - Assign an application score (0-10) for each skill.

        6. Impact on Role Performance:
           - Explain how the candidate's skill set contributes to success in the role.
           - Assess potential limitations based on the candidate's current skill levels.
           - Provide an overall impact score (0-10).

        7. Recommendations:
           - Suggest any additional training or experience that would benefit the candidate.
           - Prioritize these recommendations (High/Medium/Low).

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
        - For each experience, provide a relevance score (0-10)

        Irrelevant Experience:
        - [List irrelevant experiences and explain why they don't apply to this role]

        Gaps:
        - [Identify specific gaps in the candidate's experience relative to the job requirements]
        - For each gap, provide a severity score (0-10)

        Depth of Experience:
        - Analyze the depth of experience in key areas related to the role.
        - Provide a depth score (0-10) for each key area.

        Industry Relevance:
        - Evaluate the relevance of the candidate's industry experience.
        - Provide an industry relevance score (0-10).

        Career Progression:
        - Analyze the candidate's career progression and growth.
        - Provide a progression score (0-10).

        Areas of Improvement:
        - [Suggest concrete areas where the candidate should focus on improving]
        - Prioritize these areas (High/Medium/Low)

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
    Fitment Score: {output_data['fitment_score']}
    Skill Score: {output_data['skill_score']}
    Experience Score: {output_data['experience_score']}
    Recommendation: {output_data['recommendation']}
    Detailed Report: {output_data['detailed_report']}

    =============================
    """
    logging.info(log_entry)

# Add this line at the end of the file to log when the module is imported
logging.info("tasks.py module loaded")