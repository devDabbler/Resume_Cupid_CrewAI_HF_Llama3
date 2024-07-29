import json
from crewai import Task
from typing import List, Any

def create_calibration_task(job_description: str, resume: str, resume_calibrator: Any, role: str, parameters: List[str]) -> Task:
    return Task(
        description=f"""
        Evaluate the fitment of the provided resume against the job requirements for the role of {role}.
        Use fuzzy matching to compare skills.
        Provide a detailed evaluation report in the following JSON structure:

        {{
            "fitment_score": [Score as a percentage between 0 and 100],
            "interview_recommendation": "[Clear recommendation on whether to interview the candidate]",
            "fitment": "[Detailed analysis of how well the candidate fits the role]",
            "relevant_experience": ["[List of relevant experiences]"],
            "gaps": ["[List of identified gaps in the candidate's profile]"],
            "areas_to_improve": ["[List of areas where the candidate should focus on improving]"]
        }}

        Use the given parameters and job description to inform your evaluation.
        Parameters: {parameters}
        Job Description: {job_description[:500]}...
        Resume: {resume[:500]}...

        Ensure that your evaluation is specific to this candidate and role. Avoid generic assessments.
        Your response must be a valid JSON object.
        """,
        agent=resume_calibrator,
        expected_output="A JSON object containing the evaluation report with fitment score, interview recommendation, and detailed analysis."
    )

def create_skill_evaluation_task(job_description: str, resume: str, skills_agent: Any, role: str, weights: List[float], skills: List[str]) -> Task:
    return Task(
        description=f"""
        Evaluate the candidate's skills based on the job requirements for the role of {role}.
        Provide a detailed skill evaluation report in the following JSON structure:

        {{
            "skill_match_score": [Score as a percentage between 0 and 100],
            "matched_skills": [
                {{
                    "skill": "[Matched skill]",
                    "relevance": [Score between 0 and 10],
                    "description": "[Brief description of how this skill matches the job requirements]"
                }},
                ...
            ],
            "missing_skills": [
                {{
                    "skill": "[Missing skill]",
                    "importance": [Score between 0 and 10],
                    "suggestion": "[Suggestion for acquiring or improving this skill]"
                }},
                ...
            ],
            "overall_skill_assessment": "[Brief overall assessment of the candidate's skill set]"
        }}

        Use the following information to inform your evaluation:
        Job Description: {job_description[:500]}...
        Resume: {resume[:500]}...
        Target Skills: {skills}
        Skill Weights: {weights}

        Ensure that your evaluation takes into account the relative importance of each skill as indicated by the weights.
        Your response must be a valid JSON object.
        """,
        agent=skills_agent,
        expected_output="A JSON object containing the skill evaluation report with skill match score, matched skills, missing skills, and overall assessment."
    )

def create_experience_evaluation_task(job_description: str, resume: str, experience_agent: Any, role: str) -> Task:
    return Task(
        description=f"""
        Evaluate the candidate's work experience and history based on the job requirements for the role of {role}.
        Provide a detailed evaluation report in the following JSON structure:

        {{
            "experience_fitment_score": [Score as a number between 0 and 100],
            "overall_assessment": "[Brief overall assessment]",
            "relevant_experience": [
                {{
                    "experience": "[Relevant experience]",
                    "importance": "[Brief explanation of importance]",
                    "relevance_score": [Score between 0 and 10]
                }},
                ...
            ],
            "gaps": [
                {{
                    "gap": "[Description of the gap]",
                    "severity_score": [Score between 0 and 10]
                }},
                ...
            ],
            "areas_of_improvement": [
                {{
                    "area": "[Area to improve]",
                    "priority": "[High/Medium/Low]"
                }},
                ...
            ],
            "concluding_statement": "[Concluding statement about experience fitment]",
            "interview_recommendation": "[Clear recommendation on whether to interview]"
        }}

        Consider the following factors when determining the score:
        1. Relevance of experience to the role
        2. Years of applicable experience
        3. Demonstrated skills matching the job requirements
        4. Achievements and impact in previous roles
        5. Progression and growth in career

        Use the following information to inform your evaluation:
        Job Requirements: {job_description[:500]}...
        Resume: {resume[:500]}...

        Remember, your evaluation should be unique to this specific candidate and role. Avoid generic assessments.
        Ensure your response is a valid JSON object.
        """,
        agent=experience_agent,
        expected_output="A JSON object containing a comprehensive experience evaluation report including fitment score, relevant experiences, gaps, and recommendations."
    )

def create_project_complexity_evaluation_task(job_description: str, resume: str, project_complexity_agent: Any, role: str) -> Task:
    return Task(
        description=f"""
        Evaluate the complexity and scale of projects the candidate has worked on, based on the job requirements for the role of {role}.
        Provide a detailed project complexity evaluation report in the following JSON structure:

        {{
            "project_complexity_score": [Score as a number between 0 and 100],
            "overall_assessment": "[Brief overall assessment of project complexity experience]",
            "complex_projects": [
                {{
                    "project": "[Brief description of complex project]",
                    "complexity_factors": ["[List of factors that make this project complex]"],
                    "relevance_to_role": [Score between 0 and 10],
                    "impact": "[Brief description of the project's impact or outcome]"
                }},
                ...
            ],
            "areas_for_growth": [
                {{
                    "area": "[Area where candidate could gain more complex project experience]",
                    "importance": "[High/Medium/Low]",
                    "suggestion": "[Suggestion for how to gain this experience]"
                }},
                ...
            ],
            "concluding_statement": "[Concluding statement about the candidate's ability to handle complex projects]"
        }}

        Consider the following factors when evaluating project complexity:
        1. Scale of the projects (e.g., team size, budget, duration)
        2. Technical challenges involved
        3. Cross-functional collaboration required
        4. Impact on the organization or industry
        5. Use of advanced technologies or methodologies

        Use the following information to inform your evaluation:
        Job Requirements: {job_description[:500]}...
        Resume: {resume[:500]}...

        Ensure your evaluation is specific to this candidate and the requirements of the role. 
        Your response must be a valid JSON object.
        """,
        agent=project_complexity_agent,
        expected_output="A JSON object containing a comprehensive project complexity evaluation report including complexity score, assessment of complex projects, and areas for growth."
    )