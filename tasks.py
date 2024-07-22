from crewai import Task
import logging

logger = logging.getLogger(__name__)

def create_calibration_task(job_description, resume, resume_calibrator, role, parameters):
    return Task(
        description=f"""
        Evaluate the fitment of the provided resume against the job requirements for the role of {role}.
        Use fuzzy matching to compare skills.
        Provide a detailed evaluation report with the following structure:

        {{
            "fitment_score": [Score as a percentage between 0 and 100],
            "interview_recommendation": "[Clear recommendation on whether to interview the candidate]",
            "fitment": "[Detailed analysis of how well the candidate fits the role]",
            "relevant_experience": "[List of relevant experiences with brief explanations]",
            "gaps": "[Identify specific gaps in the candidate's profile relative to the job requirements]",
            "areas_to_improve": "[Suggest concrete areas where the candidate should focus on improving]"
        }}

        Use the given parameters and job description to inform your evaluation.
        Parameters: {parameters}
        Job Description: {job_description[:100]}...
        Resume: {resume[:100]}...

        Ensure that your evaluation is specific to this candidate and role. Avoid generic assessments.
        Your response should be a valid JSON object.
        """,
        agent=resume_calibrator,
        expected_output="A JSON object containing the evaluation report with fitment score, interview recommendation, and detailed analysis."
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
        Provide a detailed evaluation report in the following JSON structure:

        {{
            "overall_skill_assessment": {{
                "summary": "[Summary of the candidate's skill set]",
                "overall_skill_score": [Score between 0 and 100]
            }},
            "individual_skill_analysis": [
                {{
                    "skill": "[Skill name]",
                    "proficiency_level": "[Beginner/Intermediate/Advanced/Expert]",
                    "evidence": "[Evidence from resume]",
                    "alignment": "[Alignment with job requirements]",
                    "score": [Score between 0 and 10],
                    "relevance": [Score between 0 and 10]
                }},
                ...
            ],
            "skill_gaps": [
                {{
                    "gap": "[Description of the skill gap]",
                    "severity": [Score between 0 and 10],
                    "improvement_suggestion": "[Suggestion for improvement]"
                }},
                ...
            ],
            "skill_relevance": [
                {{
                    "skill": "[Skill name]",
                    "relevance_score": [Score between 0 and 10],
                    "explanation": "[Explanation of relevance]"
                }},
                ...
            ],
            "skill_application": [
                {{
                    "skill": "[Skill name]",
                    "application_example": "[Example of skill application]",
                    "application_score": [Score between 0 and 10]
                }},
                ...
            ],
            "role_performance_impact": {{
                "contribution": "[Explanation of how skills contribute to success]",
                "limitations": "[Potential limitations based on current skill levels]",
                "overall_impact_score": [Score between 0 and 10]
            }},
            "recommendations": [
                {{
                    "recommendation": "[Recommendation for additional training or experience]",
                    "priority": "[High/Medium/Low]"
                }},
                ...
            ]
        }}

        Use the following information to inform your evaluation:
        Job Requirements: {job_description[:100]}...
        Resume Skills: {resume_skills[:100]}...

        Note: Be objective and avoid overstating the candidate's qualifications.
        Ensure your response is a valid JSON object.
        """,
        agent=skills_agent,
        expected_output="A JSON object containing a detailed skill evaluation report including overall assessment, individual skill analysis, and recommendations."
    )

def create_experience_evaluation_task(job_description, resume_text, experience_agent, role):
    return Task(
        description=f"""
        Evaluate the candidate's work experience and history based on the job requirements for the role of {role}.
        Provide a detailed evaluation report with the following structure:

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
            "irrelevant_experience": [
                {{
                    "experience": "[Irrelevant experience]",
                    "explanation": "[Explanation of why it doesn't apply]"
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
            "depth_of_experience": [
                {{
                    "area": "[Key area related to the role]",
                    "depth_score": [Score between 0 and 10]
                }},
                ...
            ],
            "industry_relevance": {{
                "relevance_score": [Score between 0 and 10],
                "explanation": "[Explanation of industry relevance]"
            }},
            "career_progression": {{
                "progression_score": [Score between 0 and 10],
                "analysis": "[Analysis of career progression and growth]"
            }},
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

        Justify your score with specific examples from the resume.

        Use the following information to inform your evaluation:
        Job Requirements: {job_description[:100]}...
        Resume Experience: {resume_text[:100]}...

        Remember, your evaluation should be unique to this specific candidate and role. Avoid generic assessments.
        Ensure your response is a valid JSON object.
        """,
        agent=experience_agent,
        expected_output="A JSON object containing a comprehensive experience evaluation report including fitment score, relevant experiences, gaps, and recommendations."
    )

logger.info("tasks.py module loaded")