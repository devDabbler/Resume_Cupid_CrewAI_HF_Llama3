import logging
import json
from typing import List, Tuple, Any
import re
from crewai import Agent
import traceback
from functools import lru_cache

logger = logging.getLogger(__name__)

class EnhancedAgent(Agent):
    def __init__(self, role, goal, backstory, llm, verbose=False, cache_size=128):
        super().__init__(role=role, goal=goal, backstory=backstory, llm=llm, verbose=verbose)
        self._cache_size = cache_size
        self._setup_cache()

    def _setup_cache(self):
        self._invoke_with_cache = lru_cache(maxsize=self._cache_size)(self._invoke_llm)

    def _process_response(self, response: Any) -> str:
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    def _invoke_llm(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return self._process_response(response)

    def invoke_with_cache(self, prompt: str) -> str:
        return self._invoke_with_cache(prompt)

    def calibrate(self, resume_text: str, job_description: str, role: str) -> float:
        prompt = f"As a resume calibrator, critically assess the fitment of the given resume against the job requirements for the role of {role}. Provide a fitment score between 0 and 100. Resume: {resume_text} Job Description: {job_description}"
        try:
            response = self.invoke_with_cache(prompt)
            return self.extract_fitment_score(response)
        except Exception as e:
            logger.error(f"Error in calibrate: {str(e)}")
            return 0.0

    def evaluate_skills(self, resume_text: str, job_description: str, role: str, skills: List[str], weights: List[float]) -> Tuple[float, List[str], List[str]]:
        prompt = f"As a skills evaluator, analyze the candidate's skills based on the job requirements and resume for the role of {role}. Provide a skill match score between 0 and 100, and list the matched and missing skills. Resume: {resume_text} Job Description: {job_description} Required Skills: {', '.join(skills)}"
        try:
            response = self.invoke_with_cache(prompt)
            return self.extract_skills_info(response, skills)
        except Exception as e:
            logger.error(f"Error in evaluate_skills: {str(e)}")
            return 0.0, [], skills

    def evaluate_experience(self, resume_text: str, job_description: str, role: str) -> dict:
        prompt = f"""As an experience evaluator, analyze the candidate's work experience based on the job requirements for the role of {role}. 
        Provide the following information:
        1. Experience score (0-100): An overall score based on the candidate's total relevant experience.
        2. Relevant experience score (0-100): A score based on how well the candidate's experience matches the job requirements.
        3. List of relevant experiences: Brief descriptions of the candidate's most relevant work experiences.

        Resume: {resume_text[:2000]}... 
        Job Description: {job_description[:1000]}...
        
        Respond in the following JSON format:
        {{
            "experience_score": <score>,
            "relevant_experience_score": <score>,
            "experiences": [
                "{{position}} at {{company}} ({{duration}}) - {{brief description}}",
                ...
            ]
        }}
        
        Ensure that you provide numerical scores and a list of experiences, even if you have to make reasonable estimates based on the information provided.
        """
        try:
            logger.debug(f"Experience evaluation prompt: {prompt[:500]}...")
            response = self.invoke_with_cache(prompt)
            logger.debug(f"LLM response for experience evaluation: {response[:1000]}...")
            result = self.extract_experience_info(response)
            logger.info(f"Experience evaluation result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in evaluate_experience: {str(e)}")
            logger.error(traceback.format_exc())
            return {'experience_score': 0, 'relevant_experience_score': 0, 'experiences': []}

    def evaluate_education(self, resume_text: str) -> dict:
        prompt = f"As an education evaluator, analyze the candidate's educational background. Provide an education score between 0 and 100, the highest education level, and list all degrees. Resume: {resume_text}"
        try:
            logger.debug(f"Prompt passed to evaluate_education: {prompt}")
            response = self.invoke_with_cache(prompt)
            logger.debug(f"LLM response for education evaluation: {response}")
            return self.extract_education_info(response)
        except Exception as e:
            logger.error(f"Error in evaluate_education: {str(e)}")
            return {'education_score': 0, 'education_level': 'Unknown', 'degrees': []}

    def evaluate_project_complexity(self, resume_text: str, job_description: str, role: str) -> dict:
        prompt = f"As a project complexity evaluator, assess the complexity and scale of projects the candidate has worked on based on the job requirements for the role of {role}. Provide a project complexity score between 0 and 100, and list the key projects. Resume: {resume_text} Job Description: {job_description}"
        try:
            logger.debug(f"Prompt passed to evaluate_project_complexity: {prompt}")
            response = self.invoke_with_cache(prompt)
            logger.debug(f"LLM response for project complexity evaluation: {response}")
            return self.extract_project_complexity_info(response)
        except Exception as e:
            logger.error(f"Error in evaluate_project_complexity: {str(e)}")
            return {'project_complexity_score': 0, 'project_details': []}

    def extract_fitment_score(self, response: str) -> float:
        match = re.search(r'\b(\d{1,3}(?:\.\d+)?)\b', response)
        if match:
            score = float(match.group(1))
            return min(max(score, 0), 100)
        return 0

    def extract_skills_info(self, response: str, skills: List[str]) -> Tuple[float, List[str], List[str]]:
        lines = response.split('\n')
        score = 0
        matched_skills = []
        missing_skills = []
        for line in lines:
            if 'score:' in line.lower():
                match = re.search(r'\b(\d{1,3}(?:\.\d+)?)\b', line)
                if match:
                    score = float(match.group(1))
            elif 'matched skills:' in line.lower():
                matched_skills = [s.strip() for s in line.split(':')[1].split(',') if s.strip()]
            elif 'missing skills:' in line.lower():
                missing_skills = [s.strip() for s in line.split(':')[1].split(',') if s.strip()]
        
        if not matched_skills and not missing_skills:
            matched_skills = [skill for skill in skills if skill.lower() in response.lower()]
            missing_skills = [skill for skill in skills if skill not in matched_skills]
        
        return min(max(score, 0), 100) / 100, matched_skills, missing_skills

    def extract_experience_info(self, response: str) -> dict:
        try:
            logger.debug(f"Extracting experience info from response: {response[:1000]}...")
            
            # First, try to parse as JSON
            try:
                json_response = json.loads(response)
                if isinstance(json_response, dict):
                    experience_score = json_response.get('experience_score', 0)
                    relevant_experience_score = json_response.get('relevant_experience_score', 0)
                    experiences = json_response.get('experiences', [])
                    
                    logger.info(f"Successfully extracted experience info from JSON: score={experience_score}, relevant_score={relevant_experience_score}, experiences_count={len(experiences)}")
                    return {
                        'experience_score': max(0, min(100, float(experience_score))),
                        'relevant_experience_score': max(0, min(100, float(relevant_experience_score))),
                        'experiences': experiences
                    }
            except json.JSONDecodeError:
                logger.warning("Failed to parse response as JSON, falling back to text parsing")
            
            # If JSON parsing fails, fall back to text parsing
            experience_score = 0
            relevant_experience_score = 0
            experiences = []
            
            # Extract scores
            experience_match = re.search(r'experience score:?\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if experience_match:
                experience_score = float(experience_match.group(1))
            
            relevant_match = re.search(r'relevant experience score:?\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if relevant_match:
                relevant_experience_score = float(relevant_match.group(1))
            
            # Extract experiences
            exp_matches = re.findall(r'(?:^|\n)(?:- )?(.+?(?:at|,)\s*.+?(?:\(.*?\))?\s*-\s*.+?)(?=\n|$)', response, re.MULTILINE | re.DOTALL)
            experiences = [match.strip() for match in exp_matches if match.strip()]
            
            # If no experiences found, try a more lenient pattern
            if not experiences:
                exp_matches = re.findall(r'(?:^|\n)(?:- )?(.+?(?:at|,)\s*.+?)(?=\n|$)', response, re.MULTILINE | re.DOTALL)
                experiences = [match.strip() for match in exp_matches if match.strip()]
            
            logger.info(f"Extracted experience info via text parsing: score={experience_score}, relevant_score={relevant_experience_score}, experiences_count={len(experiences)}")
            
            # If scores are still 0, try to infer from the text
            if experience_score == 0 and relevant_experience_score == 0:
                year_match = re.search(r'(\d+)\+?\s*years?', response, re.IGNORECASE)
                if year_match:
                    years = int(year_match.group(1))
                    experience_score = min(years * 10, 100)  # 10 points per year, max 100
                    relevant_experience_score = experience_score * 0.8  # Assume 80% relevance
            
            return {
                'experience_score': max(0, min(100, experience_score)),
                'relevant_experience_score': max(0, min(100, relevant_experience_score)),
                'experiences': experiences
            }

        except Exception as e:
            logger.error(f"Error in extract_experience_info: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'experience_score': 0,
                'relevant_experience_score': 0,
                'experiences': []
            }

    def extract_education_info(self, response: str) -> dict:
        education_score = 0
        education_level = "Unknown"
        degrees = []
        lines = response.split('\n')
        for line in lines:
            if 'education score:' in line.lower():
                match = re.search(r'\b(\d{1,3}(?:\.\d+)?)\b', line)
                if match:
                    education_score = float(match.group(1))
            elif 'education level:' in line.lower():
                education_level = line.split(':')[1].strip()
            elif 'degree:' in line.lower():
                degree = line.split(':')[1].strip()
                degrees.append(degree)
        
        if not degrees and education_level == "Unknown":
            degree_matches = re.findall(r'\b(?:Bachelor|Master|PhD|Doctorate)\s*(?:of|in|\'s)?\s*\w+', response, re.IGNORECASE)
            if degree_matches:
                degrees = degree_matches
                education_level = max(degrees, key=lambda x: len(x))
        
        if education_score == 0 and education_level != "Unknown":
            if "phd" in education_level.lower() or "doctorate" in education_level.lower():
                education_score = 100
            elif "master" in education_level.lower():
                education_score = 80
            elif "bachelor" in education_level.lower():
                education_score = 60

        logger.info(f"Extracted education info: score={education_score}, level={education_level}, degrees={degrees}")
        return {
            'education_score': min(max(education_score, 0), 100),
            'education_level': education_level,
            'degrees': degrees
        }

    def extract_project_complexity_info(self, response: str) -> dict:
        project_complexity_score = 0
        project_details = []
        lines = response.split('\n')
        for line in lines:
            if 'project complexity score:' in line.lower():
                match = re.search(r'\b(\d{1,3}(?:\.\d+)?)\b', line)
                if match:
                    project_complexity_score = float(match.group(1))
            elif 'project:' in line.lower():
                project_details.append(line.split(':')[1].strip())

        if project_complexity_score == 0 and project_details:
            project_complexity_score = 50

        logger.info(f"Extracted project complexity info: score={project_complexity_score}, details={project_details}")
        return {
            'project_complexity_score': min(max(project_complexity_score, 0), 100),
            'project_details': project_details
        }

    def clear_cache(self):
        if hasattr(self, 'invoke_with_cache'):
            self.invoke_with_cache.cache_clear()

def create_agent(llm, role: str, goal: str, backstory: str) -> EnhancedAgent:
    logger.info(f"Creating agent with role: {role}")
    return EnhancedAgent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=True
    )

def create_resume_calibrator_agent(llm) -> EnhancedAgent:
    logger.info("Creating resume calibrator agent")
    return create_agent(
        llm,
        role="resume calibrator",
        goal="Critically assess the fitment of the given resume against the job requirements.",
        backstory="An expert in evaluating resumes and determining their alignment with job requirements, with a keen eye for detail and understanding of various industries."
    )

def create_skills_agent(llm) -> EnhancedAgent:
    logger.info("Creating skills evaluator agent")
    return create_agent(
        llm,
        role="skills evaluator",
        goal="Evaluate the candidate's skills based on the job requirements and resume, providing a detailed analysis of matches and gaps.",
        backstory="An expert in assessing technical and soft skills for various roles, with up-to-date knowledge of industry trends and requirements."
    )

def create_experience_agent(llm) -> EnhancedAgent:
    logger.info("Creating experience evaluator agent")
    return create_agent(
        llm,
        role="experience evaluator",
        goal="Evaluate the candidate's work experience and history based on the job requirements, focusing on relevance and impact.",
        backstory="An expert in assessing work experience and determining its relevance to the role, with a deep understanding of career progression and industry-specific achievements."
    )

def create_education_agent(llm) -> EnhancedAgent:
    logger.info("Creating education evaluator agent")
    return create_agent(
        llm,
        role="education evaluator",
        goal="Evaluate the candidate's educational background based on the job requirements, considering both formal education and continuous learning.",
        backstory="An expert in assessing educational qualifications and their relevance to the role, with knowledge of various educational systems and the value of ongoing professional development."
    )

def create_project_complexity_agent(llm) -> EnhancedAgent:
    logger.info("Creating project complexity evaluator agent")
    return create_agent(
        llm,
        role="project complexity evaluator",
        goal="Assess the complexity and scale of projects the candidate has worked on, evaluating their relevance to the job requirements.",
        backstory="An expert in analyzing project descriptions and determining their complexity, scale, and relevance to various industries and roles."
    )

logger.info("agents_module.py module loaded")
