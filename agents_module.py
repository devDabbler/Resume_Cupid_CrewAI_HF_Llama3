from crewai import Agent

def standard_scoring(criteria):
    """
    Scoring Guidelines:
    - Expert (5 points):
        - Extensive hands-on experience (5+ years) in applying the skill/technology to complex real-world projects
        - Deep understanding of the skill/technology, its best practices, and advanced concepts
        - Proven track record of delivering significant results and solving complex problems independently
    - Advanced (4 points):
        - Strong hands-on experience (3-5 years) in applying the skill/technology to complex scenarios
        - Solid understanding of the skill/technology, its best practices, and advanced concepts
        - Demonstrated ability to apply the skill/technology effectively with measurable results and minimal guidance
    - Intermediate (3 points):
        - Moderate hands-on experience (1-3 years) in applying the skill/technology to common scenarios
        - Reasonable understanding of the skill/technology, its best practices, and basic concepts
        - Ability to apply the skill/technology with some guidance and achieve tangible outcomes
    - Beginner (2 points):
        - Limited hands-on experience (less than 1 year) in applying the skill/technology to simple scenarios
        - Basic understanding of the skill/technology and its fundamental concepts
        - Requires substantial guidance and supervision to apply the skill/technology with minimal results
    - No Experience (0 points):
        - No explicit mention of hands-on experience, understanding, or results in applying the skill/technology
    """
    if criteria == 'expert':
        return 5
    elif criteria == 'advanced':
        return 4
    elif criteria == 'intermediate':
        return 3
    elif criteria == 'beginner':
        return 2
    else:
        return 0

def create_skills_agent(llm):
    skills_agent = Agent(
        role='Skills Evaluator',
        goal='Critically evaluate the candidate\'s technical skills against the required skills based on the provided scoring guidelines. Look for concrete examples, hands-on experience, and proven results for each skill. Provide a score for each skill and a detailed justification. Be objective and avoid overstating the candidate\'s qualifications. Do not assess soft skills.',
        backstory='An expert in assessing candidate technical skills based on their resume. Focuses on concrete examples, hands-on experience, and proven results. Follows a standardized scoring system and provides objective and critical evaluations.',
        verbose=True,
        llm=llm,
        max_iters=1,
        allow_delegation=False,
        tools=[],
        scoring=standard_scoring
    )
    return skills_agent

def create_resume_calibrator_agent(llm):
    resume_calibrator = Agent(
        role='Resume Calibrator',
        goal='Critically assess the fitment of the given resume against the job requirements. Provide a fitment percentage, detailed reasons for each score assigned, and identify areas where the candidate\'s qualifications may not meet the requirements. Be objective and avoid overstating the candidate\'s fitment.',
        backstory='An expert in evaluating resumes and determining their alignment with job requirements. Uses a consistent scoring methodology, provides a fitment percentage, and identifies areas of improvement. Provides objective and critical evaluations.',
        verbose=True,
        llm=llm,
        max_iters=1,
        allow_delegation=False,
        tools=[]
    )
    return resume_calibrator

def create_experience_agent(llm):
    experience_agent = Agent(
        role='Experience Evaluator',
        goal='Critically evaluate the candidate\'s work experience and history based on the provided scoring guidelines. Focus on relevant work experience, job titles, responsibilities, and their direct relevance to the required skills. Provide a detailed report with scores, reasons for each relevant experience, and identify any gaps or areas of improvement. Be objective and avoid overstating the candidate\'s experience.',
        backstory='An expert in assessing candidate work experience based on their resume. Emphasizes relevant work experience, clear job titles and responsibilities, and their direct relevance to the required skills. Follows a standardized scoring system and provides objective and critical evaluations.',
        verbose=True,
        llm=llm,
        max_iters=1,
        allow_delegation=False,
        tools=[]
    )
    return experience_agent