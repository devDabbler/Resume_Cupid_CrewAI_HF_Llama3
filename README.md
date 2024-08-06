# Resume Cupid

## Overview

Resume Cupid is an intelligent resume evaluation tool designed to streamline the hiring process. Currently developed as an in-house tool, it aims to provide detailed and accurate assessments of candidate resumes against specific job requirements.

**Note:** This project is a work in progress. Future iterations plan to extend functionality to allow candidates to self-assess their resumes against job descriptions.

## Features

- Comprehensive resume parsing and analysis
- Skill matching using NLP and fuzzy matching techniques
- Experience evaluation based on relevance and duration
- Education assessment and scoring
- Project complexity evaluation
- Overall fitment calculation with role-specific weighting
- Detailed feedback and interview recommendations

## Technology Stack

- **Backend:**
  - Python
  - Streamlit (for the web interface)
  - LangChain (for LLM integration)
  - Groq (LLM provider)
  - spaCy (for NLP tasks)
  - SQLite (for feedback storage)

- **Document Processing:**
  - PyMuPDF (for PDF parsing)
  - python-docx (for DOCX parsing)

- **Machine Learning:**
  - scikit-learn (for TF-IDF vectorization and cosine similarity)

- **Additional Libraries:**
  - python-dateutil
  - fuzzywuzzy (for string matching)
  - PyYAML (for configuration management)

## Usage

Login:

Open the Resume Cupid application in your web browser.
Enter your username and password on the login page.
Click the "Login" button to access the main application.


Upload Resume and Job Description:

In the main application interface, you'll see a form for resume evaluation.
Paste the job description into the provided text area. Ensure it's detailed and includes key aspects of the role.
Upload the candidate's resume using the file uploader. Supported formats are PDF and DOCX.


Select Job Role and Skills:

Choose the appropriate job role from the dropdown menu. If the exact role isn't listed, select "Other" and enter a custom job title.
Enter up to five key skills required for the position. These will be used to evaluate the candidate's skillset.
Assign a rank to each skill to indicate its importance (1 being the most important).


Set Experience Requirement:

Enter the minimum years of experience required for the position.


Submit for Evaluation:

Click the "Submit" button to start the resume evaluation process.


Review Results:

Once the evaluation is complete, you'll see a comprehensive analysis of the candidate's resume.
The results include:

Overall fitment score
Recommendation for interview
Detailed fitment analysis
Gap analysis and discussion points
Scores for skills, experience, education, and project complexity


Provide Feedback:

After reviewing the results, you can provide feedback on the evaluation's accuracy and quality.
Fill in the feedback form with your name, the candidate's name, client information, and ratings.
Submit your feedback to help improve the tool's performance.


Logout:

When you're finished using the application, make sure to log out to secure your session.

## Future Developments

- Integration with ATS systems
- Candidate-facing self-assessment tool
- Enhanced visualization of resume analysis results
- API development for third-party integrations

## Contributing

This project is currently for internal use only. Contribution guidelines will be provided if/when the project becomes open-source.

## License

TBD

## Contact

For any queries regarding this project, reach out to me at hello@resumecupid.ai.

---

**Disclaimer:** This tool is designed to assist in the resume evaluation process and should not be used as the sole determinant in hiring decisions. Always combine automated assessments with human judgment and interviews.