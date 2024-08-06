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

## Setup and Installation

(Instructions for setting up the project locally would go here)

## Usage

(Instructions on how to use the application would go here)

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