import PyPDF2
import re
from transformers import pipeline

# Initialize the Hugging Face NER pipeline
skills_extractor = pipeline("ner", model="dslim/bert-base-NER")

skills_keywords = ["Skills", "Technical Skills", "Programming Languages", "Tools", "Technologies"]

def extract_skills_with_huggingface(resume_text):
    # Extract skills using the Hugging Face NER pipeline
    entities = skills_extractor(resume_text)
    skills = [entity['word'] for entity in entities if entity['entity'] == 'B-SKILL']
    return skills

def read_all_pdf_pages(pdf_path):
    text = ''
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
    return text.strip()

def extract_skills_section(resume, skills_keywords):
    skills_section = ""
    resume = re.sub(r'\s+', ' ', resume)  # Normalize spacing
    for keyword in skills_keywords:
        pattern = re.compile(rf"\b{keyword}\b.*", re.DOTALL | re.IGNORECASE)
        match = pattern.search(resume)
        if match:
            skills_section = match.group().strip()
            break

    if skills_section:
        skills = extract_skills_with_huggingface(skills_section)
        return ", ".join(skills)
    else:
        return ""

def extract_experience_section(resume):
    experience_section = ""
    experience_keywords = ["Experience", "Work Experience", "Employment History", "Professional Experience"]
    resume = re.sub(r'\s+', ' ', resume)  # Normalize spacing
    for keyword in experience_keywords:
        pattern = re.compile(rf"\b{keyword}\b.*", re.DOTALL | re.IGNORECASE)
        match = pattern.search(resume)
        if match:
            experience_section = match.group().strip()
            break
    return experience_section

def display_results(results):
    import json
    import logging
    try:
        results_json = json.dumps(results, indent=4)
        print("Results:\n", results_json)
    except Exception as e:
        logging.error(f"Failed to display results: {str(e)}")

def evaluate_model_performance(model_predictions, human_evaluations):
    # Implement evaluation metrics such as precision, recall, or F1 score
    # Compare the model's predictions against human evaluations
    # Identify areas for improvement based on the performance metrics
    pass