import os
import tempfile
import logging
import json
from dotenv import load_dotenv, find_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text
from utils import read_all_pdf_pages, extract_skills_section, extract_experience_section, skills_keywords
import torch
from crewai import Crew, Process
from langchain_groq import ChatGroq  # Correct import for ChatGroq
from langchain.schema import HumanMessage
from agents_module import create_resume_calibrator_agent, create_skills_agent, create_experience_agent
from tasks import create_calibration_task, create_skill_evaluation_task, create_experience_evaluation_task
import time
from datetime import datetime

# Load environment variables
load_dotenv(find_dotenv())

# Initialize the model and tokenizer
model_save_path = "C:/Users/SEAN COLLINS/Resume_Cupid_CrewAI_HF_Llama3/fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)

# Initialize the LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)

# Initialize logging
logging.basicConfig(filename='resume_calibrator.log', level=logging.ERROR)

# Function to read all pages of a PDF
def read_all_pdf_pages(pdf_path):
    text = ''
    try:
        # Try PyMuPDF first
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        if text.strip():
            return text
    except Exception as e:
        print(f"PyMuPDF extraction failed: {e}")

    try:
        # Fallback to PDFMiner if PyMuPDF fails
        text = pdfminer_extract_text(pdf_path)
        if text.strip():
            return text
    except Exception as e:
        print(f"PDFMiner extraction failed: {e}")

    return ""

# Function to preprocess and predict fitment using the model
def predict_fitment_model(job_description, resume_text):
    inputs = tokenizer(job_description + " " + resume_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

def adjust_score(score):
    # Adjust the score based on your criteria
    adjusted_score = score * 0.6  # Reduce the score by a factor of 0.6
    return round(adjusted_score, 1)

import re

def predict_fitment_llm(job_description, resume_text, role):
    prompt = f"""Evaluate the fitment of the following resume for the role of {role} based on the given job description.
                Consider the following key requirements:
                - Strong experience in machine learning techniques, specifically regression, classification, and unsupervised learning
                - Proficiency in Python and familiarity with relevant libraries such as pandas, sklearn, tensorflow, and PyTorch
                - Ability to design and deploy machine learning algorithms for industrial applications
                - Experience with scalable ML frameworks like MapReduce or Spark
                Assign a fitment score between 0 and 10, where 0 indicates no fitment and 10 indicates a perfect match. Be conservative in your scoring, considering the criticality of the mentioned requirements.

            Job Description:
            {job_description}

            Resume:
            {resume_text}

            Fitment Score:"""
    
    response = llm.predict(prompt, max_tokens=50, temperature=0.2)
    llm_score = response.strip()
    
    # Extract the numeric score using regular expressions
    score_match = re.search(r'\b(\d+)\b', llm_score)
    if score_match:
        numeric_score = float(score_match.group(1))
        adjusted_score = adjust_score(numeric_score)
        return adjusted_score
    else:
        print(f"Unable to extract numeric score from LLM response: {llm_score}")
        return None

# Function to run the fitment prediction multiple times and collect results
def run_multiple_iterations(job_description, resume_text, role, num_iterations):
    results = {"model_predictions": [], "llm_predictions": []}
    for i in range(num_iterations):
        print(f"Iteration {i+1}")
        predicted_class = predict_fitment_model(job_description, resume_text)
        llm_score = predict_fitment_llm(job_description, resume_text, role)
        results["model_predictions"].append(predicted_class)
        results["llm_predictions"].append(llm_score)
        print(f"Model predicted class: {predicted_class}, LLM fitment score: {llm_score}")
    return results

# Set the number of iterations
num_iterations = 20  # Number of times to run the same resume and job description

# Path to resume and job description
resume_file_path = r"C:\Users\SEAN COLLINS\model_fitment_resume\Data Scientist - Jason Youk.pdf"
job_description_file_path = r"C:\Users\SEAN COLLINS\model_fitment_job_description\Data Scientist Job Description.pdf"

# Define role
role = "Data Scientist"

# Read the job description text
job_description_text = read_all_pdf_pages(job_description_file_path)

# Define skills and their rankings
skills = ["Python", "Machine Learning", "Regression", "Unsupervised Learning", "Algorithms"]
skill_rankings = [1, 1, 2, 2, 2]  # Example rankings, modify as needed

# Read the resume text
resume_text = read_all_pdf_pages(resume_file_path)

# Run the fitment prediction multiple times
results = run_multiple_iterations(job_description_text, resume_text, role, num_iterations)

# Save the results for further use
results_save_path = r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\llm_predictions.json"
with open(results_save_path, "w") as file:
    json.dump(results, file)

print("Results from multiple iterations saved to", results_save_path)
