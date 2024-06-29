import os
import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from crewai import Task
import logging
import json
from datetime import datetime

model_path = os.getenv('MODEL_PATH', '/app/model_new')

# Ensure the model path and files exist
print(f"Model path: {model_path}")
print(f"Files in model path: {os.listdir(model_path)}")

# Check if the required files exist
required_files = ["vocab.txt", "tokenizer_config.json", "special_tokens_map.json"]
for file_name in required_files:
    file_path = os.path.join(model_path, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_name} not found in {model_path}")

# Initialize the tokenizer and model
try:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path, num_labels=3)
    model = BertForSequenceClassification.from_pretrained(model_path, config=config)
except Exception as e:
    print(f"Error loading model: {e}")

# Load ONNX model
ort_session = ort.InferenceSession(os.path.join(model_path, "bert_model.onnx"))

def classify_job_title(job_description, resume_text):
    inputs = tokenizer(job_description + " " + resume_text, return_tensors="np", padding=True, truncation=True)
    
    # Dynamically determine the sequence length
    max_seq_len = ort_session.get_inputs()[0].shape[1]
    
    input_ids = np.zeros((1, max_seq_len), dtype=np.int64)
    attention_mask = np.zeros((1, max_seq_len), dtype=np.int64)
    token_type_ids = np.zeros((1, max_seq_len), dtype=np.int64)

    input_len = min(inputs['input_ids'].shape[1], max_seq_len)

    input_ids[0, :input_len] = inputs['input_ids'][0, :input_len]
    attention_mask[0, :input_len] = inputs['attention_mask'][0, :input_len]
    if 'token_type_ids' in inputs:
        token_type_ids[0, :input_len] = inputs['token_type_ids'][0, :input_len]

    ort_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }
    
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]
    probabilities = softmax(logits, axis=1)
    predicted_class = np.argmax(probabilities, axis=1).item()
    return predicted_class

def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# CrewAI task creation functions

def create_calibration_task(job_description, resume, resume_calibrator, role, parameters):
    """
    Creates a calibration task for evaluating the resume against the job description.
    """
    try:
        calibration_task = Task(
            name="Calibrate Resume",
            description=f"Job Requirements:\n{job_description}\n\nResume:\n{resume}\n\nEvaluate the fitment of the provided resume against the job requirements for the role of {role}. Use the given parameters and scoring guidelines:\n\nParameters:\n{parameters}\n\nScoring Guidelines:\n- Evaluate the candidate's skills and experience against the job requirements.\n- Assign scores based on depth of experience and skill relevance.\n- Provide only a fitment score as a percentage value between 0 and 100.",
            agent=resume_calibrator,
            expected_output="Fitment score as a percentage value between 0 and 100."
        )
        logging.info(f"Created calibration task for role: {role}")
        return calibration_task
    except Exception as e:
        logging.error(f"Failed to create calibration task: {str(e)}")
        raise

def create_skill_evaluation_task(job_description, resume_skills, skills_agent, role, required_skills, weights):
    """
    Creates a skill evaluation task for assessing candidate skills against job requirements.
    """
    try:
        skill_evaluation_task = Task(
            name="Evaluate Skills",
            description=f"Evaluate the candidate's skills against the required skills for the role of {role}. Use the following scoring guidelines:\n\nJob Requirements:\n{job_description}\n\nResume Skills:\n{resume_skills}\n\nRequired Skills:\n{', '.join(required_skills)}\n\nSkill Importance Weights:\n{weights}\n\nScoring Guidelines:\n- expert: 10 points\n- advanced: 8 points\n- intermediate: 5 points\n- beginner: 2 points\n- no experience: 0 points\n\nProvide only a skills fitment score as a percentage value between 0 and 100.",
            agent=skills_agent,
            expected_output="Skills fitment score as a percentage value between 0 and 100."
        )
        logging.info(f"Created skills evaluation task for role: {role}")
        return skill_evaluation_task
    except Exception as e:
        logging.error(f"Failed to create skills evaluation task: {str(e)}")
        raise

def create_experience_evaluation_task(job_description, resume_experience, experience_agent, role):
    """
    Creates an experience evaluation task for assessing work history relevance to the job requirements.
    """
    try:
        experience_evaluation_task = Task(
            name="Evaluate Experience",
            description=f"Evaluate the candidate's work experience and history based on the following job requirements for the role of {role}. Use the provided scoring guidelines:\n\nJob Requirements:\n{job_description}\n\nResume Experience:\n{resume_experience}\n\nScoring Guidelines:\n- Assess the relevance and depth of work experience.\n- Provide only an experience fitment score as a percentage value between 0 and 100.",
            agent=experience_agent,
            expected_output="Experience fitment score as a percentage value between 0 and 100."
        )
        logging.info(f"Created experience evaluation task for role: {role}")
        return experience_evaluation_task
    except Exception as e:
        logging.error(f"Failed to create experience evaluation task: {str(e)}")
        raise

def log_run(input_data, output_data):
    """
    Logs the input and output data of a run.
    """
    try:
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_data": input_data,
            "output_data": output_data
        }
        with open("run_log.json", "a") as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")
        logging.info("Run logged successfully.")
    except Exception as e:
        logging.error(f"Failed to log run: {str(e)}")

# Add this line at the end of the file to log when the module is imported
logging.info("tasks.py module loaded")