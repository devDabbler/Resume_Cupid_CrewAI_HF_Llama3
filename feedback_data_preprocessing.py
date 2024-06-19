import json
import pandas as pd
from datasets import Dataset
import os
from dotenv import load_dotenv

load_dotenv()

def load_feedback_data(file_path):
    with open(file_path, 'r') as f:
        feedback_data = json.load(f)
    return feedback_data

def preprocess_feedback_data(feedback_data):
    preprocessed_data = []
    for entry in feedback_data:
        preprocessed_entry = {
            'resume_id': entry['resume_id'],
            'job_role_id': entry['job_role_id'],
            'accuracy_rating': entry['accuracy_rating'],
            'content_rating': entry['content_rating'],
            'suggestions': entry['suggestions']
        }
        preprocessed_data.append(preprocessed_entry)
    
    return preprocessed_data

def save_preprocessed_data(preprocessed_data, file_path):
    df = pd.DataFrame(preprocessed_data)
    dataset = Dataset.from_pandas(df)
    dataset.to_json(file_path)

if __name__ == '__main__':
    feedback_file_path = os.getenv(r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\data\feedback_data.json", 'data/feedback_data.json')
    preprocessed_file_path = os.getenv(r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\data\preprocessed_data.json", 'data/preprocessed_data.json')
    
    feedback_data = load_feedback_data(feedback_file_path)
    preprocessed_data = preprocess_feedback_data(feedback_data)
    save_preprocessed_data(preprocessed_data, preprocessed_file_path)
