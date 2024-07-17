import json
import os
import pandas as pd
import streamlit as st

FEEDBACK_FILE = r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\data\feedback_data.json"

def load_feedback_data():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as file:
            content = file.read().strip()
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    st.error("Failed to decode feedback data. The file may be corrupted.")
                    return []
            else:
                return []
    return []

def preprocess_feedback_data(feedback_data):
    preprocessed_data = []
    for entry in feedback_data:
        preprocessed_entry = {
            'resume_id': entry['resume_id'],
            'job_role_id': entry['job_role_id'],
            'accuracy_rating': entry['accuracy_rating'],
            'content_rating': entry['content_rating'],
            'suggestions': entry['suggestions'],
            'submitted_at': entry['submitted_at'],
            'client': entry.get('client', 'N/A')
        }
        preprocessed_data.append(preprocessed_entry)
    return preprocessed_data

def display_feedback_dashboard(preprocessed_data):
    st.title("Feedback Dashboard")
    st.markdown("Analyze and visualize the feedback received from users.")

    if preprocessed_data:
        df = pd.DataFrame(preprocessed_data)
        st.write(df)
    else:
        st.write("No feedback data available.")

feedback_data = load_feedback_data()
preprocessed_data = preprocess_feedback_data(feedback_data)
display_feedback_dashboard(preprocessed_data)
