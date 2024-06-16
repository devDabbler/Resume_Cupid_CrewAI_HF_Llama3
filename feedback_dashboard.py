import streamlit as st
import json
import pandas as pd
import os

# Feedback data file
FEEDBACK_FILE = "feedback_data.json"

# Load feedback data from file
def load_feedback_data():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as file:
            return json.load(file)
    return []

# Save feedback data to file
def save_feedback_data(data):
    with open(FEEDBACK_FILE, "w") as file:
        json.dump(data, file)

# Normalize feedback data to dictionary format
def normalize_feedback_data(feedback_data):
    normalized_data = []
    for entry in feedback_data:
        if isinstance(entry, dict):
            normalized_data.append(entry)
    return normalized_data

# Streamlit page setup
st.set_page_config(page_title='Feedback Dashboard', page_icon="📝")
st.title("Feedback Dashboard")
st.markdown("Analyze and visualize the feedback received from users.")

# Load and normalize feedback data
feedback_data = load_feedback_data()
normalized_feedback_data = normalize_feedback_data(feedback_data)

# Display feedback
if len(normalized_feedback_data) > 0:
    feedback_df = pd.DataFrame(normalized_feedback_data, columns=["resume_id", "job_role_id", "name", "accuracy_rating", "content_rating", "suggestions", "submitted_at", "client"])

    # Convert ratings to numeric, setting errors='coerce' to handle non-numeric values
    feedback_df["accuracy_rating"] = pd.to_numeric(feedback_df["accuracy_rating"], errors='coerce')
    feedback_df["content_rating"] = pd.to_numeric(feedback_df["content_rating"], errors='coerce')

    # Handle datetime parsing errors and set as index
    feedback_df["submitted_at"] = pd.to_datetime(feedback_df["submitted_at"], errors='coerce')

    # Drop rows with NaT in submitted_at
    feedback_df = feedback_df.dropna(subset=["submitted_at"])

    # Set submitted_at as index
    feedback_df = feedback_df.set_index("submitted_at")

    # Fill NaNs in ratings with a default value
    feedback_df["accuracy_rating"] = feedback_df["accuracy_rating"].fillna(0)
    feedback_df["content_rating"] = feedback_df["content_rating"].fillna(0)

    # Display overall feedback metrics
    st.subheader("Overall Feedback Metrics")
    average_accuracy_rating = feedback_df["accuracy_rating"].mean()
    average_content_rating = feedback_df["content_rating"].mean()
    total_feedback = len(feedback_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Accuracy Rating", f"{average_accuracy_rating:.2f}")
    col2.metric("Average Content Rating", f"{average_content_rating:.2f}")
    col3.metric("Total Feedback", total_feedback)

    # Display feedback over time
    st.subheader("Feedback Over Time")
    feedback_counts = feedback_df.resample("D").size().reset_index(name="count")
    st.line_chart(feedback_counts.set_index("submitted_at"))

    # Display individual feedback entries
    st.subheader("Individual Feedback Entries")
    for _, row in feedback_df.iterrows():
        resume_id = row["resume_id"]
        job_role_id = row["job_role_id"]
        name = row["name"]
        accuracy_rating = row["accuracy_rating"]
        content_rating = row["content_rating"]
        suggestions = row["suggestions"]
        submitted_at = row.name
        client = row["client"]

        st.write(f"Name of Person Leaving Feedback: {name if name else 'Anonymous'}")
        st.write(f"Candidate or Resume Name: {resume_id}")
        st.write(f"Role: {job_role_id}")
        st.write(f"Client: {client}")
        st.write(f"Accuracy Rating: {accuracy_rating}")
        st.write(f"Content Quality Rating: {content_rating}")
        st.write(f"Suggestions: {suggestions}")
        st.write(f"Submitted at: {submitted_at}")
        st.write("---")

else:
    st.info("No feedback found.")