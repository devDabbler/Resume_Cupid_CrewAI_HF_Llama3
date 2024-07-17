import pandas as pd
from datasets import Dataset # type: ignore
from transformers import AutoTokenizer
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting the script...")

# Load the feedback data, run logs data, and LLM predictions data
feedback_file_path = r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\feedback_data.json"
run_logs_file_path = r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\run_logs.json"
llm_prediction_file_path = r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\llm_prediction.json"

logger.info("Loading feedback data...")
with open(feedback_file_path, 'r') as file:
    raw_feedback_data = json.load(file)

logger.info("Loading run logs data...")
with open(run_logs_file_path, 'r') as file:
    run_logs_data = json.load(file)

logger.info("Loading LLM prediction data...")
with open(llm_prediction_file_path, 'r') as file:
    llm_prediction_data = json.load(file)

# Print the structure of llm_predictions_data
logger.info("LLM prediction data structure:")
logger.info(json.dumps(llm_prediction_data, indent=2))

# Extract only the feedback entries, run log entries, and LLM predictions (skip the headers)
logger.info("Extracting feedback, run log, and LLM prediction entries...")
feedback_entries = [entry for entry in raw_feedback_data if isinstance(entry, dict)]
run_log_entries = [entry for entry in run_logs_data if isinstance(entry, dict)]

# Since llm_predictions_data is a list of predictions, create a DataFrame directly
llm_prediction_df = pd.DataFrame({
    'timestamp': pd.date_range(start='2024-06-01', periods=len(llm_prediction_data['model_predictions']), freq='D'),
    'model_prediction': llm_prediction_data['model_predictions'],
    'llm_prediction': llm_prediction_data['llm_predictions']
})

# Convert to DataFrames for analysis
feedback_df = pd.DataFrame(feedback_entries)
run_logs_df = pd.DataFrame(run_log_entries)

# Clean and prepare data for merging
logger.info("Cleaning and preparing data for merging...")
feedback_df['submitted_at'] = pd.to_datetime(feedback_df['submitted_at'])
run_logs_df['timestamp'] = pd.to_datetime(run_logs_df['timestamp'], format='ISO8601')

# Extract date from timestamp
feedback_df['date'] = feedback_df['submitted_at'].dt.date
run_logs_df['date'] = run_logs_df['timestamp'].dt.date
llm_prediction_df['date'] = llm_prediction_df['timestamp'].dt.date

# Print columns before merging
logger.info("Feedback DataFrame columns: %s", feedback_df.columns)
logger.info("Run Logs DataFrame columns: %s", run_logs_df.columns)
logger.info("LLM Prediction DataFrame columns: %s", llm_prediction_df.columns)

# Merge feedback, run logs, and LLM predictions based on date for alignment
logger.info("Merging feedback, run logs, and LLM prediction...")
merged_df = pd.merge(feedback_df, run_logs_df, on='date', how='inner')
merged_df = pd.merge(merged_df, llm_prediction_df, on='date', how='inner')

# Print columns after merging
logger.info("Merged DataFrame columns: %s", merged_df.columns)

# Check if job_description exists in the input data
if 'job_description' not in merged_df.columns:
    logger.info("job_description column not found in merged DataFrame, extracting from input data if available.")
    # Assuming job_description is part of input_data and extracting it
    def extract_job_description(input_data):
        if not isinstance(input_data, str):
            return ''
        try:
            data = json.loads(input_data)
            return data.get('job_description', '')
        except json.JSONDecodeError:
            return ''

    merged_df['job_description'] = merged_df['input_data'].apply(extract_job_description)

# Check if resume exists in the input data
if 'resume' not in merged_df.columns:
    logger.info("resume column not found in merged DataFrame, extracting from input data if available.")
    # Assuming resume is part of input_data and extracting it
    def extract_resume(input_data):
        if not isinstance(input_data, str):
            return ''
        try:
            data = json.loads(input_data)
            return data.get('resume', '')
        except json.JSONDecodeError:
            return ''

    merged_df['resume'] = merged_df['input_data'].apply(extract_resume)

# Prepare the dataset for training
logger.info("Preparing the dataset for training...")
merged_df['input'] = merged_df['input'].apply(lambda x: str(x))
merged_df['job_description'] = merged_df['job_description'].apply(lambda x: str(x))
merged_df['resume'] = merged_df['resume'].apply(lambda x: str(x))
training_data = merged_df[['input', 'job_description', 'resume', 'llm_prediction', 'accuracy_rating']].dropna()

# Initialize tokenizer
model_name = "bert-base-uncased"
logger.info(f"Initializing tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the input data
logger.info("Tokenizing the input data...")
def tokenize_function(examples):
    return tokenizer([ex + ' ' + jd + ' ' + rs for ex, jd, rs in zip(examples['input'], examples['job_description'], examples['resume'])], padding='max_length', truncation=True)

# Convert to Hugging Face dataset
logger.info("Converting to Hugging Face dataset...")
dataset = Dataset.from_pandas(training_data)

# Tokenize the dataset
logger.info("Tokenizing the dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Save the dataset
output_path = r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\path_to_save_tokenized_dataset"
logger.info(f"Saving the tokenized dataset to {output_path}...")
tokenized_dataset.save_to_disk(output_path)

logger.info("Script completed successfully.")
