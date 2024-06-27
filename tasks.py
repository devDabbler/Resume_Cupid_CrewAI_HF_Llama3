import os
import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

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
ort_session = ort.InferenceSession("/app/model_new/bert_model.onnx")

def classify_job_title(job_description, resume_text):
    inputs = tokenizer(job_description + " " + resume_text, return_tensors="np", padding=True, truncation=True)
    ort_inputs = {ort_session.get_inputs()[0].name: inputs['input_ids'].astype(np.int64)}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]
    probabilities = softmax(logits, axis=1)
    predicted_class = np.argmax(probabilities, axis=1).item()
    return predicted_class

def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def log_run(input_data, output_data):
    # Assuming this function logs data to a logging service or file
    pass
