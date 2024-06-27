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
