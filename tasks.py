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
onnx_model_path = os.path.join(model_path, "bert_model.onnx")
ort_session = ort.InferenceSession(onnx_model_path)

# Print ONNX model input requirements
input_name = ort_session.get_inputs()[0].name
input_shape = ort_session.get_inputs()[0].shape
input_type = ort_session.get_inputs()[0].type
print(f"ONNX Model Input Name: {input_name}")
print(f"ONNX Model Input Shape: {input_shape}")
print(f"ONNX Model Input Type: {input_type}")

def classify_job_title(job_description, resume_text):
    inputs = tokenizer(job_description + " " + resume_text, return_tensors="np", padding=True, truncation=True)
    ort_inputs = {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    }
    
    # Print input shapes and values for debugging
    print("Input shape to ONNX model:", ort_inputs['input_ids'].shape)
    print("Attention mask shape to ONNX model:", ort_inputs['attention_mask'].shape)
    print("Input IDs:", ort_inputs['input_ids'])
    print("Attention mask:", ort_inputs['attention_mask'])
    
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]
    probabilities = softmax(logits, axis=1)
    predicted_class = np.argmax(probabilities, axis=1).item()
    return predicted_class

def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
