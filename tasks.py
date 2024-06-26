import os
import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

# Define the model path
model_path = "/home/rezcupid2024/Resume_Cupid_CrewAI_HF_Llama3/model_new"

# Debug: Print model_path and list all files in the directory
print(f"Model path: {model_path}")
print(f"Files in the model path: {os.listdir(model_path)}")

vocab_file = os.path.join(model_path, "vocab.txt")
print(f"Vocab file path: {vocab_file}")

if not os.path.isfile(vocab_file):
    raise FileNotFoundError(f"vocab.txt not found in {model_path}")

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
config = BertConfig.from_pretrained(model_path, num_labels=3)
model = BertForSequenceClassification.from_pretrained(model_path, config=config)

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
