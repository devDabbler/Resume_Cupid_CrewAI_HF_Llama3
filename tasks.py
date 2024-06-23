import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

# Load tokenizer
model_path = "/home/rezcupid2024/Resume_Cupid_CrewAI_HF_Llama3/model_new"

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
