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
    try:
        inputs = tokenizer(job_description + " " + resume_text, return_tensors="np", padding=True, truncation=True)

        # Debugging: Print the shapes of the inputs
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Attention Mask shape: {inputs['attention_mask'].shape}")

        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }

        # Debugging: Print input shapes for ONNX model
        for name, value in ort_inputs.items():
            print(f"ONNX input name: {name}, shape: {value.shape}")

        ort_outs = ort_session.run(None, ort_inputs)

        # Debugging: Print the output shape from ONNX model
        logits = ort_outs[0]
        print(f"Logits shape: {logits.shape}")

        probabilities = softmax(logits, axis=1)
        predicted_class = np.argmax(probabilities, axis=1).item()
        return predicted_class
    except Exception as e:
        print(f"Error during classification: {e}")
        raise

def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# Test with simplified inputs
if __name__ == "__main__":
    test_job_description = "Data Scientist with experience in Python and Machine Learning."
    test_resume_text = "Experienced in Python, Machine Learning, and Data Science."
    
    print("Classifying job title with test inputs...")
    classification_result = classify_job_title(test_job_description, test_resume_text)
    print(f"Classification Result: {classification_result}")
