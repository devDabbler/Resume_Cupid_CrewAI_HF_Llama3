from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
import os

model_name = "bert-base-uncased"
save_directory = r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\model"

# Create save directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Download the tokenizer, model, and config
print("Downloading tokenizer, model, and config...")
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
config = BertConfig.from_pretrained(model_name, num_labels=3)
print("Download complete.")

# Save the tokenizer and config
print("Saving tokenizer and config...")
tokenizer.save_pretrained(save_directory)
config.save_pretrained(save_directory)

# Explicitly save the model as pytorch_model.bin
print("Saving model as pytorch_model.bin...")
torch.save(model.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
print(f"Files saved in {save_directory}")

# Check if the files are saved correctly
print(f"Files in save directory: {os.listdir(save_directory)}")
if "pytorch_model.bin" not in os.listdir(save_directory):
    print("Error: pytorch_model.bin not found in save directory.")
else:
    print("pytorch_model.bin found in save directory.")

# Convert to ONNX
print("Converting model to ONNX format...")
dummy_input = torch.ones(1, 128, dtype=torch.long)
try:
    torch.onnx.export(
        model,
        (dummy_input, dummy_input, dummy_input),  # (input_ids, attention_mask, token_type_ids)
        os.path.join(save_directory, "bert_model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["output"],
        opset_version=14,  # Use opset version 14
    )
    print("ONNX conversion complete.")
except Exception as e:
    print(f"Error during ONNX conversion: {e}")

print("Script execution complete.")
