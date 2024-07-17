import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import os

def download_model(model_name, output_dir):
    # Download model and tokenizer
    config = BertConfig.from_pretrained(model_name, num_labels=3)
    model = BertForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Check if pytorch_model.bin exists
    bin_file = os.path.join(output_dir, "pytorch_model.bin")
    if os.path.isfile(bin_file):
        print("Model successfully downloaded and saved.")
    else:
        print(f"{bin_file} file not found in the specified directory. Attempting to save manually.")
        # Manually save model
        torch.save(model.state_dict(), bin_file)
        if os.path.isfile(bin_file):
            print("Model successfully saved manually.")
        else:
            print(f"Failed to save the model file: {bin_file}")

if __name__ == "__main__":
    model_name = "bert-base-uncased"
    output_dir = "./model"
    download_model(model_name, output_dir)
