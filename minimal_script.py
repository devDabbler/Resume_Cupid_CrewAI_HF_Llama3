from transformers import AutoTokenizer
from datasets import Dataset

# Sample data for testing
data = {
    "text": ["Sample text 1", "Sample text 2"],
    "label": [1, 0]
}

# Convert to Hugging Face dataset
dataset = Dataset.from_dict(data)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the input data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Save the dataset
tokenized_dataset.save_to_disk("tokenized_dataset")
