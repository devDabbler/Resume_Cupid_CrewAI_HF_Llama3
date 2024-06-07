from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_from_disk, DatasetDict
import torch
import pandas as pd

def evaluate_model():
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the fine-tuned model and tokenizer
    model_save_path = r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\fine_tuned_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)

    # Load the tokenized dataset
    dataset_path = r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\path_to_save_tokenized_dataset"
    tokenized_dataset = load_from_disk(dataset_path)

    # Ensure label mapping
    df = pd.DataFrame(tokenized_dataset)
    print("Unique labels in dataset before mapping:", df["accuracy_rating"].unique())

    # Create a mapping from old labels to new labels
    label_mapping = {1: 0, 2: 1, 4: 2}  # Make sure this mapping covers all your labels

    def map_labels(example):
        example['accuracy_rating'] = label_mapping.get(example['accuracy_rating'], -1)
        return example

    # Apply label mapping
    tokenized_dataset = tokenized_dataset.map(map_labels)

    # Remove examples with unmapped labels (if any)
    tokenized_dataset = tokenized_dataset.filter(lambda x: x['accuracy_rating'] != -1)

    # Verify new labels
    df = pd.DataFrame(tokenized_dataset)
    print("Unique labels after mapping:", df["accuracy_rating"].unique())

    # Adjust num_labels based on unique labels
    num_labels = len(df["accuracy_rating"].unique())
    print(f"Number of labels: {num_labels}")

    # Split the dataset into training, validation, and test sets if not already split
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    validation_test_split = train_test_split["test"].train_test_split(test_size=0.5)

    # Create DatasetDict with splits
    tokenized_dataset = DatasetDict({
        "train": train_test_split["train"],
        "validation": validation_test_split["train"],
        "test": validation_test_split["test"]
    })

    # Access the test dataset
    test_dataset = tokenized_dataset["test"]

    # Add a column for labels
    test_dataset = test_dataset.rename_column("accuracy_rating", "labels")

    # Define evaluation arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=8,  # Adjust based on your GPU memory
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

if __name__ == '__main__':
    evaluate_model()
