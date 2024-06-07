from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_from_disk, DatasetDict
import pandas as pd
import torch

def main():
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the tokenized dataset
    dataset_path = r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\path_to_save_tokenized_dataset"
    tokenized_dataset = load_from_disk(dataset_path)

    # Inspect and remap labels
    df = pd.DataFrame(tokenized_dataset)
    print("Unique labels in dataset:", df["accuracy_rating"].unique())

    # Create a mapping from old labels to new labels
    label_mapping = {1: 0, 3: 1, 4: 2}  # Ensure this mapping covers all your labels

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

    # Split the dataset into training, validation, and test sets if not already split
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    validation_test_split = train_test_split["test"].train_test_split(test_size=0.5)

    # Create DatasetDict with splits
    tokenized_dataset = DatasetDict({
        "train": train_test_split["train"],
        "validation": validation_test_split["train"],
        "test": validation_test_split["test"]
    })

    # Access the splits
    train_dataset = tokenized_dataset["train"]
    validation_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]

    # Print the sizes of the datasets
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Check if any of the datasets are empty
    if len(train_dataset) == 0:
        raise ValueError("The training dataset is empty. Please check the data and ensure it is correctly loaded.")

    # Initialize the model
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,  # Increase batch size for GPU
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',            # Directory for storing logs
        logging_steps=10,
        disable_tqdm=False,  # Enable tqdm progress bar
        dataloader_num_workers=2,  # Increase workers for faster data loading
    )

    # Add a column for labels
    train_dataset = train_dataset.rename_column("accuracy_rating", "labels")
    validation_dataset = validation_dataset.rename_column("accuracy_rating", "labels")

    # Initialize the Trainer with updated arguments
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,  # Ensure tokenizer is passed to trainer
    )

    # Fine-tune the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save the fine-tuned model
    model_save_path = r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\fine_tuned_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

if __name__ == '__main__':
    main()