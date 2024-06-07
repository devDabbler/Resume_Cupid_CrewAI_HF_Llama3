import pandas as pd
from datasets import load_from_disk, Dataset, DatasetDict

# Load the tokenized dataset
dataset_path = r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\path_to_save_tokenized_dataset"
tokenized_dataset = load_from_disk(dataset_path)

# Inspect the dataset structure
print("Dataset structure:")
print(tokenized_dataset)

# Check unique values of labels before splitting
df = pd.DataFrame(tokenized_dataset)
print("Unique labels in dataset:", df["accuracy_rating"].unique())

# Split the dataset into training, validation, and test sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
validation_test_split = train_test_split["test"].train_test_split(test_size=0.5)

# Create DatasetDict with splits
tokenized_dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": validation_test_split["train"],
    "test": validation_test_split["test"]
})

# Verify splits
print("Train dataset size:", len(tokenized_dataset["train"]))
print("Validation dataset size:", len(tokenized_dataset["validation"]))
print("Test dataset size:", len(tokenized_dataset["test"]))

# Convert to pandas DataFrame for easy inspection
train_df = pd.DataFrame(tokenized_dataset["train"])
validation_df = pd.DataFrame(tokenized_dataset["validation"])
test_df = pd.DataFrame(tokenized_dataset["test"])

# Check unique values of labels
print("Unique labels in training dataset:", train_df["accuracy_rating"].unique())
print("Unique labels in validation dataset:", validation_df["accuracy_rating"].unique())
print("Unique labels in test dataset:", test_df["accuracy_rating"].unique())
