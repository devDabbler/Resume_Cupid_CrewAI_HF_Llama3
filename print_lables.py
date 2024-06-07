from datasets import load_from_disk
import pandas as pd

# Load the tokenized dataset
dataset_path = r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\path_to_save_tokenized_dataset"
tokenized_dataset = load_from_disk(dataset_path)

# Print unique labels before mapping
df = pd.DataFrame(tokenized_dataset)
print("Unique labels in dataset before mapping:", df["accuracy_rating"].unique())