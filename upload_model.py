from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

def upload_model_to_hub(model_path, model_name, model_description, tags):
    api = HfApi()
    api.upload_folder(
        folder_path=model_path,
        repo_id=model_name,
        repo_type='model',
        commit_message='Upload fine-tuned model',
        commit_description=model_description,
        tags=tags
    )

if __name__ == '__main__':
    model_path = os.getenv(r"C:\Users\SEAN COLLINS\Resume_Cupid_CrewAI_HF_Llama3\fine_tuned_model", './fine_tuned_model')
    model_name = 'devDabbler/resume-fitment-bert'
    model_description = 'Fine-tuned BERT model for resume fitment prediction'
    tags = ['bert', 'resume-screening', 'fine-tuned']
    
    upload_model_to_hub(model_path, model_name, model_description, tags)
