from setuptools import setup, find_packages

setup(
    name='resume_cupid',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'streamlit_authenticator',
        'python-dotenv',
        'transformers==4.39.3',  # Ensure compatibility with other packages
        'torch',
        'tenacity',
        'pymupdf',
        'pdfminer.six',
        'crew',
        'crewai',
        'langchain==0.1.20',
        'langchain-core',
        'langchain-community',
        'langchain-groq',
        'tokenizers==0.15.2',  # Compatible with langchain
        'threadpoolctl',
        'scikit-learn',
        'aiofiles<24.0.0,>=23.1.0',  # Required by chainlit
        'asyncer<0.0.3,>=0.0.2',  # Required by chainlit
        'fastapi-socketio<0.0.11,>=0.0.10',  # Required by chainlit
        'filetype<2.0.0,>=1.2.0',  # Required by chainlit
        'lazify<0.5.0,>=0.4.0',  # Required by chainlit
        'literalai',  # Align version with chainlit requirements
        'nest-asyncio<2.0.0,>=1.5.6',  # Required by chainlit
        'nltk'
        'numpy'
        'pyjwt<3.0.0,>=2.8.0',  # Required by chainlit
        'python-multipart<0.0.10,>=0.0.9',  # Required by chainlit
        'syncer<3.0.0,>=2.0.3',  # Required by chainlit
        'tomli<3.0.0,>=2.0.1',  # Required by chainlit
        'uptrace<2.0.0,>=1.22.0',  # Required by chainlit
        'uvicorn<0.26.0,>=0.25.0',  # Required by chainlit
        'watchfiles<0.21.0,>=0.20.0',  # Required by chainlit
        'dirtyjson<2.0.0,>=1.0.8',  # Required by llama-index-core
        'chardet',  # Required by unstructured
        'lxml',  # Required by unstructured
        'PyPDF2'  # Missing dependency
    ],
    entry_points={
        'console_scripts': [
            'resume_cupid = app:main',  # Define entry point for your app, change as needed
        ],
    },
)
