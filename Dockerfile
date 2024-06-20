FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libpoppler-cpp-dev \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create and set working directory
WORKDIR /app

# Clone the GitHub repository
RUN git clone https://your_personal_access_token@github.com/devDabbler/Resume_Cupid_CrewAI_HF_Llama3.git /app

# Copy the model files
COPY ./model /app/model

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords

# Run the application
CMD ["streamlit", "run", "resume_calibrator_docker.py"]
