# Use the official lightweight Python image.
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libxrender1 \
    libz-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    libtiff-dev \
    libgtk2.0-dev \
    liblcms2-dev \
    libffi-dev \
    tesseract-ocr \
    libleptonica-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the latest version of your repository
RUN git clone https://github.com/devDabbler/Resume_Cupid_CrewAI_HF_Llama3.git /app

# Set the working directory to the cloned repository
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501
EXPOSE 8501

# Run resume_calibrator.py when the container launches
CMD ["streamlit", "run", "resume_calibrator.py", "--server.port=8501", "--server.address=0.0.0.0"]
