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

# Remove any existing files and clone the latest version of your repository
RUN rm -rf /app/* && git clone https://github.com/your-username/your-repository.git /app

# Set the working directory to the cloned repository
WORKDIR /app

# Copy the config.yaml file into the container
COPY config.yaml /app/config.yaml

# Copy the fine_tuned_model directory into the container
COPY fine_tuned_model /app/fine_tuned_model

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Upgrade transformers to the latest version
RUN pip install --upgrade transformers

# Expose port 8501
EXPOSE 8501

# Run resume_calibrator.py when the container launches
CMD ["streamlit", "run", "resume_calibrator.py", "--server.port=8501", "--server.address=0.0.0.0"]
