# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Copy the entire current directory to the container
COPY . .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Install nltk and download the stopwords data
RUN pip install --no-cache-dir --default-timeout=1000 nltk && python -m nltk.downloader stopwords

# Copy the local model files to the container
COPY model /app/model

# Run the application
CMD ["streamlit", "run", "resume_calibrator_docker.py"]
