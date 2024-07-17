# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container
COPY . .

# Install NLTK and download required resources
RUN pip install nltk
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV NAME World

# Run the application
CMD ["streamlit", "run", "resume_calibrator_docker.py"]
