# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Add build argument for GitHub token
ARG GITHUB_TOKEN

# Clone the repository using the personal access token
RUN git clone https://$GITHUB_TOKEN@github.com/devDabbler/Resume_Cupid_CrewAI_HF_Llama3.git /app

# Copy the local model files into the Docker image
COPY ./model /app/model

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV MODEL_PATH=/app/model

# Run app.py when the container launches
CMD ["streamlit", "run", "resume_calibrator_docker.py"]
