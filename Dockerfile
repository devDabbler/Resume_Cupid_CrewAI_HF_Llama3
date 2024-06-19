# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Clone the repository
RUN git clone https://github.com/devDabbler/Resume_Cupid_CrewAI_HF_Llama3.git /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV MODEL_PATH=/app/model

# Pull the latest changes from the repository
RUN cd /app && git pull

# Run app.py when the container launches
CMD ["streamlit", "run", "resume_calibrator_docker.py"]
