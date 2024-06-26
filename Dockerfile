# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Install nltk and download the stopwords data
RUN pip install --no-cache-dir --default-timeout=1000 nltk && python -m nltk.downloader stopwords

# Copy the entire current directory to the container
COPY . .

# Copy the config.toml file to the container
COPY config.toml .

# Create necessary directories
RUN mkdir -p /app/model_new

# Copy the model files (adjust the path if necessary)
COPY model_new /app/model_new

# Copy tasks.py to the app directory
COPY tasks.py /app/tasks.py

# Set correct permissions
RUN chmod -R 755 /app

# Set the Python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Run the application
CMD ["streamlit", "run", "resume_calibrator_docker.py", "--server.address", "0.0.0.0", "--server.port", "8501"]