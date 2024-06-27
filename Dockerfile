FROM python:3.10

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional NLTK data
RUN python -m nltk.downloader stopwords

# Copy the rest of the application code
COPY . .

# Copy the .env file
COPY .env .env

# Ensure environment variables are set
RUN echo "source /app/.env" >> /root/.bashrc

# Make sure the model directory exists
RUN mkdir -p /app/model_new

# Set the correct permissions
RUN chmod -R 755 /app

# Set the Python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Run the Streamlit app
CMD ["streamlit", "run", "resume_calibrator_docker.py", "--server.port", "8501", "--server.address", "0.0.0.0"]