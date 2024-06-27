FROM python:3.10

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Copy the .env file
COPY .env .env

# Ensure environment variables are set
RUN echo "source /app/.env" >> /root/.bashrc

CMD ["streamlit", "run", "resume_calibrator_docker.py"]
