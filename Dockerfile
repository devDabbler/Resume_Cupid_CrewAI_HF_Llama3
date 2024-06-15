FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies with an increased timeout
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt && pip check

# Copy the rest of the application code into the container
COPY . .

# Expose port 80 to the outside world
EXPOSE 80

# Command to run the application
CMD ["streamlit", "run", "resume_calibrator.py"]
