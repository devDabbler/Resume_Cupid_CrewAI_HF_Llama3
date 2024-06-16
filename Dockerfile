FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies with an increased timeout
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt && pip check

# Copy the rest of the application code into the container
COPY . .

# List installed packages for debugging
RUN pip list

# Expose port 8501 to the outside world
EXPOSE 8501

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the application
CMD ["streamlit", "run", "resume_calibrator.py"]
