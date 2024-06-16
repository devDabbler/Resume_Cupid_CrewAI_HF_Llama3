FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies with an increased timeout
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt
RUN pip install --default-timeout=300 --no-cache-dir flask

# Copy the application code into the container
COPY . .

# Install the dependencies using setup.py
RUN pip install --default-timeout=300 --no-cache-dir .

# List installed packages for debugging
RUN pip list

# Copy the Supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the ports for Flask and Streamlit
EXPOSE 5000 8501

# Run Supervisor as the entry point
CMD ["/usr/bin/supervisord"]