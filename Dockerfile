FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 nltk && python -m nltk.downloader stopwords

COPY . .
COPY ./model_new /app/model_new

RUN chmod -R 755 /app

ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["streamlit", "run", "resume_calibrator_docker.py", "--server.address", "0.0.0.0", "--server.port", "8501"]