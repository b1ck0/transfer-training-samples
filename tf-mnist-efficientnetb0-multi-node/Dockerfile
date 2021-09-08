FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-6:latest

WORKDIR /app/
COPY main.py main.py
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENV ROLE='None'
ENTRYPOINT ["python", "main.py"]
