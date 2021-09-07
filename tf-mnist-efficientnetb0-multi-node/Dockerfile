FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app/
COPY main.py main.py
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]
