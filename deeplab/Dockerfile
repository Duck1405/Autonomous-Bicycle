FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /zapp

COPY requirements.txt .
RUN pip install -r requirements.txt



CMD ["python", "train.py"]