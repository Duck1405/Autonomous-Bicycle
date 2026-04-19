FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/program

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY sagemaker/requirements.train.txt /opt/program/requirements.train.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 \
        torchvision==0.23.0 && \
    pip install -r /opt/program/requirements.train.txt

COPY train.py /opt/program/train.py
COPY sagemaker/train_entrypoint.py /opt/program/train_entrypoint.py

ENTRYPOINT ["python", "/opt/program/train_entrypoint.py"]
