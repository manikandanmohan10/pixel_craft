FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM python:3.10.15-slim

WORKDIR /app

COPY requirements.txt /app/

RUN python3.10 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN python3.10 load.py

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]