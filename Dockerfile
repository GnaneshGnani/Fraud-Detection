FROM python:3.12-slim
# FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
WORKDIR /app

COPY ./requirements.txt .
# COPY ./data ./data
COPY ./*.py .
COPY ./fraud_model.pth .
COPY ./min_max_scaler.pkl .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
