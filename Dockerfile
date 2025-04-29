FROM python:3.12-slim
WORKDIR /app

COPY ./requirements.txt .
COPY ./artifacts ./artifacts
COPY ./src ./src

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]