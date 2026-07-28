FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["sh", "-c", "python build_index.py & uvicorn main:app --host 0.0.0.0 --port 8000"]
