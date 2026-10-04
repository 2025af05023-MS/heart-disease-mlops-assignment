FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY artifacts/model.joblib ./artifacts/model.joblib
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
ENV PYTHONPATH=/app/src MODEL_PATH=/app/artifacts/model.joblib
EXPOSE 8000
CMD ["uvicorn", "heartlab.api:app", "--host", "0.0.0.0", "--port", "8000"]
