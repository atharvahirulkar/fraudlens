# Dockerfile for FraudLens inference service
FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements-api.txt .
COPY constraints.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -c constraints.txt -r requirements-api.txt

# Copy application code
COPY src/ /app/src/
COPY data/fraud_patterns/ /app/data/fraud_patterns/

# Bundle model artifacts so the container is self-contained.
# In local docker-compose, the volume mount overrides this with the live mlruns.
# In CI, the deploy workflow syncs from S3 before docker build.
COPY mlruns/ /app/mlruns/

# Pre-download embedding model for offline availability
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')" 2>&1 | tail -1 || echo "Model cached"

# Default to the bundled file-based MLflow store.
# docker-compose overrides this to http://mlflow:5000 for local dev.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=file:///app/mlruns \
    QDRANT_URL= \
    LLM_BACKEND=openai

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
