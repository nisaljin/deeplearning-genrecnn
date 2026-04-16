ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

COPY requirements.txt .

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Backend files
COPY infer_api.py fma_classes.json ./
COPY outputs/best_fma_multilabel.pt ./outputs/

EXPOSE 8000

CMD ["uvicorn", "infer_api:app", "--host", "0.0.0.0", "--port", "8000"]
