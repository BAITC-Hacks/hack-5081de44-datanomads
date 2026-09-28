FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ARG WITH_MODEL=false
RUN if [ "$WITH_MODEL" = "true" ]; then \
      pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir 'transformers>=4.45,<5'; \
    fi

COPY app ./app
COPY contracts ./contracts
COPY artifacts ./artifacts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
