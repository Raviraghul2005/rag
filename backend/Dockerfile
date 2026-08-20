FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
COPY config ./config

RUN pip install --no-cache-dir -e .

EXPOSE 7860

# Render (and most PaaS hosts) inject $PORT and expect the app to bind it, not a fixed port
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
