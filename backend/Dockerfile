FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
COPY config ./config

# --extra-index-url pulls the CPU-only torch wheel (a few hundred MB) instead of
# PyPI's default CUDA build (multiple GB, useless on this container - no GPU here)
RUN pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 7860

# Render (and most PaaS hosts) inject $PORT and expect the app to bind it, not a fixed port
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
