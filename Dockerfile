# syntax=docker/dockerfile:1
FROM python:3.12-slim

# libgomp1: required at import time by TensorFlow / scikit-learn's
# OpenMP-based ops. Not present on python:slim -> silent ImportError
# on `import tensorflow` without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces containers run as uid 1000 — mirror that so file
# ownership matches what the platform expects instead of fighting it.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH="/home/user/.local/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Keras / HF-hub / Matplotlib all try to write cache files at
    # import time. Only /tmp and WORKDIR are guaranteed writable on
    # Spaces, so caches are redirected there defensively.
    KERAS_HOME=/tmp/.keras \
    HF_HOME=/tmp/.huggingface \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.matplotlib

WORKDIR /app

# Dependencies installed before the rest of the source is copied, so
# `docker build` only reinstalls TensorFlow/pandas/etc. when
# requirements.txt actually changes, not on every code push.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["python", "app.py"]