FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install base system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    espeak-ng \
    git \
    libsndfile1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support first
RUN pip3 install --no-cache-dir torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies from requirements.txt
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create model directory and set ownership
RUN mkdir -p /app/Kokoro-82M && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Run with Python unbuffered output for live logging
ENV PYTHONUNBUFFERED=1

# Copy only necessary application code
COPY --chown=appuser:appuser api /app/api

# Set Python path (app first for our imports, then model dir for model imports)
ENV PYTHONPATH=/app:/app/Kokoro-82M

# Run FastAPI server with debug logging and reload
CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8880", "--log-level", "debug"]
