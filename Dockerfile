FROM python:3.10-slim

# Evita prompts de timezone e configura instalação silenciosa
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema para scipy, soundfile, pydub, av, etc.
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    libsndfile1 \
    git \
    ca-certificates \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
