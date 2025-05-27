FROM python:3.10-slim

# Instala dependências de sistema para libs Python (soundfile, pydub, av, etc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavresample-dev \
    libsndfile1 \
    git \
    ca-certificates \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cria o diretório de trabalho
WORKDIR /app

# Copia o requirements e instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante da aplicação
COPY . .

# Expõe a porta da aplicação
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
