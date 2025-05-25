FROM python:3.10-slim

# Instala libs de sistema necessárias para soundfile, pydub e av (PyAV)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho no container
WORKDIR /app

# Copia e instala as dependências do projeto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante da aplicação
COPY . .

# Expõe a porta da aplicação
EXPOSE 8000

# Comando de execução padrão
CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
