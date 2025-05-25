FROM python:3.10-slim

WORKDIR /app

# Copia os fontes primeiro
COPY . .

# Instala pipreqs temporariamente
RUN pip install pipreqs && \
    pipreqs ./api/src --force && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y pipreqs

EXPOSE 8000

CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
