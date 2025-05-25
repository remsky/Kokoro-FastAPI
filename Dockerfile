FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia todos os arquivos do projeto para o container
COPY . .

# Instala pipreqs, gera requirements.txt, instala dependências e remove pipreqs
RUN pip install pipreqs && \
    pipreqs ./api/src --force && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y pipreqs

# Expõe a porta da API
EXPOSE 8000

# Comando de inicialização da aplicação
CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
