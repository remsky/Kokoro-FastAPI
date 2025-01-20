FROM python:3.10-slim

WORKDIR /app/ui

# Install dependencies
RUN pip install gradio==5.9.1 requests==2.32.3

# Create necessary directories
RUN mkdir -p data/inputs data/outputs

# Copy the application files
COPY . .

ENV API_HOST=kokoro-tts
ENV API_PORT=8880

# Run the Gradio app
CMD ["python", "app.py"]
