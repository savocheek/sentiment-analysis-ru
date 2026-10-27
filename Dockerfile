# CPU image: Gradio demo (default) or REST API.
#   docker build -t ru-reviews .
#   docker run -p 7860:7860 ru-reviews                                  # demo  → http://localhost:7860
#   docker run -p 8000:8000 ru-reviews uvicorn api:app --host 0.0.0.0 --port 8000   # API → /docs
# The model is taken from ./model at build time; or set MODEL_PATH to a Hugging Face Hub id at run time.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 DEVICE=cpu GRADIO_SERVER_NAME=0.0.0.0
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sentiment.py app.py api.py ./
COPY model ./model
EXPOSE 7860 8000
CMD ["python", "app.py"]
