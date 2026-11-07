# CPU image: docker build -t resenh . && docker run -p 7860:7860 resenh
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 RESENH_WEIGHTS=/app/weights
WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && \
    pip install einops huggingface_hub "gradio>=5" opencv-python-headless pillow numpy
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY weights/release ./weights
RUN pip install --no-deps .
EXPOSE 7860
CMD ["resenh", "app", "--port", "7860"]
