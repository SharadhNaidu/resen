# CPU image: docker build -t resenh . && docker run -p 7860:7860 resenh
# Also the Render deployment (render.yaml); Render injects $PORT.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 RESENH_WEIGHTS=/app/weights RESENH_EXAMPLES=/app/app/examples \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GRADIO_ANALYTICS_ENABLED=False
WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && \
    pip install einops "gradio>=5" opencv-python-headless pillow numpy
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
RUN pip install --no-deps .
# bake the public weights into the image (downloaded from Hugging Face, sha256-verified)
RUN RESENH_CACHE=/app/weights python -c "from resenh.models.weights import REGISTRY, resolve; [resolve(n) for n, s in REGISTRY.items() if s.get('sha256') and not n.endswith('lolv1')]"
EXPOSE 7860
CMD ["sh", "-c", "resenh app --host 0.0.0.0 --port ${PORT:-7860}"]
