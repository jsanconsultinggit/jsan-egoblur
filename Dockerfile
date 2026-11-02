# EgoBlur (Gen2) on CPU. Model weights live in ./models as <100 MB zip chunks (GitHub file limit)
# and are reassembled at build time; that layer is cached until the weights change.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TORCH_NUM_THREADS=2 MODEL_DIR=/models

RUN apt-get update && apt-get install -y --no-install-recommends unzip curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ /tmp/models/
RUN mkdir -p /models \
 && for m in face lp; do cat /tmp/models/ego_blur_${m}_gen2.zip.part* > /tmp/m.zip && unzip -q -o /tmp/m.zip -d /models && rm /tmp/m.zip; done \
 && rm -rf /tmp/models && ls -la /models

COPY app.py .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
