# ── Stage 1: Base Python image ──
# python:3.12-slim is smaller than full python image (150MB vs 1GB)
# 'slim' removes build tools we don't need at runtime
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# ── Stage 2: Install dependencies ──
# Copy requirements FIRST (before code) — Docker caches this layer
# So if only code changes (not dependencies), this layer is reused
# This makes rebuilds much faster
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Copy application code ──
COPY backend/app ./app

# ── Stage 4: Configure runtime ──
# Expose port 8000 (documentation — doesn't actually open the port)
EXPOSE 8000

# Health check — Docker/K8s uses this to know if container is alive
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server
# --host 0.0.0.0 makes it accessible from outside the container
# --workers 1 because we're loading ML models (one worker = one copy)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]