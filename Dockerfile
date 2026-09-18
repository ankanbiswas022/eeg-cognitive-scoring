# Inference image: CPU-only torch keeps it small enough for a standard container host.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 EEGSCORE_ARTIFACTS=/app/artifacts PORT=8000

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir .

# trained artifacts are mounted or copied in at build time
COPY artifacts ./artifacts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "eegscore.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
