# Multi-stage image shared by api / consumer / trainer / dashboard (different CMD per workload).
FROM python:3.11-slim AS builder
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --prefix=/install --extra-index-url https://download.pytorch.org/whl/cpu \
        ".[streaming,storage,dashboard,mlops]"

FROM python:3.11-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src ./src
COPY dashboard ./dashboard
COPY feature_repo ./feature_repo
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 FRAUD_ARTIFACTS=/app/artifacts
USER app
EXPOSE 8000 8501
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "fraud.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
