FROM python:3.11-slim AS builder
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --prefix=/install ".[cache,triton,ui]"

FROM python:3.11-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src ./src
COPY ui ./ui
COPY scripts ./scripts
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=2 \
    RECSYS_ARTIFACTS=/app/artifacts
USER app
EXPOSE 8100 8501
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8100/health')" || exit 1
CMD ["uvicorn", "recsys.serving.api:app", "--host", "0.0.0.0", "--port", "8100"]
