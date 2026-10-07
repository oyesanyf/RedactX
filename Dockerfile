# RedactX production image (CPU by default).
#
#   docker build -t redactx:latest .
#   docker build -t redactx:cuda --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 .
#
#   docker run --rm -p 8000:8000 \
#     -v /srv/redactx/models/RedactX-v2:/models/redactx:ro \
#     --env-file redactx.env \
#     redactx:latest
#
# The model is NOT baked into the image: mount an exported checkpoint (a directory with config.json,
# model weights, redactx_span_locator.pt, temperature.json, redactx_thresholds.json) at /models/redactx.
# The container runs fully offline (HF_HUB_OFFLINE=1): exported checkpoints are self-contained.

FROM python:3.11-slim AS base

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG SPACY_MODEL=en_core_web_lg

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# PyTorch first (largest layer, changes least often).
RUN pip install --index-url ${TORCH_INDEX} "torch>=2.2"

COPY pyproject.toml README.md ./
COPY redactx ./redactx
RUN pip install ".[production]" \
 && python -m spacy download ${SPACY_MODEL}

# Operational scripts (calibration, validation, load test) for use inside the container.
COPY calibrate_thresholds.py validate_model.py loadtest.py ./

RUN useradd --create-home --uid 10001 redactx \
 && mkdir -p /models /var/log/redactx \
 && chown -R redactx:redactx /var/log/redactx
USER redactx

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    REDACTX_MODEL_DIR=/models/redactx \
    REDACTX_MODE=hybrid \
    REDACTX_DEVICE=cpu \
    REDACTX_AUDIT_LOG=-

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

# One worker per container: each worker loads its own copy of the model. Scale with replicas.
CMD ["redactx", "serve", "--host", "0.0.0.0", "--port", "8000"]
