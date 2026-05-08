ARG BASE_IMAGE=python:3.12-slim
ARG INSTALL_TF_DEPS=0    # 0/1  -> TensorFlow + hls4ml + keras-tuner (grupo tf)

FROM ${BASE_IMAGE} AS base
WORKDIR /app

ENV PYTHONPATH=/app \
    JUPYTER_ENABLE_LAB=yes \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir "poetry>=1.8.0"
RUN poetry config virtualenvs.create false


# Instalación condicional:
# - CPU:    sólo deps base (numpy, pandas, sklearn, xgboost, jupyterlab, ...)
# - TF-GPU: deps base + grupo [tf] (hls4ml, keras-tuner). TensorFlow ya viene en la imagen base.
RUN if [ "$INSTALL_TF_DEPS" = "1" ]; then \
      echo "==> Modo TensorFlow: deps base + grupo [tf]" && \
      poetry lock && poetry install --with tf --no-root --no-interaction --no-ansi; \
    else \
      echo "==> Modo CPU: sólo deps base" && \
      poetry lock && poetry install --no-root --no-interaction --no-ansi; \
    fi

RUN mkdir -p /app/data /app/notebooks /app/scripts

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
