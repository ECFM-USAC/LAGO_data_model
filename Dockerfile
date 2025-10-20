ARG BASE_IMAGE=python:3.12-slim
ARG INSTALL_GPU_DEPS=0   # 0/1  -> PyTorch (grupo gpu)
ARG INSTALL_TF_DEPS=0    # 0/1  -> TensorFlow (grupos tf-cpu/tf-gpu)

FROM ${BASE_IMAGE} AS base
WORKDIR /app

ENV PYTHONPATH=/app \
    JUPYTER_ENABLE_LAB=yes \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir "poetry>=1.8.0"
RUN poetry config virtualenvs.create false


# Instalación condicional de dependencias:
# - CPU: deps base sin grupo gpu
# - GPU-PyTorch: deps base + grupo gpu
# - TF-GPU: deps base (sin torch) + hls4ml (no instalamos tensorflow: ya viene en la imagen TF)
RUN if [ "$INSTALL_TF_DEPS" = "1" ]; then \
      echo "==> Modo TensorFlow: deps base + grupo [tf]" && \
      poetry lock && poetry install --with tf --without gpu --no-root --no-interaction --no-ansi; \
    elif [ "$INSTALL_GPU_DEPS" = "1" ]; then \
      echo "==> Modo PyTorch GPU: deps base + grupo [gpu]" && \
      poetry lock && poetry install --with gpu --no-root --no-interaction --no-ansi; \
    else \
      echo "==> Modo CPU: deps base (SIN grupos extra)" && \
      poetry lock && poetry install --without gpu --no-root --no-interaction --no-ansi; \
    fi

RUN mkdir -p /app/data /app/notebooks /app/scripts

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
