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

# --- Instalación condicional (TF tiene prioridad) ---
# - Si INSTALL_TF_DEPS=1 y además quieres GPU, pasa TF_FLAVOR=tf-gpu (ver Makefile).
# - Si INSTALL_TF_DEPS=1 y CPU, instalamos tf-cpu.
# - Si INSTALL_TF_DEPS=0 y INSTALL_GPU_DEPS=1 -> PyTorch (gpu).
# - Si ambos 0 -> base CPU sin frameworks.
ARG TF_FLAVOR=tf-cpu  # tf-cpu | tf-gpu

RUN if [ "$INSTALL_TF_DEPS" = "1" ]; then \
      if [ "$TF_FLAVOR" = "tf-gpu" ]; then \
        echo "Instalando TensorFlow GPU..." && \
        poetry install --with tf-gpu --no-root --no-interaction --no-ansi ; \
      else \
        echo "Instalando TensorFlow CPU..." && \
        poetry install --with tf-cpu --no-root --no-interaction --no-ansi ; \
      fi ; \
    elif [ "$INSTALL_GPU_DEPS" = "1" ]; then \
      echo "Instalando PyTorch CUDA (sin TensorFlow)..." && \
      poetry install --with gpu --no-root --no-interaction --no-ansi ; \
    else \
      echo "Instalando base (sin TF ni Torch)..." && \
      poetry install --no-root --no-interaction --no-ansi ; \
    fi

RUN mkdir -p /app/data /app/notebooks /app/scripts

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
