# ---------- Parámetros ----------
ARG BASE_IMAGE=python:3.12-slim
ARG INSTALL_GPU_DEPS=0  # 0=CPU, 1=GPU

# ---------- Etapa base ----------
FROM ${BASE_IMAGE} AS base

WORKDIR /app

ENV PYTHONPATH=/app \
    JUPYTER_ENABLE_LAB=yes \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Paquetes de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiamos manifests primero (para cache)
COPY pyproject.toml poetry.lock ./

# Poetry
RUN pip install --no-cache-dir "poetry>=1.8.0"
RUN poetry config virtualenvs.create false

# Instalación condicional:
# - CPU: sin grupo gpu
# - GPU: con grupo gpu (torch/vision/audio desde índice CUDA)
RUN if [ "$INSTALL_GPU_DEPS" = "1" ]; then \
      echo "Instalando CON grupo gpu..." && \
      poetry lock && poetry install --with gpu --no-root --no-interaction --no-ansi ; \
    else \
      echo "Instalando SIN grupo gpu..." && \
      poetry lock && poetry install --without gpu --no-root --no-interaction --no-ansi ; \
    fi

# Estructura de trabajo
RUN mkdir -p /app/data /app/notebooks /app/scripts

# Exponer Jupyter
EXPOSE 8888

# Comando por defecto
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
