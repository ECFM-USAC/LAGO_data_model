# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV JUPYTER_ENABLE_LAB=yes

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and poetry.lock to leverage Docker cache
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install --no-cache-dir poetry

# Configure Poetry: Don't create virtual env (we're already in container)
RUN poetry config virtualenvs.create false

# Install dependencies using Poetry
RUN poetry install --only=main --no-dev

# Create necessary directories
RUN mkdir -p /app/data /app/notebooks /app/scripts

# Expose JupyterLab port
EXPOSE 8888

# Set default command to start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]