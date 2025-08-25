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

# Copy pyproject.toml first to leverage Docker cache
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy>=1.26.0 \
    pandas>=2.1.0 \
    seaborn>=0.12.2 \
    jupyterlab>=4.0.0 \
    "tqdm>=4.67.1,<5.0.0"

# Create necessary directories
RUN mkdir -p /app/data /app/notebooks /app/scripts

# Expose JupyterLab port
EXPOSE 8888

# Set default command to start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]