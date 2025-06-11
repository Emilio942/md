# Dockerfile for ProteinMD containerized deployment
FROM python:3.10-slim

# Set maintainer
LABEL maintainer="Emilio <your.email@example.com>"
LABEL description="ProteinMD - Molecular Dynamics Simulation System"
LABEL version="0.1.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PROTEINMD_DOCKER=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Copy application code
COPY proteinMD/ ./proteinMD/
COPY README.md .
COPY setup.py .

# Install the application
RUN pip install --no-cache-dir -e .

# Create non-root user for security
RUN groupadd -r proteinmd && useradd -r -g proteinmd proteinmd
RUN chown -R proteinmd:proteinmd /app

# Switch to non-root user
USER proteinmd

# Expose port for potential web interfaces
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import proteinMD; print('ProteinMD container healthy')" || exit 1

# Default command
CMD ["python", "-c", "import proteinMD; print('ProteinMD container ready. Use docker exec to run simulations.')"]
