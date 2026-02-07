# Filename: Dockerfile
# VAVE AI - JSW MGI Production Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies
# Note: Keeping gcc and build tools for some Python packages that need compilation
RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    && pip install --upgrade pip setuptools wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
# This is done first to leverage Docker's build cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK data (before copying app code for better caching)
RUN python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Copy the rest of the application's code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/temp/uploads /app/backups

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV FLASK_DEBUG=False

# Create non-root user for security
RUN useradd -m -u 1000 vave && \
    chown -R vave:vave /app

# Switch to non-root user
USER vave

# Expose the port the app runs on
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5000').read()" || exit 1

# Run the app using gunicorn when the container launches
# This is the command for production
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "app:app"]