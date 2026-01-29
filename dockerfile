FROM python:3.11-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY source/ source/

# Expose Flask port
EXPOSE 5000

# MLflow tracking URI (can be overridden in ECS)
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5001

# Run Flask as module
CMD ["python", "-m", "app.flask_app"]
