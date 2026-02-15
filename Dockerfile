
# Base image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install system deps
RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential git curl \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Create logs directory
RUN mkdir -p /app/logs

EXPOSE 8000

# Use a non-root user for security
RUN useradd --create-home appuser || true
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

