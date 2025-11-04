FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package code
COPY titanic_ml/ ./titanic_ml/
COPY setup.py .
RUN pip install -e .

# Ensure package defaults look in container data directory
ENV TITANIC_DATA_DIR=/app

# Copy notebooks and data (optional, for development)
COPY *.ipynb ./
COPY *.csv ./

# Expose port for API (if serving)
EXPOSE 5000

# Default command (can be overridden)
CMD ["python", "-c", "import titanic_ml; print('Titanic ML package installed successfully')"]







