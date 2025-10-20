# Use Python 3.11 slim image
FROM python:3.11-slim

# Install minimal system dependencies for opencv-python-headless
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8081

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]
