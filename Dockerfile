# Use pre-built OpenCV image
FROM opencv/opencv:4.10.0-python3.11

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
