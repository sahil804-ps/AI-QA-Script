# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies if any (minimal for scikit-learn/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the toolkit code
COPY . .

# Create reports directory for persistent output
RUN mkdir -p reports

# Default command: Run interactive validator (can be overridden)
ENTRYPOINT ["python"]
CMD ["01_Response_Validator/interactive.py"]
