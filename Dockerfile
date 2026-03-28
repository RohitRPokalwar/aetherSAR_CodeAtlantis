# Dockerfile for Hugging Face Spaces / Render / AWS
# Using Python 3.9 slim for smaller footprint

FROM python:3.9-slim

# Install system dependencies for OpenCV and GIS
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement files
COPY requirements.txt .
COPY backend/requirements.txt backend/

# Install Python dependencies
# Using --no-cache-dir to save space
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r backend/requirements.txt \
    && pip install --no-cache-dir uvicorn

# Copy the entire project
COPY . .

# Expose port (Hugging Face uses 7860 by default, Render uses $PORT)
ENV PORT=7860
EXPOSE 7860

# Run the FastAPI app
# For Hugging Face, we use 0.0.0.0 and port 7860
CMD uvicorn backend.api:app --host 0.0.0.0 --port $PORT
