# Install Python 3.12 on a slim Debian-based image
FROM python:3.12-slim
LABEL authors="amalreji"

# Build argument (can be overridden at build time)
ARG MODEL_VERSION=1

# Runtime environment variable (available to Python code)
ENV MODEL_VERSION=${MODEL_VERSION}
ENV MODEL_DIR=/app/models/model_v${MODEL_VERSION}

# Set the working directory in the container
WORKDIR /app

# 1. Copy and install requirements first (best caching)
COPY app/requirements_inference.txt .
RUN pip install --no-cache-dir -r requirements_inference.txt

# 2. Copy ALL application code (including ALL model versions for now)
COPY app/ /app/

# 3. Remove ALL model version folders and recreate the one we actually want
RUN rm -rf /app/models/model_v* && \
    mkdir -p ${MODEL_DIR}

# 4. Copy ONLY the chosen model version into the container
COPY app/models/model_v${MODEL_VERSION}/ ${MODEL_DIR}/

# Expose the port that the FastAPI app will run on, 7860 is commonly used for Hugging Face Spaces
EXPOSE 7860
# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
