# Install Python 3.12 on a slim Debian-based image
FROM python:3.12-slim
LABEL authors="amalreji"

# Set environment variables
ENV MODEL_VERSION=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if required
#RUN apt-get update && apt-get install -y \

# Inference-only dependencies (lighter thn full requirements.txt)
COPY app/requirements_inference.txt .
RUN pip install --no-cache-dir -r requirements_inference.txt

# Copy the application code and selected model files using MODEL_VERSION
COPY app/ /app/
# Even though we have a .dockerignore, we explicitly copy the model files to ensure they are included in the image
COPY app/models/model_v${MODEL_VERSION}/ /app/models/model_v${MODEL_VERSION}/

# Expose the port that the FastAPI app will run on, 7860 is commonly used for Hugging Face Spaces
EXPOSE 7860
# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
