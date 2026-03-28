# Install Python 3.12 on a slim Debian-based image
FROM python:3.12-slim
LABEL authors="amalreji"

# Set environment variables
ENV MODEL_VERSION=1

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container
COPY app/ /app/
# Install inference-only dependencies (lighter than full requirements.txt)
RUN pip install --no-cache-dir -r requirements_inference.txt
# Even though app/models is in .dockerignore, we can explicitly copy the chosen model's files
COPY app/models/model_v${MODEL_VERSION}/ /app/models/model_v${MODEL_VERSION}/

# Expose the port that the FastAPI app will run on, 7860 is commonly used for Hugging Face Spaces
EXPOSE 7860
# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
