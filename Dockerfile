FROM python:3.12-slim

# Hugging Face Spaces runs as a non-root user; port 7860 is the default
# exposed port for Docker Spaces. We also support 8000 for local use.
WORKDIR /app

# Install dependencies first (layer-cached)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what the API needs at runtime
COPY api.py .
COPY models/final/final_model.pkl     models/final/final_model.pkl
COPY models/final/final_model_metadata.json models/final/final_model_metadata.json
COPY models/preprocessor.pkl          models/preprocessor.pkl
COPY reports/phase_7/final_threshold.json reports/phase_7/final_threshold.json

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Start uvicorn. On HF Spaces the host must be 0.0.0.0 and port 7860.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
