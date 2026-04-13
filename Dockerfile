FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_PROGRESS_BAR off
ENV PIP_NO_COLOR 1
ENV RAG_CONFIG_PATH /app/config_rag.yaml

WORKDIR /app

# The research server profile blocks apt-get; skipping system dependencies per SERVER_DEPLOYMENT_GUIDE.md
# If native libraries like libgomp1 are required, we will address them via specialized base images later.

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn fastapi python-multipart

# Copy project files
COPY . .

# Expose port 8000
EXPOSE 8000

# Default command
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
