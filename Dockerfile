# Python 3.12 slim base image — kept lightweight (not the full image)
FROM python:3.12-slim

# System dependencies needed by PyMuPDF, torch, etc
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first — takes advantage of Docker layer caching;
# if code changes but requirements don't, dependencies won't be reinstalled
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the entire codebase
COPY . .

# Create data folder (for vector DB and trace storage)
RUN mkdir -p data/vector_db data/uploads

# Streamlit ka default port
EXPOSE 8501

# Health check — verify the container is alive using Streamlit's own endpoint
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to start the app
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
