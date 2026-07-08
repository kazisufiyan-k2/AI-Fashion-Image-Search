FROM python:3.11-slim

# libgl1 / libglib2.0-0 are needed by Pillow/FAISS on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so Docker layer caching skips this step on code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860

# $PORT is set by Railway/Render; falls back to 7860 for Hugging Face Spaces
CMD streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0
