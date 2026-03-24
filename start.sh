#!/bin/bash
# Startup script for HF Spaces — runs FastAPI + Streamlit in one container

echo "=== Starting RMN Engine ==="

# Start FastAPI backend in background (internal port 8000)
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait until backend is healthy (max 60s)
echo "Waiting for backend to be ready..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/ > /dev/null 2>&1; then
        echo "Backend is ready after ${i}s"
        break
    fi
    sleep 1
done

# Start Streamlit on port 7860 (HF Spaces default exposed port)
echo "Starting Streamlit on port 7860..."
python -m streamlit run src/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
