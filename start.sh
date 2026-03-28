#!/bin/bash
# Startup script — works on both HF Spaces and Railway

echo "=== Starting RMN Engine ==="

# Railway injects $PORT for the public-facing service.
# We use it for Streamlit (the public UI).
# If not set (HF Spaces / local), default to 7860.
STREAMLIT_PORT=${PORT:-7860}

# FastAPI always runs on internal port 8000 (not exposed publicly)
echo "Starting FastAPI backend on port 8000 (internal)..."
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

# Start Streamlit on $STREAMLIT_PORT (Railway uses PORT, HF Spaces uses 7860)
echo "Starting Streamlit on port ${STREAMLIT_PORT}..."
python -m streamlit run src/streamlit_app.py \
    --server.port "${STREAMLIT_PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
