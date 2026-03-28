#!/bin/bash
# Startup script — works on HF Spaces, Railway, and locally

echo "=== Starting RMN Engine ==="
echo "  ENV: PORT=${PORT} (Railway assigns, Streamlit will bind to this)"

# Use Railway's $PORT — Railway routes public traffic AND runs health checks on this port.
# If not set (local/HF), default to 7860.
STREAMLIT_PORT=${PORT:-7860}

# FastAPI runs on FIXED internal port 18000 — never conflicts with anything.
API_PORT=18000

echo "  Streamlit → port ${STREAMLIT_PORT} (public)"
echo "  FastAPI   → port ${API_PORT}  (internal only)"

# Start FastAPI backend in background
python -m uvicorn src.api:app --host 127.0.0.1 --port ${API_PORT} &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait until backend is healthy (max 90s)
echo "Waiting for backend to be ready..."
for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:${API_PORT}/ > /dev/null 2>&1; then
        echo "Backend is ready after ${i}s"
        break
    fi
    sleep 1
done

# Start Streamlit on port 7860 (hardcoded)
echo "Starting Streamlit on port ${STREAMLIT_PORT}..."
python -m streamlit run src/streamlit_app.py \
    --server.port "${STREAMLIT_PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false

