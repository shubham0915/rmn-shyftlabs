# ── Hugging Face Spaces — Docker image ───────────────────────────────────────
# Runs both FastAPI (port 8000, internal) + Streamlit (port 7860, public).
FROM python:3.11-slim

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install grpcio binary wheel FIRST (before chromadb pulls it in as a dep)
# --only-binary=grpcio prevents pip from compiling grpcio from C++ source code,
# which can take 60-90 minutes on a slow CPU. This keeps builds under 5 minutes.
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=grpcio grpcio==1.62.3 grpcio-tools==1.62.3

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Pre-download the embedding model at BUILD time so runtime startup is fast
# (avoids a 30-60s cold start on first request)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create data directory (DuckDB will be created here at runtime)
RUN mkdir -p data

# Make startup script executable
RUN chmod +x start.sh

# NOTE: No EXPOSE directive here.
# When EXPOSE is absent, Railway uses its $PORT env var for BOTH:
#   - routing public traffic
#   - health check target
# Our start.sh reads $PORT and passes it to Streamlit, so all three match.

CMD ["./start.sh"]
