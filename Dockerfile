# ==========================================================
# Stage 1: Build the React Frontend
# ==========================================================
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy frontend config and code
COPY frontend/package.json ./
RUN npm install

# Copy shared tool definitions
COPY tools.json ../

COPY frontend/ ./
RUN npm run build

# ==========================================================
# Stage 2: Build the FastAPI Backend & Bundle Frontend
# ==========================================================
FROM python:3.11-slim

# Install system dependencies (including git/curl for MCP bash tool capabilities, plus wireguard client utilities)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    wireguard-tools \
    iptables \
    iproute2 \
    openresolv \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy shared tool definitions
COPY tools.json ./

# Copy frontend static build files to FastAPI static mounts
COPY --from=frontend-builder /app/frontend/dist ./backend/static

# Create workspace directory (which user will mount)
RUN mkdir -p /workspace
ENV WORKSPACE_DIR=/workspace

RUN chmod +x /app/backend/entrypoint.sh

EXPOSE 8000

# Set entrypoint and default uvicorn run command
ENTRYPOINT ["/app/backend/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
