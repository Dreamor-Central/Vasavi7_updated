# ─────────────── Builder Stage ───────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system packages required for build
# build-essential for compiling C extensions (often needed for Python packages)
# libpq-dev for PostgreSQL client libraries (if you use psycopg2 or similar)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
# Using --user to install into /root/.local, which is then copied to the runtime stage
# --no-cache-dir reduces image size
# --timeout and --retries help with flaky network during build
# -i https://pypi.org/simple ensures standard PyPI
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt \
        --timeout=60 \
        --retries=5 \
        -i https://pypi.org/simple

# Preload the SentenceTransformer model for faster cold starts
# This downloads the model during build, making the runtime image larger but startup faster.
# Be very mindful of the memory implications for this model on your deployment platform's plan.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

# ─────────────── Runtime Stage ───────────────
FROM python:3.12-slim

WORKDIR /app

# Install minimal runtime tools (optional but useful for debugging)
# curl for making HTTP requests
# dnsutils for host lookups (dig, nslookup)
# iputils-ping for basic network connectivity checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    dnsutils \
    iputils-ping \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only the user-installed Python packages from the builder stage
# This keeps the runtime image clean and small by not including build dependencies
COPY --from=builder /root/.local /root/.local

# Update PATH environment variable to include the directory where user-installed packages' executables reside
ENV PATH=/root/.local/bin:$PATH

# Copy only the necessary application source files into the runtime image
COPY . .

# Expose the port FastAPI will run on.
# This is documentation for Docker, it doesn't actually publish the port.
# The actual port mapping happens during 'docker run -p' or in your cloud service configuration.
EXPOSE 8000

# Run the FastAPI app with production-grade settings
# CORRECTED: Using the "shell form" of CMD to allow environment variable expansion.
# IMPORTANT: Remove the square brackets [] and commas, as well as the quotes around the entire command.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 60