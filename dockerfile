FROM python:3.10-slim

WORKDIR /

# Install system dependencies (curl required for uv)
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.bashrc

# Add uv to PATH for all RUN commands
ENV PATH="/root/.local/bin:${PATH}"

# Install Python packages using uv (much faster than pip)
RUN uv pip install numpy pandas requests mcp

CMD ["python3"]
