FROM python:3.10-slim

WORKDIR /

# Install system deps
RUN apt-get update && apt-get install -y curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv (it goes to ~/.local/bin)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.bashrc

ENV PATH="/root/.local/bin:${PATH}"

# Create a venv and install packages into it
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir numpy pandas requests mcp

# Make the venv the default PATH for subsequent RUN and for the container
ENV PATH="/opt/venv/bin:${PATH}"

CMD ["python3"]
