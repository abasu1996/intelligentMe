#!/bin/sh
set -eu

exec python -m gunicorn \
  --bind 0.0.0.0:8000 \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 600 \
  main:app
