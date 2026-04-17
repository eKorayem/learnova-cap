#!/bin/bash
set -e

echo "Starting Mini-RAG application..."

# Ensure required directories exist
mkdir -p /app/assets/files
mkdir -p /app/assets/databases

# No database migrations needed - MongoDB uses schema-on-read
# The application creates collections and indexes on startup

echo "Starting uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4