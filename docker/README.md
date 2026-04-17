# Mini-RAG Docker Deployment Guide

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐
│   Nginx     │────▶│   FastAPI   │
│   (Port 80) │     │  (Port 8000)│
└─────────────┘     └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  MongoDB    │   │   Qdrant    │   │ Prometheus  │
│  (27007)    │   │  (6333/34)  │   │   (9090)    │
└─────────────┘   └─────────────┘   └──────┬──────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │   Grafana   │
                                    │   (3000)    │
                                    └─────────────┘
```

## Quick Start

### 1. Configure Environment

```bash
cd docker/env

# Copy example and configure
cp .env.app.example .env.app
cp .env.mongodb.example .env.mongodb
cp .env.grafana.example .env.grafana

# Edit .env.app and add your API keys:
# - GROQ_API_KEY (for text generation)
# - JINA_API_KEY (for embeddings)
# - QUESTION_OPENAI_API_KEY (for question generation)
nano .env.app
```

### 2. Start All Services

```bash
cd docker
docker compose up -d
```

### 3. Verify Health

```bash
# Check all containers are running
docker compose ps

# Test API health
curl http://localhost:8000/api/v1/health

# Check logs
docker compose logs -f fastapi
```

### 4. Access Services

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000/api/v1 | Main REST API |
| Nginx | http://localhost:80 | Reverse proxy |
| MongoDB | localhost:27007 | Database |
| Qdrant | http://localhost:6333 | Vector DB UI |
| Prometheus | http://localhost:9090 | Metrics |
| Grafana | http://localhost:3000 | Dashboards |

## API Endpoints

### Document Management

```bash
# Upload a document
curl -X POST http://localhost:8000/api/v1/courses/{project_id}/documents \
  -F "file=@your_document.pdf"

# Process document into chunks (RAG)
curl -X POST http://localhost:8000/api/v1/courses/{project_id}/documents/chunks \
  -H "Content-Type: application/json" \
  -d '{"chunk_size": 100, "overlap_size": 20, "do_reset": 0}'

# Index into vector DB
curl -X POST http://localhost:8000/api/v1/courses/{project_id}/rag/index \
  -H "Content-Type: application/json" \
  -d '{"do_reset": 0}'
```

### RAG Query

```bash
# Search
curl -X POST http://localhost:8000/api/v1/courses/{project_id}/rag/search \
  -H "Content-Type: application/json" \
  -d '{"text": "your query", "limit": 5}'

# Get answer
curl -X POST http://localhost:8000/api/v1/courses/{project_id}/rag/answer \
  -H "Content-Type: application/json" \
  -d '{"text": "your question", "limit": 10}'
```

### Structure Analysis

```bash
# Analyze document structure
curl -X POST http://localhost:8000/api/v1/courses/{project_id}/structure/analyze \
  -H "Content-Type: application/json" \
  -d '{"max_topics": 10, "use_all_chunks": false}'

# Process with structure
curl -X POST http://localhost:8000/api/v1/courses/{project_id}/documents/chunks/structure \
  -H "Content-Type: application/json" \
  -d '{"chunk_size": 1000, "overlap_size": 100}'
```

### Question Generation

```bash
# Process for questions
curl -X POST http://localhost:8000/api/v1/courses/{project_id}/documents/chunks/questions \
  -H "Content-Type: application/json" \
  -d '{"chunk_size": 1500, "overlap_size": 150}'

# Generate questions
curl -X POST http://localhost:8000/api/v1/courses/{project_id}/questions/generate \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req-123",
    "course_id": 1,
    "project_id": "your-project-id",
    "topics": [
      {
        "topic_id": 1,
        "topic_title": "Introduction",
        "question_configs": [
          {"type": "multiple_choice", "difficulty": "easy", "count": 3},
          {"type": "short_answer", "difficulty": "medium", "count": 2}
        ]
      }
    ]
  }'
```

## Troubleshooting

### FastAPI won't start

```bash
# Check logs
docker compose logs fastapi

# Common issues:
# 1. Missing API keys in .env.app
# 2. MongoDB not healthy (check: docker compose logs mongodb)
# 3. Qdrant not healthy (check: docker compose logs qdrant)
```

### MongoDB connection failed

```bash
# Verify MongoDB is running
docker compose exec mongodb mongosh --eval "db.adminCommand('ping')"

# Reset MongoDB (WARNING: deletes all data)
docker compose down -v
docker compose up -d mongodb
```

### Qdrant issues

```bash
# Check Qdrant UI
open http://localhost:6333

# Check collections
curl http://localhost:6333/collections
```

## Stop Services

```bash
# Stop all
docker compose down

# Stop and remove volumes (WARNING: deletes all data)
docker compose down -v
```

## Production Deployment

### Systemd Service

```bash
# Copy and edit the service file
sudo cp docker/minirag.service /etc/systemd/system/
sudo nano /etc/systemd/system/minirag.service

# Update:
# - User=your_username
# - WorkingDirectory=/path/to/mini-rag-app/docker

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable minirag
sudo systemctl start minirag
sudo systemctl status minirag
```

### Security Checklist

- [ ] Replace `allow_origins=["*"]` in main.py with specific origins
- [ ] Add authentication middleware
- [ ] Use secrets management (Docker secrets, Vault, etc.)
- [ ] Enable HTTPS with TLS termination
- [ ] Configure firewall rules
- [ ] Set up log aggregation
- [ ] Configure backup strategy for MongoDB and Qdrant
