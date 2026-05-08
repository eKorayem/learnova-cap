# Learnova AI Backend (Capstone Project)

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=flat&logo=docker&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=flat&logo=mongodb&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red)

## Overview
Learnova is an intelligent, AI-driven educational platform designed to revolutionize how students and educators interact with academic materials. 

This repository houses the core **AI Backend Engine**, built with FastAPI. It is responsible for ingesting complex academic documents (PDFs, textbooks, lecture slides), parsing them using state-of-the-art Large Language Models, and transforming them into interactive, structured learning experiences.

## Core Features

* **Universal Academic Parser:** Dynamically ingests diverse file formats. It uses advanced spatial heuristics and AI to flawlessly extract Table of Contents, chapter hierarchies, and sparse lecture slides, while actively filtering out academic noise and boilerplate.
* **Bilingual Native Support:** Fully optimized to process, parse, and generate content in both **English** and **Arabic** natively without breaking structural integrity.
* **Intelligent Question Generation:** Leverages deep reasoning models (like GPT-4o / gpt-oss-120b) to generate rigorous Multiple Choice Questions (MCQs) with highly plausible educational distractors.
* **RAG-Powered Tutoring:** Embeds document chunks into a Vector Space to power a highly accurate, hallucination-free chatbot that answers student questions based strictly on the uploaded curriculum.

## Architecture & Tech Stack

* **Backend Framework:** FastAPI (Python)
* **Databases:**
  * MongoDB (Document & Metadata Storage)
  * Qdrant (Vector Database for RAG Embeddings)
  * Supabase (Cloud File Storage)
* **AI & Machine Learning:**
  * LLM Providers: OpenRouter, Groq
  * Supported Models: Claude 3.5 Sonnet, GPT-4o, GPT-OSS-120B
  * Embeddings: Jina-Embeddings-v3, Cohere Multilingual
* **DevOps:** Docker & Docker Compose V2

---

## Getting Started (Local Development)

### Prerequisites
* Docker Engine & **Docker Compose V2** plugin installed.
* Git

### 1. Clone the repository
```bash
git clone [https://github.com/](https://github.com/)[your-username]/learnova-cap.git
cd learnova-cap
```

### 2. Environment Configuration
Create a .env file in the root directory and configure your API keys and database URIs:

```bash
# AI Providers
GENERATION_BACKEND="OPENROUTER"
GENERATION_MODEL_ID="openai/gpt-oss-120b" # or anthropic/claude-3.5-sonnet
QUESTION_GENERATION_BACKEND="OPENROUTER"
QUESTION_GENERATION_MODEL_ID="openai/gpt-4o"

# Embeddings
EMBEDDING_BACKEND="JINA" # or COHERE
EMBEDDING_MODEL_ID="jina-embeddings-v3"
EMBEDDING_MODEL_SIZE=1024

# Databases
MONGODB_URI="your_mongo_connection_string"
QDRANT_URL="your_qdrant_url"
```

### 3. Build and Run via Docker

To boot up the entire backend pipeline, run:

```bash
cd docker
sudo docker compose up -d --build
```

### 4. Monitor the Engine
Watch the real-time processing logs, including the custom AI Pipeline Debug Summaries:

```bash
sudo docker compose logs -f fastapi
```


## Project Structure (Key Directories)

- `/controllers/` - Contains core business logic (e.g., StructureController.py for dynamic AI parsing).

- `/models/` - Database schemas and Pydantic validation models.

- `/docker/` - Dockerfiles and docker-compose.yml configurations.

## Contributors
The Learnova platform was developed collaboratively as a Senior Capstone Project.

* Eslam Atia - AI Engineering
* Mazen Salah - AI Engineering

## Related Repositories
This repository contains the AI Backend Engine. The full Learnova platform is decoupled into separate repositories:
* **Frontend Application:** []
* **Backend Engine:** (This Repository)
