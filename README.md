# Learnova AI Core Microservice

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=flat&logo=docker&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=flat&logo=mongodb&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-orange)

## Overview

This repository contains the **AI Core Microservice Engine** for the **Learnova Smart Study Companion Platform**. The microservice is built using **FastAPI** to handle the system's heavy LLM processing, deep document parsing, structural context decomposition, and semantic vector operations. 

The main Learnova platform repository, which houses both the frontend user interfaces and the central monolithic business web application backend, can be accessed here:

* **Learnova Main Repository (Frontend & Backend):** [Repo](https://github.com/EslamMDahy/Learnova-Smart-Study-Companion)
* **Project Video Demonstration:** [Video](https://www.youtube.com/watch?v=KYJVsY2uzZ0)

---

## Architectural Topology & Component Flow

The engine applies a fully decoupled **Layered Abstract Architecture** leveraging the **Factory and Adapter Design Patterns**. This completely isolates endpoint routes from core model execution blocks, allowing zero-downtime hot-swapping between API provider backends (e.g., OpenRouter, Groq, local SentenceTransformers).

### The Big Picture

![System Architecture](Assets/Arch.png)

### System Infrastructure Graph

![alt text](Assets/SystemInfrastructure.png)


## Cryptographic Integration Security

To secure the academic workflows, communications between the central monolithic backend and this AI container utilize a strict **HMAC SHA256 Signature Verification Layer**. This is managed natively via FastAPI dependencies.

![System Architecture](Assets/Crypto.png)

The matching verification payload is calculated deterministically to prevent man-in-the-middle tampering:

```text
Canonical String = HTTP_METHOD + "\n" + URL_PATH + "\n" + REQUEST_ID + "\n" + TIMESTAMP + "\n" + SHA256_BODY_HASH
```

---

## Core Feature Engines

### 1. Document Structure Tree Parser (`StructureController`)

Analyzes textbook material structures to output organized structural maps.

* **The TOC Assassin:** Programmatically evaluates text fragments to detect and drop pre-existing Table of Contents indices, forcing models to parse authentic textbook nodes.


* **Heuristic Heading Analysis:** Filters document visual markers using strict regex definitions, cleaning out non-structural noise, syntax configurations, and prose page markers.


* **Parallel Outlining:** Splits headings into micro-chunks (`MAX_LLM_INPUT_CHARS_PER_BATCH: 10000`), processes them concurrently using `asyncio.gather`, and joins individual JSON layers into a unified parent-child outline.



### 2. Multi-Tier Spatial Chunking Strategy

Instead of applying uniform splitting constraints across all tasks, files are broken up into three isolated data streams:

* **RAG Collections (`RAG_CHUNK_SIZE: 100`)**: Small, precise textual facts to ensure highly accurate chatbot interactions while minimizing LLM hallucinations.


* **Structure Maps (`STRUCTURE_CHUNK_SIZE: 1000`)**: Mid-size structural windows optimized for hierarchical document layout analysis.


* **Question Contexts (`QUESTION_CHUNK_SIZE: 1500`)**: Broad context lengths that provide deep concept descriptions to help models generate accurate questions and detailed grading rubrics.



### 3. Automated Assessment Builder (`QuestionGenerationController`)

Generates comprehensive question sets across objective (MCQ, True/False) and subjective (Short Answer, Essay) formats. It is bounded by structural validation scripts to check answer options and remove ungradable questions before firing callbacks.


### 4. In-Line Question Extraction  (`ExtractionController`)
Scans educational literature utilizing clustering heuristics to find authentic, verbatim review exercises and assignment blocks natively embedded in textbooks, while actively skipping rhetorical prose.

### 5. Rigid AI Grading Engine (`GradingController`)
Grades subjective student essay answers and short responses strictly against generated structural rubrics. It features an edge-case block that catches blank inputs or prompt injection attempts, calculating scores based purely on clear evidence maps.  

---

## Project Structure

```text
├── src/
│   ├── main.py                 # Core engine initialization, state persistence & router registration
│   ├── controllers/            # Domain logic and pipeline orchestration engines
│   │   ├── BaseController.py   # Sandboxed disk pathing and shared utility methods
│   │   ├── DataController.py   # Cryptographic file system storage allocation & sanitization
│   │   ├── NLPController.py    # Dual batch indexing drivers & RAG context assembly loops
│   │   ├── StructureController.py # Table of Contents outline generation pipelines
│   │   ├── QuestionGenerationController.py # High-tier evaluation question builder
│   │   ├── ExtractionController.py # Native book exam scanner & prompt cleaner
│   │   └── GradingController.py # Strict, prompt-injection shielded evaluation engine
│   ├── core/
│   │   └── security/           # HMAC validation, payload serialization and dependencies
│   ├── helpers/
│   │   └── config.py           # Strictly typed environment variables matching via Pydantic
│   ├── models/                 # Database managers and async MongoDB ODM records tracking
│   │   └── db_schemas/         # Pydantic representations for Projects, Assets, and DataChunks
│   ├── routes/                 # FastAPI HTTP entry endpoints and asynchronous workers mapping
│   │   └── schemas/            # Request/Response data validation envelopes
│   ├── stores/
│   │   ├── llm/                # Abstract provider interface & custom integration classes
│   │   └── vectordb/           # Qdrant client adapter drivers
│   └── utils/
│       └── metrics.py          # Prometheus ASGI request/duration tracker middleware
└── docker/                     # Infrastructure deployment & multi-container orchestration envs

```

---

## Getting Started (Local Development)

### Prerequisites

* Docker Engine & **Docker Compose V2** plugin installed.


* Git.



### 1. Project Initialization

```bash
git clone [https://github.com/your-username/learnova-cap-ai.git](https://github.com/your-username/learnova-cap-ai.git)
cd learnova-cap-ai

```

### 2. Environment Configuration

Create a production configuration file at `src/.env` or adapt `docker/env/.env.app`. Ensure no trailing spaces are left inside the configuration values:

```env
APP_NAME="learnova-cap"
APP_VERSION="0.1"

FILE_ALLOWED_TYPES=["text/plain", "application/pdf"]
FILE_ALLOWED_SIZE=100
FILE_DEFAULT_CHUNK_SIZE=512000

MONGODB_URL="mongodb://admin:admin@mongodb:27017"
MONGODB_DATABASE="learnova-cap"

# Vector Store Drivers
VECTOR_DB_BACKEND="QDRANT"
VECTOR_DB_URL="http://qdrant:6333"
VECTOR_DB_DISTANCE_METHOD="cosine"

# AI Model Orchestration Infrastructure
EMBEDDING_BACKEND="JINA"
EMBEDDING_MODEL_ID="jina-embeddings-v3"
EMBEDDING_MODEL_SIZE=1024

STRUCTURE_BACKEND="OPENROUTER"
STRUCTURE_MODEL_ID="google/gemini-2.5-flash-lite"

QUESTION_GENERATION_BACKEND="OPENROUTER"
QUESTION_GENERATION_MODEL_ID="google/gemini-2.5-flash"

GRADING_BACKEND="OPENROUTER"
GRADING_MODEL_ID="google/gemini-2.5-flash"

GENERATION_BACKEND="OPENROUTER"
GENERATION_MODEL_ID="openai/gpt-4o-mini"

# Platform Gateway Secrets
AI_SHARED_SECRET="YOUR_SECURE_HMAC_SECRET_KEY_HERE"
LEARNOVA_BACKEND_URL="[https://www.learnova-edu.com/api/](https://www.learnova-edu.com/api/)"

# API Provider Access Credentials
GROQ_API_KEY="gsk_..."
DEEPSEEK_API_KEY="sk-..."
JINA_API_KEY="jina_..."
OPENROUTER_API_KEY="sk-or-v1-..."

```

### 3. Build & Run Application Container

Deploy the container system using Docker Compose V2 from the docker directory:

```bash
cd docker
docker compose --env-file env/.env.app up -d --build
```

### 4. Live Log Monitoring

Monitor background processing workers and check prompt validation text formats using live logging:

```bash
docker compose logs -f fastapi
```

---

## System Health & Metrics

* **Prometheus Metrics Endpoint:** `GET /Elkhalas_3shan_khalas`

* **System Inbound Health Check:** `GET /api/v1/health`


## Contributors

The Learnova platform was developed collaboratively as a Senior Capstone Project.

- **AI Microservice Engineering**

    * [Eslam Atia](https://github.com/eKorayem) - AI Infrastructure & Systems Microservice Engineering
    * Mazen Salah - Prompt Engineering

- **Technical Leader & Frontend Engineering**: [Eslam Dahy](https://github.com/EslamMDahy)


- **Main Backend Engineering**: [Ahmed Waheed](https://github.com/Waheed7000)

- **UI/UX Designer**: [Khaled Khodary](https://github.com/ikhalled)

- **QA & Testing**: [Farouk Mohsen](https://github.com/faroukmohsen)


---

*Learnova Platform - Decoupled AI Core Engine Framework 2026.*
