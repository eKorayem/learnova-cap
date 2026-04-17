from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient

from routes import base, data, nlp, ai
from routes.question import question_router
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from utils.metrics import setup_metrics

app = FastAPI()

# Setup Prometheus metrics
setup_metrics(app)

# CORS middleware - configure based on your frontend needs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def startup_span():
    settings = get_settings()

    app.settings = settings
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]

    llm_provider_factory = LLMProviderFactory(settings)
    vectordb_provider_factory = VectorDBProviderFactory(settings)

    # ── RAG: Generation Client ──────────────────────────────────
    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    # ── RAG: Embedding Client ───────────────────────────────────
    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE
    )

    # ── VectorDB Client ─────────────────────────────────────────
    app.vectordb_client = vectordb_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND,
    )
    app.vectordb_client.connect()

    # ── Template Parser ─────────────────────────────────────────
    app.template_parser = TemplateParser(
        language=settings.PRIMARY_LAN,
        default_language=settings.DEFAULT_LAN
    )

    # ── Question Generation Client (separate from RAG) ──────────
    app.question_generation_client = llm_provider_factory.create(
        provider=settings.QUESTION_GENERATION_BACKEND
    )
    app.question_generation_client.set_generation_model(
        model_id=settings.QUESTION_GENERATION_MODEL_ID
    )

async def shutdown_span():
    app.mongo_conn.close()
    app.vectordb_client.disconnect()


app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)

# Registering Routers
app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
app.include_router(ai.ai_router)
app.include_router(question_router)