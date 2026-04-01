"""
Debug script for structure analysis testing.
Run this to test different configurations and see what's happening.
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv('src/.env')

from motor.motor_asyncio import AsyncIOMotorClient
from models.ChunkModel import ChunkModel
from stores.llm.LLMProviderFactory import LLMProviderFactory
from controllers.StructureController import StructureController

async def test_structure_analysis():
    # 1. Connect to MongoDB
    mongo_conn = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
    db_client = mongo_conn[os.getenv("MONGODB_DATABASE")]

    # 2. Get chunk model
    chunk_model = await ChunkModel.create_instance(db_client=db_client)

    # 3. Create LLM client
    from helpers.config import get_settings
    settings = get_settings()

    llm_factory = LLMProviderFactory(
        GENERATION_BACKEND=settings.GENERATION_BACKEND,
        GROQ_API_KEY=settings.GROQ_API_KEY,
        OPENAI_API_KEY=settings.OPENAI_API_KEY,
        OPENAI_API_URL=settings.OPENAI_API_URL,
        INPUT_DAFAULT_MAX_CHARACTERS=settings.INPUT_DAFAULT_MAX_CHARACTERS,
        GENERATION_DAFAULT_MAX_TOKENS=settings.GENERATION_DAFAULT_MAX_TOKENS,
        GENERATION_DAFAULT_TEMPERATURE=settings.GENERATION_DAFAULT_TEMPERATURE,
    )

    generation_client = llm_factory.create(provider=os.getenv("GENERATION_BACKEND"))
    generation_client.set_generation_model(model_id=os.getenv("GENERATION_MODEL_ID"))

    # 4. Test with a project
    project_id = input("Enter project_id to test: ").strip()

    print(f"\n=== Fetching chunks for project: {project_id} ===")
    chunks = await chunk_model.get_chunks_by_project_id(project_id=project_id, page_no=1, page_size=1000)

    if not chunks:
        print("ERROR: No chunks found!")
        return

    print(f"Found {len(chunks)} chunks")
    print(f"Average chunk size: {sum(len(c.chunk_text) for c in chunks) / len(chunks):.0f} chars")

    # 5. Show first few chunks
    print("\n=== First 3 chunks (raw) ===")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} (order={chunk.chunk_order}, len={len(chunk.chunk_text)}) ---")
        print(chunk.chunk_text[:200] + "..." if len(chunk.chunk_text) > 200 else chunk.chunk_text)

    # 6. Test structure analysis
    print("\n=== Running Structure Analysis ===")
    controller = StructureController(generation_client=generation_client)

    structure = await controller.analyze_lecture_structure(
        chunk_model=chunk_model,
        project_id=project_id,
        max_topics=10
    )

    if structure:
        print("\n=== RESULT ===")
        import json
        print(json.dumps(structure, indent=2))
    else:
        print("\n=== RESULT: FAILED (returned None) ===")

    # Cleanup
    mongo_conn.close()

if __name__ == "__main__":
    asyncio.run(test_structure_analysis())
