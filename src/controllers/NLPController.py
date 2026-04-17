from .BaseController import BaseController
from models.db_schemas import Project, DataChunk
from typing import List
from stores.llm.LLMEnums import DocumentTypeEnums
import json
import logging


class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client, embedding_client, template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser

        self.logger = logging.getLogger(__name__)

    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}"

    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)

    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)
        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        ) if collection_info else None

    async def index_into_vector_db(self, project: Project,
                             chunks: List[DataChunk],
                             chunks_ids: List[int],
                             do_reset: bool = False):

        collection_name = self.create_collection_name(project_id=project.project_id)

        texts = [chunk.chunk_text for chunk in chunks]
        metadatas = [chunk.chunk_metadata for chunk in chunks]

        # ── BATCH EMBEDDING ────────────────────────────────────────────────
        if hasattr(self.embedding_client, 'embed_batch_async'):
            # Jina async provider — fire all requests in parallel
            self.logger.info(f"Using Jina async batch embedding for {len(texts)} chunks")
            vectors = await self.embedding_client.embed_batch_async(texts)

        elif hasattr(self.embedding_client, 'embed_batch'):
            # SentenceTransformer local GPU provider
            EMBED_BATCH_LIMIT = 2000
            total_texts = len(texts)
            vectors = []
            self.logger.info(f"Batch embedding {total_texts} chunks in blocks of {EMBED_BATCH_LIMIT}")

            for i in range(0, total_texts, EMBED_BATCH_LIMIT):
                end = i + EMBED_BATCH_LIMIT
                block_vectors = self.embedding_client.embed_batch(texts[i:end])
                if not block_vectors:
                    self.logger.error(f"Embedding failed for block starting at {i}")
                    return False
                vectors.extend(block_vectors)

            self.logger.info(f"Embedding complete — {len(vectors)} vectors generated")

        else:
            # Fallback: sequential for Cohere, OpenAI etc.
            self.logger.info(f"Using sequential embedding for {len(texts)} chunks")
            vectors = [
                self.embedding_client.embed_text(
                    text=text,
                    document_type=DocumentTypeEnums.DOCUMENT.value
                )
                for text in texts
            ]
        # ───────────────────────────────────────────────────────────────────

        if not vectors or len(vectors) == 0:
            self.logger.error("Embedding failed — no vectors generated")
            return False

        import asyncio

        # Create collection if not exists (or reset if requested)
        # Wrap in to_thread since QdrantClient is synchronous
        await asyncio.to_thread(
            self.vectordb_client.create_collection,
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_rest=do_reset
        )

        # ── Insert into Qdrant in safe batches of 500 ─────────────────────
        # Wrapped in asyncio.to_thread so sync Qdrant doesn't block the
        # async event loop while Jina requests are still running.
        insert_batch_size = 500
        total = len(texts)

        for i in range(0, total, insert_batch_size):
            end = i + insert_batch_size
            self.logger.info(
                f"Inserting Qdrant batch {i // insert_batch_size + 1} "
                f"({i}-{min(end, total)} of {total})"
            )
            await asyncio.to_thread(
                self.vectordb_client.insert_many,
                collection_name,
                texts[i:end],
                vectors[i:end],
                metadatas[i:end],
                chunks_ids[i:end]
            )
        # ───────────────────────────────────────────────────────────────────

        return True

    async def search_vector_db_collection(self, project: Project, text: str, limit: int = 5):
        import asyncio
        collection_name = self.create_collection_name(project_id=project.project_id)

        # Single query embedding — run in thread since embed_text is sync
        vector = await asyncio.to_thread(
            self.embedding_client.embed_text,
            text,
            DocumentTypeEnums.QUERY.value
        )

        if not vector or len(vector) == 0:
            return False

        search_results = await asyncio.to_thread(
            self.vectordb_client.search_by_vector,
            collection_name,
            vector,
            limit
        )

        if not search_results:
            return None

        return search_results

    async def answer_rag_question(self, project: Project, query: str, limit: int = 10):
        answer, full_prompt, chat_history = None, None, None

        # step1: Retrieve related documents
        retrieved_documents = await self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            return answer, full_prompt, chat_history

        # step2: Construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx,
                "chunk_text": doc.text
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

        # step3: Construct generation client prompts
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value
            )
        ]

        full_prompt = "\n\n".join([documents_prompts, footer_prompt])

        # Use async generation if available (Groq, OpenAI, etc.)
        if hasattr(self.generation_client, 'generate_text_async'):
            answer = await self.generation_client.generate_text_async(
                prompt=full_prompt,
                chat_history=chat_history,
            )
        else:
            # Fallback to sync (runs in thread pool)
            import asyncio
            answer = await asyncio.to_thread(
                self.generation_client.generate_text,
                prompt=full_prompt,
                chat_history=chat_history,
            )

        return answer, full_prompt, chat_history