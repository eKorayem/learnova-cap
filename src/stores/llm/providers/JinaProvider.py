from ..LLMInterface import LLMInterface
import logging
import asyncio
import httpx


class JinaProvider(LLMInterface):

    JINA_BASE_URL = "https://api.jina.ai/v1"

    def __init__(self, api_key: str,
                 defualt_input_max_characters: int = 50000,
                 defualt_generation_max_out_tokens: int = None,
                 default_generation_temperature: float = None):

        self.api_key = api_key
        self.defualt_input_max_characters = defualt_input_max_characters
        self.embedding_model_id = None
        self.embedding_size = None
        self.enums = None
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.logger.warning("JinaProvider does not support text generation.")

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = int(embedding_size)
        self.logger.info(f"Jina embedding model set: {model_id} (size={embedding_size})")

    def process_text(self, text: str):
        return text[:self.defualt_input_max_characters + 1].strip()

    def generate_text(self, prompt: str, chat_history: list = [],
                      max_output_tokens: int = None,
                      temperature: float = None):
        raise NotImplementedError("JinaProvider does not support text generation.")

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": self.process_text(prompt)}

    def embed_text(self, text: str, document_type: str = None):
        """
        Sync single embedding — used for RAG search queries only.
        Single call so sync is fine here.
        """
        import requests
        try:
            response = requests.post(
                f"{self.JINA_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.embedding_model_id,
                    "input": [self.process_text(text)],
                    "dimensions": self.embedding_size
                },
                timeout=30
            )
            if response.status_code != 200:
                self.logger.error(f"Jina embed_text error: {response.text[:200]}")
                return None
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            self.logger.error(f"Jina embed_text failed: {e}")
            return None

    async def embed_batch_async(self, texts: list, batch_size: int = 100):
        """
        Async batch embedding — fires all HTTP requests in parallel.
        Must be called with 'await' from async context.
        3612 chunks → 37 parallel requests → ~3-5 seconds total.
        """
        if not texts:
            return []

        processed = [self.process_text(t) for t in texts]
        total = len(processed)
        batches = [
            processed[i:i + batch_size]
            for i in range(0, total, batch_size)
        ]

        self.logger.info(
            f"Jina: firing {len(batches)} parallel requests for {total} texts"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async def fetch_batch(client, batch, idx):
            try:
                response = await client.post(
                    f"{self.JINA_BASE_URL}/embeddings",
                    headers=headers,
                    json={
                        "model": self.embedding_model_id,
                        "input": batch,
                        "dimensions": self.embedding_size
                    }
                )
                if response.status_code != 200:
                    self.logger.error(
                        f"Jina batch {idx} error {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                    return None
                vectors = [item["embedding"] for item in response.json()["data"]]
                self.logger.info(f"Jina batch {idx + 1}/{len(batches)} ✓")
                return vectors
            except Exception as e:
                self.logger.error(f"Jina batch {idx} failed: {e}")
                return None

        async with httpx.AsyncClient(timeout=60) as client:
            # Free tier allows only 2 concurrent requests
            # Semaphore limits parallelism without losing async benefits
            semaphore = asyncio.Semaphore(2)

            async def fetch_with_semaphore(batch, idx):
                async with semaphore:
                    return await fetch_batch(client, batch, idx)

            results = await asyncio.gather(*[
                fetch_with_semaphore(batch, idx)
                for idx, batch in enumerate(batches)
            ])

        if any(r is None for r in results):
            self.logger.error("One or more Jina batches failed")
            return None

        all_vectors = []
        for batch_vectors in results:
            all_vectors.extend(batch_vectors)

        self.logger.info(f"Jina complete — {len(all_vectors)} vectors")
        return all_vectors