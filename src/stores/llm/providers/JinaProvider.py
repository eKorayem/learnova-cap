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
        Async batch embedding — fires HTTP requests in parallel.
        Includes Smart Retry with 60-second backoff for Jina 429 Rate Limits.
        """
        if not texts:
            return []

        processed = [self.process_text(t) for t in texts]
        total = len(processed)
        batches = [
            processed[i:i + batch_size]
            for i in range(0, total, batch_size)
        ]

        self.logger.info(f"Jina: firing {len(batches)} parallel requests for {total} texts")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async def fetch_batch(client, batch, idx, max_retries=3):
            retries = 0
            while retries < max_retries:
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
                    
                    # SUCCESS: We got the vectors!
                    if response.status_code == 200:
                        vectors = [item["embedding"] for item in response.json()["data"]]
                        self.logger.info(f"Jina batch {idx + 1}/{len(batches)} ✓")
                        return vectors
                        
                    # RATE LIMITED: Jina's 100k TPM limit hit. Sleep and retry.
                    elif response.status_code == 429:
                        retries += 1
                        self.logger.warning(
                            f"⚠️ Jina Rate Limit (429) hit on batch {idx+1}! "
                            f"Sleeping 60 seconds... (Attempt {retries}/{max_retries})"
                        )
                        import asyncio
                        await asyncio.sleep(60)
                        continue # Restart the while loop
                        
                    # FATAL ERROR: 400 Bad Request, etc.
                    else:
                        self.logger.error(f"Jina batch {idx+1} error {response.status_code}: {response.text[:200]}")
                        return None
                        
                except Exception as e:
                    self.logger.error(f"Jina batch {idx+1} failed: {e}")
                    return None
                    
            self.logger.error(f"Jina batch {idx+1} failed completely after {max_retries} retries.")
            return None

        async with httpx.AsyncClient(timeout=60) as client:
            # Free tier limits to 2 concurrent requests
            import asyncio
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

    # =================================================================
    # RESTORED: The required Abstract Method that prevented startup
    # =================================================================
    async def generate_structured_response(self, system_prompt: str, user_prompt: str, response_schema: dict, temperature: float = None):
        raise NotImplementedError("JinaProvider does not support text generation.")