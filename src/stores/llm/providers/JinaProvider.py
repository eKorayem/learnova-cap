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
        Async batch embedding with Staggered Cascade and Dynamic Header Parsing.
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

        async def fetch_batch(client, batch, idx, max_retries=5):
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
                        
                    # RATE LIMITED: Read the headers for exact wait time!
                    elif response.status_code == 429:
                        # Check if Jina tells us exactly how long to wait
                        retry_after = response.headers.get("retry-after")
                        
                        if retry_after:
                            wait_time = float(retry_after)
                        else:
                            # Fallback: Fast Exponential Backoff (2s, 4s, 8s...) instead of 60s
                            wait_time = 2 * (2 ** retries)
                            
                        retries += 1
                        self.logger.warning(
                            f"⚠️ Jina 429 Limit! Sleeping exactly {wait_time}s... (Attempt {retries}/{max_retries})"
                        )
                        import asyncio
                        await asyncio.sleep(wait_time)
                        continue
                        
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
            import asyncio
            semaphore = asyncio.Semaphore(2)

            async def fetch_with_semaphore(batch, idx):
                # THE STAGGERED CASCADE: 
                # Batch 0 waits 0s, Batch 1 waits 0.5s, Batch 2 waits 1.0s...
                # This naturally spaces out the traffic so Jina doesn't get overwhelmed!
                await asyncio.sleep(idx * 0.5) 
                
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