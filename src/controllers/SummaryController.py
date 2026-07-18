import json
import logging
import asyncio
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from controllers.BaseController import BaseController
from models.ChunkModel import ChunkModel
from stores.llm.LLMInterface import LLMInterface
from routes.schemas.summary import SummaryPayload

class SummaryController(BaseController):

    def __init__(self, generation_client: LLMInterface):
        super().__init__()
        self.generation_client = generation_client
        self.logger = logging.getLogger('uvicorn.error')

    async def generate_summary(
        self,
        chunk_model: ChunkModel,
        project_id: str,
        target_sections: List[Dict[str, Any]],
        summary_type: str,
        asset_id: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        
        self.logger.info(f"Generating '{summary_type}' summary for project {project_id}")
        compiled_texts = []
        
        # 1. Fetch targeted page boundaries or match keywords via the retrieval engine
        for section in target_sections:
            text = await self._get_section_content(
                chunk_model=chunk_model, 
                project_id=project_id, 
                topic_title=section.get("topic_title"), 
                page_start=section.get("page_start"), 
                page_end=section.get("page_end"),
                asset_id=asset_id
            )
            if text:
                compiled_texts.append(f"--- SECTION: {section.get('topic_title')} ---\n{text}")

        if not compiled_texts:
            self.logger.error("Summary Generation Failed: No content could be retrieved.")
            return None

        combined_document = "\n\n".join(compiled_texts)
        
        # 2. Enforce structural context limit
        if len(combined_document) > 80000:
            self.logger.warning("Document exceeds strict character budget. Truncating context bounds.")
            combined_document = combined_document[:80000] + "\n\n...[CONTENT TRUNCATED FOR LENGTH]"

        system_prompt = self._build_system_prompt(summary_type)
        
        # 3. Fire Structured Call to LLM with a 3-Attempt Retry Loop
        llm_response = None
        for attempt in range(3):
            try:
                llm_response = await self.generation_client.generate_structured_response(
                    system_prompt=system_prompt,
                    user_prompt=f"COURSE MATERIAL:\n\n{combined_document}",
                    response_schema=self._pydantic_schema(SummaryPayload),
                    temperature=0.2,
                    max_output_tokens=8000
                )

                # If we get a valid, reasonably sized response, break the loop
                if llm_response and len(llm_response.strip()) > 20:
                    break
                
                self.logger.warning(f"Attempt {attempt + 1}: LLM returned empty or truncated response. Retrying...")
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} API Error: {e}")
                await asyncio.sleep(2)

        if not llm_response or len(llm_response.strip()) <= 20:
            self.logger.error("Summary Generation Failed: LLM continually returned invalid data after 3 attempts.")
            return None

        # 4. Parse and return
        try:
            parsed_payload = self._load_structured_payload(llm_response)
            if not parsed_payload:
                return None

            return parsed_payload.model_dump(exclude_none=True)

        except Exception as e:
            self.logger.error(f"Error executing summary generation parsing: {e}")
            return None

    # =============================================================
    # HYBRID RETRIEVAL MECHANISM
    # =============================================================

    async def _get_section_content(
        self, chunk_model: ChunkModel, project_id: str, 
        topic_title: str, page_start: Optional[int], page_end: Optional[int],
        asset_id: Optional[Any]
    ) -> Optional[str]:
        
        chunks = await chunk_model.get_chunks_by_project_id(
            project_id=project_id,
            page_no=1,
            page_size=5000, 
            chunk_type="structure",
            asset_id=asset_id
        )

        if not chunks:
            return None

        chunks = sorted(chunks, key=lambda c: c.chunk_order)
        relevant_chunks = []

        if page_start is not None and page_end is not None:
            for chunk in chunks:
                chunk_page = chunk.chunk_metadata.get("page")
                if chunk_page is not None:
                    actual_page = int(chunk_page) + 1  
                    if page_start <= actual_page <= page_end:
                        relevant_chunks.append(chunk)

        if not relevant_chunks:
            topic_keywords = [word.lower() for word in str(topic_title).split() if len(word) > 3]
            if topic_keywords:
                scored_chunks = []
                for chunk in chunks:
                    chunk_lower = chunk.chunk_text.lower()
                    score = sum(1 for keyword in topic_keywords if keyword in chunk_lower)
                    if score > 0:
                        scored_chunks.append((score, chunk))

                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                relevant_chunks = [item[1] for item in scored_chunks[:15]]

        if not relevant_chunks:
            return None

        relevant_chunks = sorted(relevant_chunks, key=lambda c: c.chunk_order)
        full_text = "\n\n".join([c.chunk_text for c in relevant_chunks])
        
        return full_text[:40000]

    # =============================================================
    # PROMPTS & PARSERS
    # =============================================================

    def _build_system_prompt(self, summary_type: str) -> str:
        return f"""You are an expert academic summarizer. 
Synthesize the provided course material into a comprehensive, structured study guide.
The desired output style is: '{summary_type}'.

CRITICAL RULES:
1. Provide a single overarching 'title' at the JSON root level.
2. Write a clear, holistic 'executive_summary' covering the main theme of the provided text.
3. For each section provided in the text, generate a 'SummarySection' object containing a detailed 'content' paragraph and a 'key_points' array containing the most vital takeaways.
4. Output ONLY valid JSON matching the schema exactly.
5. YOU MUST RETURN THE COMPLETE JSON OBJECT. Do not stop early. Do not wrap your response in markdown fences.
"""

    def _load_structured_payload(self, response: str) -> Optional[SummaryPayload]:
        cleaned = response.strip()
        if cleaned.startswith("```json"): cleaned = cleaned[7:]
        elif cleaned.startswith("```"): cleaned = cleaned[3:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]
        
        try:
            return SummaryPayload(**json.loads(cleaned.strip()))
        except Exception as e:
            self.logger.error(f"Summary payload parsing error: {e}")
            return None

    def _pydantic_schema(self, model_cls) -> Dict[str, Any]:
        return model_cls.model_json_schema()