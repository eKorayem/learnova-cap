import json
import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from controllers.BaseController import BaseController
from models.ChunkModel import ChunkModel
from stores.llm.LLMInterface import LLMInterface
from routes.schemas.presentation import GeneratedSlide, PresentationPayload

class PresentationController(BaseController):

    # Validated frontend visual layouts engine maps
    VALID_LAYOUTS = {
        "title_slide", "lecture_objectives", "section_divider", "concept_explanation",
        "text_with_image", "full_image", "key_points", "comparison", "process_steps",
        "timeline", "diagram", "table", "equation_explanation", "equation_derivation",
        "summary", "references", "adaptive_cards", "single_card_center",
        "two_card_horizontal", "three_card_horizontal"
    }

    def __init__(self, generation_client: LLMInterface):
        super().__init__()
        self.generation_client = generation_client
        self.logger = logging.getLogger('uvicorn.error')

    async def generate_presentation(
        self,
        chunk_model: ChunkModel,
        project_id: str,
        target_sections: List[Dict[str, Any]],
        slide_count: int,
        asset_id: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Orchestrates compiling text across multi-topic sections and orchestrating structured slide generation.
        """
        self.logger.info(f"Generating {slide_count}-slide presentation for project {project_id}")
        compiled_texts = []
        
        # 1. Sequentially fetch targeted page boundaries or match keywords via the retrieval engine
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
            self.logger.error("Presentation Generation Failed: No content could be retrieved from database states.")
            return None

        combined_document = "\n\n".join(compiled_texts)
        
        # 2. Enforce structural microservice cap limit to protect the input context window
        if len(combined_document) > 80000:
            self.logger.warning("Document exceeds strict character budget. Truncating context bounds safely.")
            combined_document = combined_document[:80000] + "\n\n...[CONTENT TRUNCATED FOR LENGTH]"

        system_prompt = self._build_system_prompt(slide_count)
        
        # 3. Fire Structured Call to the LLM Gateway
        try:
            llm_response = await self.generation_client.generate_structured_response(
                system_prompt=system_prompt,
                user_prompt=f"COURSE MATERIAL:\n\n{combined_document}",
                response_schema=self._pydantic_schema(PresentationPayload),
                temperature=0.3,
                max_output_tokens=8000
            )

            if not llm_response:
                self.logger.error("Structured generation response returned empty data.")
                return None

            parsed_payload = self._load_structured_payload(llm_response)
            if not parsed_payload:
                return None

            # 4. Return complete schema matching the global root contract definition
            return {
                "title": str(parsed_payload.title),
                "slides": [s.model_dump(exclude_none=True) for s in parsed_payload.slides]
            }

        except Exception as e:
            self.logger.error(f"Error executing presentation layout compilation sequence: {e}")
            return None

    # =============================================================
    # HYBRID RETRIEVAL MECHANISM
    # =============================================================

    async def _get_section_content(
        self, chunk_model: ChunkModel, project_id: str, 
        topic_title: str, page_start: Optional[int], page_end: Optional[int],
        asset_id: Optional[Any]
    ) -> Optional[str]:
        """
        Retrieves database document chunks strictly targeted by internal asset limits and page boundaries.
        Falls back seamlessly to multi-token keyword scoring matrices if metadata fields arrive null.
        """
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

        # Strategy A: Strict Page Range Allocation Matcher
        if page_start is not None and page_end is not None:
            for chunk in chunks:
                chunk_page = chunk.chunk_metadata.get("page")
                if chunk_page is not None:
                    actual_page = int(chunk_page) + 1  
                    if page_start <= actual_page <= page_end:
                        relevant_chunks.append(chunk)

        # Strategy B: Linguistic Keyword Scoring Fallback Window
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
    # PROMPT ARCHITECTURE & PARSERS
    # =============================================================

    def _build_system_prompt(self, slide_count: int) -> str:
        return f"""You are an expert academic presentation designer. 
Synthesize the provided course material into exactly {slide_count} slides.
Output ONLY valid JSON matching the schema provided.

CRITICAL ROOT REQUIREMENT:
You must provide a single overarching 'title' at the JSON root level that captures the core subject of the entire slide deck.

CRITICAL MAPPING RULES:
You must choose a 'layout_type' for each individual slide and populate the EXACT top-level fields required for that layout. Do NOT put layout-specific fields inside the 'content' object unless specified.

1. "title_slide": Requires slide-level 'course', 'instructor', 'term', 'visual'.
2. "lecture_objectives": Requires slide-level 'outcome' (string) and 'objectives' (array of strings).
3. "section_divider": Requires slide-level 'section_number', 'descriptor', 'subtitle'.
4. "concept_explanation": Requires slide-level 'content' object containing (lead, body, bullets, key_term, definition, example).
5. "text_with_image": Requires slide-level 'content' object AND slide-level 'visual' object.
6. "full_image": Requires slide-level 'visual' object.
7. "key_points": Requires slide-level 'overview' (string) and slide-level 'cards' array.
8. "comparison": Requires slide-level 'comparison' object (with 'left' and 'right' objects).
9. "process_steps": Requires slide-level 'overview' (string) and slide-level 'process' array.
10. "timeline": Requires slide-level 'overview' (string) and slide-level 'timeline' array.
11. "diagram": Requires slide-level 'overview' (string) and slide-level 'diagram' object (nodes and edges).
12. "table": Requires slide-level 'overview' (string) and slide-level 'table' object (headers and rows).
13. "equation_explanation": Requires slide-level 'equation' object and slide-level 'content' object.
14. "equation_derivation": Requires slide-level 'overview' (string) and slide-level 'equation' object (with steps).
15. "summary": Requires slide-level 'summary' object (points, takeaway, next).
16. "references": Requires slide-level 'references' array of strings.
17. "adaptive_cards" / "single_card_center" / "two_card_horizontal" / "three_card_horizontal": Requires slide-level 'cards' array.

Every slide MUST have a 'slide_number', 'layout_type', 'kicker', and 'title'.
Do not wrap your response in markdown fences. Return raw JSON text.
"""

    def _load_structured_payload(self, response: str) -> Optional[PresentationPayload]:
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
            
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        cleaned_response = cleaned_response.strip()

        try:
            return PresentationPayload(**json.loads(cleaned_response))
        except Exception as e:
            self.logger.error(f"Payload structural verification parsing error: {e}")
            return None

    def _pydantic_schema(self, model_cls) -> Dict[str, Any]:
        return model_cls.model_json_schema()