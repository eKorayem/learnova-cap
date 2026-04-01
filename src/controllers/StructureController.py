from .BaseController import BaseController
from models.ChunkModel import ChunkModel
from typing import List
import json
import logging


class StructureController(BaseController):
    
    def __init__(self, generation_client):
        super().__init__()
        self.generation_client = generation_client
        self.logger = logging.getLogger(__name__)
    
    
    
    
    async def analyze_lecture_structure(
        self,
        chunk_model: ChunkModel,
        project_id: str,
        max_topics: int = None,
        use_all_chunks: bool = False
    ):
        """
        Analyze chunks and extract topics/subtitles structure

        Key improvements:
        1. Reconstructs continuous text from chunks (preserves topic boundaries)
        2. Smart truncation at chunk boundaries (not mid-topic)
        3. Optional full-document mode for better accuracy
        """

        # 1. Get all chunks
        chunks = await chunk_model.get_chunks_by_project_id(
            project_id=project_id,
            page_no=1,
            page_size=1000
        )

        if not chunks:
            self.logger.error(f"No chunks found for project {project_id}")
            return None

        # Sort by chunk_order to ensure correct sequence
        chunks = sorted(chunks, key=lambda c: c.chunk_order)

        total_chunks = len(chunks)
        self.logger.info(f"Analyzing {total_chunks} chunks")

        # 2. Select chunks - prioritize continuity over coverage
        if use_all_chunks or total_chunks <= 50:
            # Use all chunks for short-medium documents
            selected_chunks = chunks
            self.logger.info(f"Using all {len(selected_chunks)} chunks")
        else:
            # For long documents: take continuous blocks, not scattered samples
            # This preserves topic boundaries and context
            max_chunks = 50  # Limit to avoid token overflow

            # Strategy: Take first 60% + last 40% as two continuous blocks
            # This captures intro/topics + conclusion/summary
            first_block_size = int(max_chunks * 0.6)
            last_block_size = max_chunks - first_block_size

            selected_chunks = chunks[:first_block_size] + chunks[-last_block_size:]
            self.logger.info(f"Selected {len(selected_chunks)} chunks (first {first_block_size} + last {last_block_size})")

        # 3. Reconstruct continuous text from chunks
        # Join with minimal separator to preserve original flow
        full_text = "\n".join([c.chunk_text for c in selected_chunks])

        # 4. Smart truncation at paragraph/chunk boundaries
        max_chars = self.app_settings.INPUT_DAFAULT_MAX_CHARACTERS
        if len(full_text) > max_chars:
            # Find last complete paragraph before limit
            truncated = full_text[:max_chars]
            last_break = truncated.rfind("\n\n")
            if last_break > max_chars * 0.8:  # Only if we don't lose too much
                full_text = truncated[:last_break]
            else:
                full_text = truncated
            self.logger.info(f"Truncated text to {len(full_text)} characters at boundary")

        self.logger.info(f"Final text length: {len(full_text)} chars from {len(selected_chunks)} chunks")

        # 5. Build prompt and analyze
        prompt = self._build_structure_prompt(full_text, max_topics)

        try:
            response = self.generation_client.generate_text(
                prompt=prompt,
                temperature=self.app_settings.GENERATION_DAFAULT_TEMPERATURE,
                max_output_tokens=self.app_settings.GENERATION_DAFAULT_MAX_TOKENS
            )

            structure = self._parse_structure_response(response)

            if structure and "topics" in structure:
                self.logger.info(f"Successfully extracted {len(structure['topics'])} topics")

            return structure

        except Exception as e:
            self.logger.error(f"Error analyzing structure: {e}")
            return None
    
    def _build_structure_prompt(self, text: str, max_topics: int=None) -> str:
        """Build prompt for structure analysis"""

        # Build max_topics rule if specified
        max_topics_rule = f"Limit to {max_topics} topics maximum.\n" if max_topics else ""

        # Improved prompt with clearer instructions and examples
        prompt = f"""You are a document structure extractor. Extract topics and subtitles from the document.

RULES:
1. Output ONLY valid JSON with this exact structure:
{{
  "topics": [
    {{
      "title": "Topic name from document",
      "order": 1,
      "subtitles": [
        {{"title": "Subtitle 1", "order": 1}},
        {{"title": "Subtitle 2", "order": 2}}
      ]
    }}
  ]
}}

2. "topics" = main section headings (H1/H2 level)
3. "subtitles" = direct child headings under each topic (H3 level)
4. Use exact wording from the document
5. Preserve document order
6. {max_topics_rule}7. NO markdown, NO explanations, NO code blocks - JSON ONLY
8. If no clear subtitles exist, use empty array: "subtitles": []

Document content:
{text}

JSON output:"""
        return prompt
    
    def _parse_structure_response(self, response: str) -> dict:
        response = response.strip()
        
        print("Raw LLM response:", response)
        
        # Remove markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            # Find content between first ``` and last ```
            parts = response.split("```")
            if len(parts) >= 3:
                response = parts[1]  # Content between first and second ```
            else:
                response = response.replace("```", "")
        
        response = response.strip()
        
        self.logger.info(f"After markdown removal: {response[:200]}")
        
        # Determine if it's an object {...} or array [...]
        is_object = response.strip().startswith("{")
        is_array = response.strip().startswith("[")
        
        # Extract JSON
        if is_object:
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                response = response[start_idx:end_idx]
        elif is_array:
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            if start_idx != -1 and end_idx > start_idx:
                response = response[start_idx:end_idx]
        
        response = response.strip()
        
        self.logger.info(f"Final JSON to parse: {response[:200]}")
        
        try:
            structure = json.loads(response)
            
            # If it's an array, wrap it
            if isinstance(structure, list):
                structure = {"topics": structure}
            
            # Validate
            if "topics" not in structure:
                self.logger.error("Response missing 'topics' field")
                return self._create_fallback_structure("")
            
            self.logger.info(f"Successfully parsed structure with {len(structure['topics'])} topics")
            return structure
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse failed: {e}")
            self.logger.error(f"Attempted to parse: {response[:500]}")
            return self._create_fallback_structure(response)

    def _create_fallback_structure(self, text: str) -> dict:
        """Create a simple fallback structure when JSON parsing fails"""
        return {
            "topics": [
                {
                    "title": "Document Content",
                    "order": 1,
                    "subtitles": [
                        {"title": "Main Content", "order": 1}
                    ]
                }
            ]
        }