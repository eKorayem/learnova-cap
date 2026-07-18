from models.ChunkModel import ChunkModel
from controllers.BaseController import BaseController
from typing import List, Optional, Dict, Any
import json
import re
import logging
import time
import asyncio

class StructureController(BaseController):
    """
    Universal Parser for academic document structure.
    Enforces strict 2-level hierarchy, chronological sorting, and deterministic page bounds.
    """

    def __init__(self, generation_client=None):
        super().__init__()
        self.generation_client = generation_client
        self.logger = logging.getLogger('uvicorn.error')
        self.logger.setLevel(logging.INFO)

        self.MAX_LLM_INPUT_CHARS_PER_BATCH = 15000
        self.MAX_STRUCTURE_BATCHES = getattr(self.app_settings, "STRUCTURE_MAX_BATCHES", 25)

    # =============================================================
    # 1. CORE ORCHESTRATOR
    # =============================================================

    async def analyze_lecture_structure(
        self,
        chunk_model: ChunkModel,
        project_id: str,
        max_topics: int = None,
        asset_id: str = None, 
        use_all_chunks: bool = False
    ) -> dict:
        
        start_time = time.time()
        self.logger.info(f"========== [STARTED] STRUCTURE EXTRACTION FOR {project_id} ==========")
        
        chunks = await chunk_model.get_chunks_by_project_id(
            project_id=project_id, page_no=1, page_size=5000, chunk_type="structure", asset_id=asset_id
        )

        if not chunks:
            self.logger.error("0 Chunks found. Aborting.")
            return self._create_fallback_structure()

        chunks = sorted(chunks, key=lambda c: c.chunk_order)
        last_doc_page = max([int(c.chunk_metadata.get('page', 0)) + 1 for c in chunks] + [1])
        
        # Step A: Quarantine the Table of Contents
        toc_pages = self._identify_toc_pages(chunks)
        self.logger.info(f"Quarantined TOC Pages: {sorted(list(toc_pages))}")

        # Step B: Reconstruct text and extract ONLY Headings
        text_blocks = []
        for c in chunks:
            page_num = c.chunk_metadata.get('page')
            page_tag = f"--- [PAGE {page_num + 1}] ---\n" if page_num is not None else ""
            text_blocks.append(f"{page_tag}{c.chunk_text}")
                
        full_text = "\n\n".join(text_blocks)
        doc_type = self._detect_document_type(full_text)
        
        # Extract headings, skipping any page in the TOC Quarantine
        llm_input = self._extract_headings_only(full_text, doc_type, toc_pages)

        # Step C: Batch and send to LLM
        input_batches = self._split_into_batches(llm_input, self.MAX_LLM_INPUT_CHARS_PER_BATCH)[:self.MAX_STRUCTURE_BATCHES]
        batch_structures = []
        
        async def process_single_batch(i: int, batch_text: str):
            try:
                prompt = self._build_structure_prompt(batch_text, max_topics)
                response = await self._generate_with_retry(prompt, temperature=0.1, max_output_tokens=8000)
                return i, response
            except Exception as e:
                self.logger.error(f"Batch {i + 1} CRASHED: {e}")
                return i, None

        self.logger.info(f"Firing {len(input_batches)} batches to LLM...")
        results = await asyncio.gather(*[
            process_single_batch(i, batch) for i, batch in enumerate(input_batches)
        ], return_exceptions=True)

        for res in results:
            if isinstance(res, tuple) and res[1]:
                parsed = self._parse_json_response(res[1])
                if parsed and parsed.get("topics"):
                    batch_structures.append(parsed)

        # Step D: Merge, Sort, and Apply Deterministic Page Bounds
        raw_merged = self._merge_batches(batch_structures)
        final_structure = self._enforce_strict_hierarchy_and_bounds(raw_merged, last_doc_page)

        extracted_count = len(final_structure.get("topics", []))
        self.logger.info(f"Extraction COMPLETE. Found {extracted_count} Topics in {time.time() - start_time:.2f}s")

        return final_structure if extracted_count > 0 else self._create_fallback_structure()

    async def analyze_material_structure(self, chunk_model, project_id, asset_id=None, max_topics=None, use_all_chunks=False):
        raw_structure = await self.analyze_lecture_structure(chunk_model, project_id, max_topics, asset_id, use_all_chunks)
        normalized = self.normalize_structure(raw_structure)
        return normalized, "completed" if normalized else "failed"

    # =============================================================
    # 2. DETERMINISTIC MATH ENGINE (HARD-CLAMPING BOUNDS)
    # =============================================================

    def _enforce_strict_hierarchy_and_bounds(self, structure: dict, last_page: int) -> dict:
        """
        Enforces perfectly chronological sorting and strictly clamps bounds 
        so it is mathematically impossible to fail the backend validation.
        """
        if not structure or 'topics' not in structure:
            return structure

        topics = structure['topics']
        
        # 1. Clean data: Ensure all have valid integer page_starts
        valid_topics = []
        for t in topics:
            try: t['page_start'] = int(t['page_start'])
            except: continue
            
            valid_subs = []
            for s in t.get('subtitles', []):
                try: 
                    s['page_start'] = int(s['page_start'])
                    valid_subs.append(s)
                except: continue
            
            t['subtitles'] = valid_subs
            valid_topics.append(t)

        if not valid_topics:
            return {"topics": []}

        # 2. Sort Parents Chronologically
        valid_topics.sort(key=lambda x: x['page_start'])

        # 3. Pull Parent Start Backward to Envelope Earliest Subtitle (if hallucinated earlier)
        for t in valid_topics:
            if t['subtitles']:
                t['subtitles'].sort(key=lambda x: x['page_start'])
                if t['subtitles'][0]['page_start'] < t['page_start']:
                    t['page_start'] = t['subtitles'][0]['page_start']

        # Re-sort just in case pulling them backward changed their order
        valid_topics.sort(key=lambda x: x['page_start'])

        # 4. Calculate Parent End
        for i in range(len(valid_topics)):
            curr_start = valid_topics[i]['page_start']
            next_start = valid_topics[i + 1]['page_start'] if i + 1 < len(valid_topics) else (last_page + 1)
            valid_topics[i]['page_end'] = max(curr_start, next_start - 1)

        # 5. STRICT SUBTITLE CLAMPING (Backend 400 Prevention)
        for t in valid_topics:
            parent_start = t['page_start']
            parent_end = t['page_end']
            subs = t['subtitles']
            
            for j in range(len(subs)):
                # CLAMP START: Force subtitle start to physically sit inside parent
                clamped_start = max(parent_start, min(subs[j]['page_start'], parent_end))
                subs[j]['page_start'] = clamped_start
                
                # Determine where the next subtitle begins
                next_start = parent_end + 1
                if j + 1 < len(subs):
                    next_start = max(parent_start, min(subs[j + 1]['page_start'], parent_end))
                
                # CLAMP END: Force subtitle end to sit before the next subtitle and inside parent
                clamped_end = max(clamped_start, min(next_start - 1, parent_end))
                subs[j]['page_end'] = clamped_end

        return {"topics": valid_topics}

    def _merge_batches(self, batch_structures: List[dict]) -> dict:
        all_topics = []
        for struct in batch_structures:
            all_topics.extend(struct.get("topics", []))
        return {"topics": all_topics}

    # =============================================================
    # 3. TOC QUARANTINE & HEADING EXTRACTION
    # =============================================================

    def _identify_toc_pages(self, chunks: list) -> set:
        toc_pages = set()
        toc_keywords = ["table of contents", "\ncontents\n", "قائمة المحتويات", "فهرس", "محتويات"]
        
        for c in chunks:
            page = int(c.chunk_metadata.get('page', 0)) + 1
            if page > 25: continue
                
            text_lower = c.chunk_text.lower()
            if any(kw in text_lower[:500] for kw in toc_keywords):
                toc_pages.add(page)
                continue
                
            dot_leaders = len(re.findall(r'(?:\.{3,}|\s{4,}|_{3,})\d+\s*$', c.chunk_text, re.MULTILINE))
            if dot_leaders >= 3:
                toc_pages.add(page)
                
        return toc_pages

    def _extract_headings_only(self, text: str, doc_type: str, toc_pages: set) -> str:
        lines = text.split("\n")
        heading_lines = []
        current_page = 1

        for raw in lines:
            line = re.sub(r"\s+", " ", raw.replace("\u00a0", " ").strip())
            line = re.sub(r"\s+\d{1,4}$", "", line)
            line = re.sub(r"^\d{1,4}\s+", "", line).strip()
            
            page_match = re.match(r"^---\s*\[PAGE\s+(\d+)\]\s*---$", line, re.IGNORECASE)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            if current_page in toc_pages or not line or self._is_noise(line):
                continue

            is_heading = False
            if re.match(r"^(chapter|section|part|unit|topic|module|lecture|الفصل|الباب|الوحدة|الدرس|الجزء)\s+[\d\wأ-ي]+", line, re.IGNORECASE):
                is_heading = True
            elif re.match(r"^\d+(\.\d+)*\.?\s+[A-Zأ-ي]", line):
                is_heading = True
            elif re.match(r"^[IVXLCDM]+\.\s+[A-Z]", line):
                is_heading = True
            elif doc_type == "lecture" and len(line.split()) >= 2:
                words = line.split()
                if sum(1 for w in words if w and w[0].isupper()) / len(words) >= 0.6: 
                    is_heading = True

            if is_heading and len(line.split()) <= 15:
                is_l1 = bool(re.match(r"^(chapter|part|unit|module|الفصل|الباب|الوحدة|الجزء)\s+", line, re.IGNORECASE) or 
                             re.match(r"^[IVXLCDM]+\.\s+[A-Z]", line) or 
                             re.match(r"^\d+\.?\s+[A-Zأ-ي]", line))
                
                tag = "[L1]" if is_l1 else "[L2]"
                heading_lines.append(f"{tag} {line} (Page {current_page})")

        return "\n".join(heading_lines)

    def _is_noise(self, line: str) -> bool:
        if len(line) < 3 or len(line) > 120: return True
        if re.fullmatch(r"\d+(?:\.\d+)*", line) or re.search(r"[=+\-*/<>$≈≠≤≥∑∫∞]", line): return True
        alpha_count = sum(1 for c in line if c.isalpha())
        if len(line) > 0 and (alpha_count / len(line)) < 0.5: return True
        if line.endswith("?") or line.endswith("؟"): return True
        return False

    def _detect_document_type(self, text: str) -> str:
        text_lower = text.lower()
        book_patterns = [r"\bchapter\s+\d+\b", r"\bpart\s+[ivxlcdm\d]+\b", r"\bالفصل\s+[\dأ-ي]+\b"]
        if sum(1 for p in book_patterns if re.search(p, text_lower, re.IGNORECASE)) >= 2:
            return "book"
        return "lecture"

    def _split_into_batches(self, text: str, max_chars: int) -> List[str]:
        if len(text) <= max_chars: return [text] if text.strip() else []
        lines = text.split("\n")
        batches, curr, curr_len = [], [], 0
        for line in lines:
            if curr and curr_len + len(line) > max_chars:
                batches.append("\n".join(curr))
                curr, curr_len = [line], len(line)
            else:
                curr.append(line)
                curr_len += len(line)
        if curr: batches.append("\n".join(curr))
        return batches

    # =============================================================
    # 4. LLM PROMPTING & PARSING
    # =============================================================

    def _build_structure_prompt(self, text: str, max_topics: int = None) -> str:
        return f"""You are an expert academic parser constructing a strict 2-level hierarchy from document headings.

CRITICAL RULES:
1. STRICTLY 2 LEVELS: Top-Level Topics (L1) and Subtitles (L2). Group any L2 headings under their preceding L1 heading.
2. PAGE START ONLY: Extract the (Page X) number to 'page_start'. DO NOT include a 'page_end' field.
3. FLATTEN: Ignore or merge sub-subsections (e.g., 1.1.1).
4. VALID JSON ONLY: Match the exact schema below.

EXTRACTED HEADINGS:
{text}

OUTPUT JSON SCHEMA:
{{
  "topics": [
    {{
      "title": "Topic Title",
      "description": "Short 3-6 word summary",
      "page_start": 5,
      "subtitles": [
        {{"title": "Subtitle Title", "description": "Short 3-6 word summary", "page_start": 6}}
      ]
    }}
  ]
}}"""

    async def _generate_with_retry(self, prompt: str, temperature: float, max_output_tokens: int):
        for attempt in range(3):
            try:
                res = await asyncio.to_thread(
                    self.generation_client.generate_text, prompt, [], max_output_tokens, temperature
                )
                if res: return res
            except Exception:
                await asyncio.sleep(2)
        return None

    def _parse_json_response(self, response: str) -> dict:
        if not response: return None
        response = response.strip()
        if "```json" in response: response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response: response = response.split("```")[1].strip() if len(response.split("```")) >= 3 else response.replace("```", "").strip()
        
        start = response.find("{")
        if start == -1: return None
        
        try:
            struct = json.loads(response[start:])
            if "topics" in struct: return struct
        except: pass
        return None

    # =============================================================
    # 5. DATABASE NORMALIZATION
    # =============================================================

    def normalize_structure(self, raw_structure: dict) -> List[Dict[str, Any]]:
        normalized = []
        if not raw_structure or "topics" not in raw_structure:
            return self._create_fallback_structure()["topics"]

        topic_counter = 0

        for topic in raw_structure["topics"]:
            topic_counter += 1
            topic_temp_id = f"topic_{topic_counter}"
            
            normalized.append({
                "temp_id": topic_temp_id,
                "title": str(topic.get("title", ""))[:250],
                "description": str(topic.get("description", ""))[:500],
                "order_index": topic_counter - 1,
                "parent_temp_id": None,
                "page_start": topic.get("page_start"),
                "page_end": topic.get("page_end")
            })

            for subtitle in topic.get("subtitles", []):
                topic_counter += 1
                normalized.append({
                    "temp_id": f"topic_{topic_counter}",
                    "title": str(subtitle.get("title", ""))[:250],
                    "description": str(subtitle.get("description", ""))[:500],
                    "order_index": topic_counter - 1,
                    "parent_temp_id": topic_temp_id,
                    "page_start": subtitle.get("page_start"),
                    "page_end": subtitle.get("page_end")
                })

        return normalized

    def _create_fallback_structure(self) -> dict:
        return {"topics": [{"temp_id": "topic_1", "title": "Document Content", "description": "Content extracted from document.", "order_index": 0, "parent_temp_id": None, "page_start": 1, "page_end": 1}]}