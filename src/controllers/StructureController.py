from .BaseController import BaseController
from models.ChunkModel import ChunkModel
from typing import List, Optional, Dict, Any
import json
import re
import logging


class StructureController(BaseController):

    def __init__(self, generation_client):
        super().__init__()
        self.generation_client = generation_client
        self.logger = logging.getLogger(__name__)

    # =============================================================
    # PUBLIC ENTRY POINTS
    # =============================================================

    async def analyze_lecture_structure(
        self,
        chunk_model: ChunkModel,
        project_id: str,
        max_topics: int = None,
        use_all_chunks: bool = False
    ) -> dict:
        """
        Analyze chunks and extract topics/subtopics structure.

        Strategy:
          1. Load ALL structure chunks (no arbitrary limit).
          2. Run cheap string operations on full text:
             - Detect document type (book vs lecture)
             - Detect Table of Contents
          3. Prepare a COMPACT input for the LLM:
             - If ToC found  → send only the ToC text (~500-1500 chars)
             - If no ToC     → extract heading-like lines only (~1000-3000 chars)
             This fits any document size within Groq's free 12k TPM limit
             while giving the LLM cleaner signal than raw body text.
          4. Single LLM call → parse → normalize.
        """

        # ── 1. Load all structure chunks ──────────────────────────────────
        chunks = await chunk_model.get_chunks_by_project_id(
            project_id=project_id,
            page_no=1,
            page_size=2000,
            chunk_type="structure"
        )

        if not chunks:
            self.logger.error(f"No structure chunks found for project {project_id}")
            return self._create_fallback_structure()

        chunks = sorted(chunks, key=lambda c: c.chunk_order)
        total_chunks = len(chunks)

        # ── 2. Reconstruct full document text ─────────────────────────────
        full_text = "\n\n".join([c.chunk_text for c in chunks])
        self.logger.info(f"Full document: {len(full_text)} chars across {total_chunks} chunks")

        # ── 3. Cheap detections on full text (no LLM, no token cost) ──────
        toc_text = self._extract_potential_toc(full_text)
        has_toc  = toc_text is not None
        doc_type = self._detect_document_type(full_text, total_chunks)
        self.logger.info(f"Document type: {doc_type} | Has ToC: {has_toc}")

        # ── 4. Build compact LLM input ────────────────────────────────────
        #
        # KEY INSIGHT: For structure extraction we only need heading lines,
        # not body paragraphs. Stripping body text reduces a 70-page doc
        # from ~80,000 chars (~20,000 tokens) to ~3,000 chars (~750 tokens)
        # — well within any model's TPM limit and giving the LLM cleaner signal.
        #
        # Priority:
        #   a) ToC found → use it directly (most compact + most reliable)
        #   b) No ToC   → extract heading-only lines from full text
        if has_toc:
            llm_input = toc_text
            self.logger.info(f"Using ToC as LLM input ({len(llm_input)} chars)")
        else:
            llm_input = self._extract_headings_only(full_text)
            self.logger.info(f"Heading-only input: {len(llm_input)} chars (from {len(full_text)} chars)")

            if len(llm_input.strip()) < 200:
                self.logger.warning("Heading extraction yielded too little — using raw text fallback")
                llm_input = full_text[:8000]

        # Hard cap: ~9,000 chars ≈ 2,250 tokens for document input.
        # Prompt template adds ~500 tokens, output needs ~1,500 tokens.
        # Total ~4,250 tokens — safe for llama3-8b-8192's 30k TPM free tier
        # AND safe for llama-3.3-70b-versatile's 12k TPM if someone uses it.
        MAX_LLM_INPUT_CHARS = 9000
        if len(llm_input) > MAX_LLM_INPUT_CHARS:
            llm_input = llm_input[:MAX_LLM_INPUT_CHARS]
            self.logger.info(f"LLM input hard-capped at {MAX_LLM_INPUT_CHARS} chars")

        self.logger.info(f"Final LLM input:\n{llm_input}")

        

        # ── 5. Build prompt and call LLM ──────────────────────────────────
        try:
            if has_toc:
                prompt = self._build_toc_extraction_prompt(
                    toc_text=llm_input,
                    doc_type=doc_type,
                    max_topics=max_topics
                )
            else:
                prompt = self._build_headings_extraction_prompt(
                    headings_text=llm_input,
                    doc_type=doc_type,
                    max_topics=max_topics
                )

            response = self.generation_client.generate_text(
                prompt=prompt,
                temperature=0.1,
                max_output_tokens=self.app_settings.GENERATION_DAFAULT_MAX_TOKENS or 4096
            )

            if not response:
                self.logger.error("Empty response from LLM")
                return self._create_fallback_structure()

            structure = self._parse_structure_response(response)

            if structure and "topics" in structure and len(structure["topics"]) > 0:
                self.logger.info(f"Extracted {len(structure['topics'])} top-level topics")
                return structure

            self.logger.warning("LLM returned empty/invalid topics")
            return self._create_fallback_structure()

        except Exception as e:
            self.logger.error(f"Error during structure analysis: {e}")
            return self._create_fallback_structure()

    async def analyze_material_structure(
        self,
        chunk_model: ChunkModel,
        project_id: str,
        max_topics: int = None,
        use_all_chunks: bool = False
    ) -> tuple:
        """
        Analyze structure and return normalized flat format.
        project_id in this system = material_id in the backend system.

        Returns:
            tuple: (normalized_topics_list, status)
        """
        raw_structure = await self.analyze_lecture_structure(
            chunk_model=chunk_model,
            project_id=project_id,
            max_topics=max_topics,
            use_all_chunks=use_all_chunks
        )
        normalized = self.normalize_structure(raw_structure)
        status = "completed" if normalized else "failed"
        return normalized, status

    # =============================================================
    # DOCUMENT TYPE DETECTION
    # =============================================================

    def _detect_document_type(self, text: str, total_chunks: int) -> str:
        """
        Returns "book" or "lecture" based on content signals and length.
        """
        text_lower = text.lower()

        book_patterns = [
            r"\bchapter\s+\d+\b",
            r"\bchapter\s+[ivxlcdm]+\b",
            r"\bpart\s+[ivxlcdm\d]+\b",
            r"\bunit\s+\d+\b",
            r"\btable\s+of\s+contents?\b",
        ]

        book_hits = sum(1 for p in book_patterns if re.search(p, text_lower))

        if book_hits >= 1 or (total_chunks > 30 and len(text) > 20000):
            return "book"
        return "lecture"

    # =============================================================
    # TABLE OF CONTENTS DETECTION
    # =============================================================

    def _extract_potential_toc(self, text: str) -> Optional[str]:
        """
        Scan the first 8000 chars for a Table of Contents.
        Returns the ToC block as a string, or None if not found.
        """
        scan = text[:8000]
        scan_lower = scan.lower()

        toc_headers = [
            r"table\s+of\s+contents?",
            r"^contents?\s*$",
        ]

        toc_start = -1
        for pattern in toc_headers:
            m = re.search(pattern, scan_lower, re.MULTILINE)
            if m:
                toc_start = m.start()
                break

        if toc_start == -1:
            # No explicit header — look for a dense block of numbered headings
            numbered = re.findall(
                r"^\s*\d+[\.\)]\s+[A-Z][^\n]{3,60}$",
                scan,
                re.MULTILINE
            )
            if len(numbered) >= 5:
                return "\n".join(numbered)
            return None

        # Extract up to 3000 chars from the ToC start
        toc_block = scan[toc_start:toc_start + 3000]

        # Trim at likely end of ToC
        end_match = re.search(r"\n\n[A-Z][a-z]{3,}", toc_block[200:])
        if end_match:
            toc_block = toc_block[:200 + end_match.start()]

        return toc_block.strip() if len(toc_block.strip()) > 100 else None

    # =============================================================
    # HEADING EXTRACTION  (key to token efficiency)
    # =============================================================

    def _extract_headings_only(self, text: str) -> str:
        """
        Extract only heading-like lines from the full document text.

        This reduces a 70-page document (~80,000 chars, ~20,000 tokens) to
        only the structural skeleton (~2,000-4,000 chars, ~500-1,000 tokens).

        A line is treated as a heading if it matches ANY of these signals:
          - Numbered:   "1.", "1.1", "Chapter 3", "Section 2.4"
          - Short caps: Line <= 80 chars and mostly Title Case or UPPER CASE
          - Isolated:   Short line surrounded by blank lines (common in PDFs)
          - Colon:      Short line ending with a colon
        """
        lines = text.split("\n")
        heading_lines = []
        prev_blank = True

        for i, line in enumerate(lines):
            stripped = line.strip()
            next_blank = (i + 1 >= len(lines)) or (lines[i + 1].strip() == "")

            if not stripped:
                prev_blank = True
                continue

            is_heading = False

            # Signal 1: explicit section keyword + number
            if re.match(
                r"^(chapter|section|part|unit|topic|module|lecture)\s+[\d\w]+",
                stripped, re.IGNORECASE
            ):
                is_heading = True

            # Signal 2: numeric outline  "1.", "1.1", "2.3.1"
            elif re.match(r"^\d+(\.\d+)*\.?\s+\S", stripped):
                is_heading = True

            # Signal 3: Roman numeral heading  "IV. Methods"
            elif re.match(r"^[IVXLCDM]+\.\s+[A-Z]", stripped):
                is_heading = True

            # Signal 4: short line that is mostly Title Case or ALL CAPS
            elif len(stripped) <= 80:
                words = stripped.split()
                if len(words) >= 2:
                    capitalized = sum(1 for w in words if w and (w[0].isupper() or w.isupper()))
                    if capitalized / len(words) >= 0.7:
                        is_heading = True

            # Signal 5: isolated short line (blank before AND after)
            if not is_heading and prev_blank and next_blank and len(stripped) <= 80:
                is_heading = True

            # Signal 6: short line ending with colon
            if not is_heading and stripped.endswith(":") and len(stripped) <= 80:
                is_heading = True

            # Reject lines that are clearly body text
            if is_heading:
                if len(stripped.split()) > 15:
                    is_heading = False  # too long to be a heading
                if stripped.endswith(".") and not re.match(r"^\d", stripped):
                    is_heading = False  # ends like a sentence

            if is_heading:
                heading_lines.append(stripped)

            prev_blank = (stripped == "")

        return "\n".join(heading_lines)

    # =============================================================
    # PROMPT BUILDERS
    # =============================================================

    def _build_toc_extraction_prompt(
        self,
        toc_text: str,
        doc_type: str,
        max_topics: int = None
    ) -> str:
        max_topics_rule = f"Include at most {max_topics} top-level topics.\n" if max_topics else ""

        depth_guide = (
            "This is a TEXTBOOK or multi-chapter document.\n"
            "- Top-level topics = chapters or major parts\n"
            "- Subtitles = sections within each chapter (include ALL of them)\n"
            "- Maximum depth = 2 levels only"
        ) if doc_type == "book" else (
            "This is a SINGLE LECTURE or short document.\n"
            "- Top-level topics = main sections\n"
            "- Subtitles = sub-points within each section"
        )

        return f"""You are an expert educational content analyzer.
A Table of Contents was found. Parse it to extract the COMPLETE structure — do not skip any entry.

{depth_guide}
{max_topics_rule}
RULES:
- Extract EVERY topic and subtitle listed — do not truncate
- Use exact title wording from the ToC
- Write a 1-sentence description for each item based on its title
- Return ONLY valid JSON — no markdown, no explanation

TABLE OF CONTENTS:
{toc_text}

JSON schema:
{{
  "topics": [
    {{
      "title": "Title from ToC",
      "description": "One sentence about what this covers.",
      "order": 0,
      "subtitles": [
        {{"title": "Subtitle", "description": "One sentence.", "order": 0}}
      ]
    }}
  ]
}}

Output JSON only:"""

    def _build_headings_extraction_prompt(
        self,
        headings_text: str,
        doc_type: str,
        max_topics: int = None
    ) -> str:
        max_topics_rule = f"Include at most {max_topics} top-level topics.\n" if max_topics else ""

        depth_guide = (
            "This is a TEXTBOOK or multi-chapter document.\n"
            "- Top-level topics = chapters or major parts\n"
            "- Group sub-numbered headings (e.g. 2.1, 2.2) as subtitles under their chapter\n"
            "- Include ALL subtitles — do not skip any\n"
            "- A textbook typically has 5–20 chapters each with multiple subtopics"
        ) if doc_type == "book" else (
            "This is a SINGLE LECTURE or short document.\n"
            "- Top-level topics = main sections\n"
            "- Subtitles = sub-points within each section\n"
            "- A typical lecture has 3-8 main topics"
        )

        return f"""You are an expert educational content structure analyzer.
The following lines are the headings extracted from a document.
Reconstruct the full hierarchical structure from them.

{depth_guide}
{max_topics_rule}
RULES:
RULES:
- Use the EXACT wording from the headings — do not rename or generalize
- Write a 1-sentence description per topic/subtitle based on its title
- Do NOT invent topics not present in the headings
- Do NOT return generic titles like "Document Content" or "Main Section"
- If structure is flat, return all items as top-level topics with empty subtitles []
- Ignore review questions, checkpoints, exercises, quiz items, discussion questions, and self-test questions
- Ignore headings that are phrased as questions such as "What is...?", "List...", "Why...?", "How...?"
- Return ONLY valid JSON — no markdown, no explanation

DOCUMENT HEADINGS:
{headings_text}

JSON schema:
{{
  "topics": [
    {{
      "title": "Heading title",
      "description": "One sentence about what this section covers.",
      "order": 0,
      "subtitles": [
        {{"title": "Sub-heading title", "description": "One sentence.", "order": 0}}
      ]
    }}
  ]
}}

Output JSON only:"""

    # =============================================================
    # RESPONSE PARSER
    # =============================================================

    def _parse_structure_response(self, response: str) -> dict:
        if not response:
            return self._create_fallback_structure()

        response = response.strip()
        print("\n================ LLM FULL RESPONSE ================\n")
        print(response)
        print("\n==================================================\n")

        # Strip markdown fences
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            parts = response.split("```")
            response = parts[1] if len(parts) >= 3 else response.replace("```", "")

        response = response.strip()

        first_brace   = response.find("{")
        first_bracket = response.find("[")

        if first_brace == -1 and first_bracket == -1:
            self.logger.error("No JSON found in LLM response")
            return self._create_fallback_structure()

        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start, end = first_brace, response.rfind("}") + 1
        else:
            start, end = first_bracket, response.rfind("]") + 1

        if end <= start:
            return self._create_fallback_structure()

        json_str = response[start:end].strip()

        try:
            structure = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse failed: {e} — attempting repair")
            structure = self._attempt_json_repair(json_str)
            if not structure:
                return self._create_fallback_structure()

        if isinstance(structure, list):
            structure = {"topics": structure}

        if "topics" not in structure:
            return self._create_fallback_structure()

        valid_topics = []
        bad_titles = {"untitled", "none", "null", "document content", "main section"}

        for i, topic in enumerate(structure["topics"]):
            title = str(topic.get("title", "")).strip()
            if not title or title.lower() in bad_titles:
                continue

            subtitles = []
            for j, sub in enumerate(topic.get("subtitles", [])):
                sub_title = str(sub.get("title", "")).strip()
                if not sub_title or sub_title.lower() in bad_titles:
                    continue
                subtitles.append({
                    "title": sub_title,
                    "description": str(sub.get("description", "")).strip() or f"Subtopic of {title}.",
                    "order": sub.get("order", j)
                })

            valid_topics.append({
                "title": title,
                "description": str(topic.get("description", "")).strip() or f"This section covers {title}.",
                "order": topic.get("order", i),
                "subtitles": subtitles
            })

        if not valid_topics:
            self.logger.warning("No valid topics after normalization")
            return self._create_fallback_structure()

        self.logger.info(f"Parsed {len(valid_topics)} valid topics")
        structure["topics"] = valid_topics
        return structure

    def _attempt_json_repair(self, json_str: str) -> Optional[dict]:
        for suffix in ["]}]}", "]}", "}"]:
            try:
                return json.loads(json_str + suffix)
            except json.JSONDecodeError:
                continue
        last_complete = json_str.rfind("},")
        if last_complete > 0:
            try:
                return json.loads(json_str[:last_complete + 1] + "]}")
            except json.JSONDecodeError:
                pass
        return None

    # =============================================================
    # FALLBACK
    # =============================================================

    def _create_fallback_structure(self) -> dict:
        self.logger.warning("Using fallback structure — all extraction attempts failed")
        return {
            "topics": [
                {
                    "title": "Document Content",
                    "description": "Content extracted from the uploaded document.",
                    "order": 0,
                    "subtitles": []
                }
            ]
        }

    # =============================================================
    # NORMALIZER  (nested → flat backend format)
    # =============================================================

    def normalize_structure(self, raw_structure: dict) -> List[Dict[str, Any]]:
        """
        Convert nested structure to flat list matching the backend expected format.

        Each item:
        {
            "temp_id":        "topic_N",
            "title":          str,
            "description":    str,
            "order_index":    int,   # global zero-based flat index
            "parent_temp_id": str | null
        }
        """
        normalized = []
        topic_counter = 0

        if not raw_structure or "topics" not in raw_structure:
            return normalized

        for topic in raw_structure["topics"]:
            title = topic.get("title", "").strip()
            if not title:
                continue

            topic_counter += 1
            topic_temp_id = f"topic_{topic_counter}"

            normalized.append({
                "temp_id": topic_temp_id,
                "title": title,
                "description": topic.get("description", "").strip() or f"This section covers {title}.",
                "order_index": topic_counter - 1,
                "parent_temp_id": None
            })

            for subtitle in topic.get("subtitles", []):
                sub_title = subtitle.get("title", "").strip()
                if not sub_title:
                    continue

                topic_counter += 1
                normalized.append({
                    "temp_id": f"topic_{topic_counter}",
                    "title": sub_title,
                    "description": subtitle.get("description", "").strip() or f"Subtopic under {title}.",
                    "order_index": topic_counter - 1,
                    "parent_temp_id": topic_temp_id
                })

        return normalized