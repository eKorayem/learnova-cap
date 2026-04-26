from models.ChunkModel import ChunkModel
from controllers.BaseController import BaseController
from typing import List, Optional, Dict, Any, Tuple
import json
import re
import logging


class StructureController(BaseController):
    """
    Universal Parser for academic document structure extraction.
    Handles: Textbooks (with/without ToC), Lecture Slides, Bilingual (EN/AR) content.
    """

    def __init__(self, generation_client=None):
        super().__init__()
        self.generation_client = generation_client
        self.logger = logging.getLogger(__name__)

        # Document type detection thresholds
        self.BOOK_CHUNK_THRESHOLD = 30
        self.BOOK_CHAR_THRESHOLD = 30000
        self.TOC_SCAN_LIMIT = 25000
        self.TOC_BLOCK_SIZE = 6000
        self.MAX_LLM_INPUT_CHARS = 20000
        self.HEADING_MIN_LENGTH = 3
        self.HEADING_MAX_LENGTH = 120
        self.MAX_HEADING_WORDS = 12

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
        import time # <--- Make sure this is imported at the top of the file!
        start_time = time.time()

        self.logger.info(f"========== [STARTED] STRUCTURE EXTRACTION FOR PROJECT {project_id} ==========")
        
        # 1. Fetch chunks
        chunks = await chunk_model.get_chunks_by_project_id(
            project_id=project_id, page_no=1, page_size=3000, chunk_type="structure"
        )

        if not chunks:
            self.logger.error("DEBUG: 0 Chunks found. Aborting.")
            return self._create_fallback_structure()

        chunks = sorted(chunks, key=lambda c: c.chunk_order)
        total_chunks = len(chunks)
        full_text = "\n\n".join([c.chunk_text for c in chunks])
        
        self.logger.info(f"DEBUG: Fetched {total_chunks} chunks. Total raw characters: {len(full_text)}")

        # 2. Universal Detection
        doc_type = self._detect_document_type(full_text, total_chunks)
        toc_text = self._extract_potential_toc(full_text)
        has_toc = toc_text is not None

        # 3. Dynamic Extraction Routing
        if has_toc:
            llm_input = toc_text
            prompt = self._build_toc_prompt(llm_input, max_topics)
            strategy = "Table of Contents"
        elif doc_type == "book":
            llm_input = self._extract_headings_only(full_text, doc_type)
            if len(llm_input.strip()) < 200:
                llm_input = full_text[:10000]
            prompt = self._build_book_headings_prompt(llm_input, max_topics)
            strategy = "Book Headings"
        else:
            llm_input = self._extract_headings_only(full_text, doc_type)
            if len(llm_input.strip()) < 200:
                llm_input = full_text[:10000]
            prompt = self._build_lecture_prompt(llm_input, max_topics)
            strategy = "Lecture Slides"

        # Hard cap input
        if len(llm_input) > self.MAX_LLM_INPUT_CHARS:
            llm_input = llm_input[:self.MAX_LLM_INPUT_CHARS]

        # Calculate approximate input tokens (1 token ≈ 4 chars)
        approx_in_tokens = len(prompt) // 4
        
        # Log the exact prompt being sent (useful for checking Arabic handling)
        print(f"\n[DEBUG] === EXACT PROMPT SENT TO LLM ===\n{prompt}\n========================================\n")

        # 4. Generate AI Prompt
        try:
            llm_start_time = time.time()
            response = self.generation_client.generate_text(
                prompt=prompt,
                temperature=0.1,
                max_output_tokens=self.app_settings.GENERATION_DAFAULT_MAX_TOKENS or 4096
            )
            llm_execution_time = time.time() - llm_start_time

            if not response:
                self.logger.error("DEBUG: LLM returned an empty string.")
                return self._create_fallback_structure()

            approx_out_tokens = len(response) // 4
            
            # 5. Parse the Response
            structure = self._parse_structure_response(response)
            
            extracted_count = len(structure.get("topics", [])) if structure else 0
            status = "SUCCESS" if extracted_count > 0 else "FAILED (Fallback Used)"

            # ================= DEBUG SUMMARY REPORT =================
            total_time = time.time() - start_time
            print("\n" + "="*50)
            print("PIPELINE DEBUG SUMMARY: STRUCTURE EXTRACTION")
            print("="*50)
            print(f"Document Type     : {doc_type.upper()}")
            print(f"Parsing Strategy  : {strategy}")
            print(f"Chunks Processed  : {total_chunks}")
            print(f"Text Reduction    : {len(full_text)} chars -> {len(llm_input)} chars sent")
            print(f"Approx In Tokens  : ~{approx_in_tokens} tokens")
            print(f"Approx Out Tokens : ~{approx_out_tokens} tokens")
            print(f"LLM Latency       : {llm_execution_time:.2f} seconds")
            print(f"Total Time        : {total_time:.2f} seconds")
            print(f"Final Status      : {status} ({extracted_count} topics found)")
            print("="*50 + "\n")
            # ========================================================

            if extracted_count > 0:
                return structure

            return self._create_fallback_structure()

        except Exception as e:
            self.logger.error(f"DEBUG: Fatal error during extraction: {e}")
            return self._create_fallback_structure()

    async def analyze_material_structure(
        self,
        chunk_model: ChunkModel,
        project_id: str,
        max_topics: int = None,
        use_all_chunks: bool = False
    ) -> tuple:
        raw_structure = await self.analyze_lecture_structure(
            chunk_model=chunk_model, project_id=project_id,
            max_topics=max_topics, use_all_chunks=use_all_chunks
        )
        normalized = self.normalize_structure(raw_structure)
        return normalized, "completed" if normalized else "failed"

    # =============================================================
    # UNIVERSAL DETECTION
    # =============================================================

    def _detect_document_type(self, text: str, total_chunks: int) -> str:
        """
        Detects if document is a 'book' or 'lecture' based on keywords and size.
        Book indicators: chapter markers, part/unit markers, ToC headers, Arabic equivalents.
        """
        text_lower = text.lower()
        book_patterns = [
            r"\bchapter\s+\d+\b", r"\bchapter\s+[ivxlcdm]+\b", r"\bpart\s+[ivxlcdm\d]+\b",
            r"\bunit\s+\d+\b", r"\btable\s+of\s+contents?\b",
            r"\bالفصل\s+[\dأ-ي]+\b", r"\bالباب\s+[\dأ-ي]+\b", r"\bالجزء\s+[\dأ-ي]+\b"
        ]
        book_hits = sum(1 for p in book_patterns if re.search(p, text_lower, re.IGNORECASE))

        if book_hits >= 2 or (total_chunks > self.BOOK_CHUNK_THRESHOLD and len(text) > self.BOOK_CHAR_THRESHOLD):
            return "book"
        return "lecture"

    def _extract_potential_toc(self, text: str) -> Optional[str]:
        """
        Extracts Table of Contents from document.
        Scans first 25k chars for ToC headers in English and Arabic.
        Falls back to detecting dense clusters of numbered headings.
        """
        scan = text[:self.TOC_SCAN_LIMIT]
        scan_lower = scan.lower()

        toc_headers = [
            r"table\s+of\s+contents?", r"^\s*contents?\s*$",
            r"فهرس", r"جدول\s+المحتويات", r"المحتويات", r"فهرس\s+المحتويات"
        ]

        toc_start = -1
        toc_header_end = 0
        for pattern in toc_headers:
            m = re.search(pattern, scan_lower, re.MULTILINE | re.IGNORECASE)
            if m:
                toc_start = m.start()
                toc_header_end = m.end()
                break

        if toc_start == -1:
            # Fallback: implicit ToC from dense numbered headings
            numbered = re.findall(r"^\s*\d+[\.\)]\s+[A-Zأ-ي][^\n]{3,80}$", scan, re.MULTILINE)
            if len(numbered) >= 5:
                return "\n".join(numbered[:50])  # Cap at 50 entries
            return None

        # Extract ToC block - look for where actual content begins
        remaining = scan[toc_header_end:]

        # Find end of ToC: either first substantial paragraph or 6000 chars
        toc_block = remaining[:self.TOC_BLOCK_SIZE]

        # Look for patterns that indicate end of ToC / start of content
        content_start_patterns = [
            r"\n\n+[A-Zأ-ي][a-zأ-ي\s]{20,100}\n",  # Paragraph start
            r"\n(?:preface|introduction|chapter\s+1|الفصل\s+1)\s*",  # First chapter
            r"\n\s*\d+\n",  # Page number alone
        ]

        for pattern in content_start_patterns:
            end_match = re.search(pattern, toc_block, re.IGNORECASE)
            if end_match:
                toc_block = toc_block[:end_match.start()]
                break

        # Clean: remove page numbers, excessive whitespace
        toc_lines = []
        for line in toc_block.split("\n"):
            line = line.strip()
            # Remove trailing page numbers
            line = re.sub(r"\s+\d{1,4}$", "", line)
            if line and len(line) > 2:
                toc_lines.append(line)

        result = "\n".join(toc_lines[:100])  # Cap entries
        return result if len(result) > 50 else None

    # =============================================================
    # HEADINGS SCRUBBER (Slide & Noise Immunity)
    # =============================================================

    def _normalize_line(self, line: str) -> str:
        """Normalize unicode, whitespace, and strip trailing page numbers."""
        line = line.replace("\u00a0", " ").replace("\u2002", " ").replace("\u2003", " ").replace("\u2009", " ")
        line = line.replace("\ufeff", " ")
        line = re.sub(r"\s+", " ", line.strip())
        # Strip trailing page numbers (e.g., "Introduction to AI  5" or "Chapter 1  12")
        line = re.sub(r"\s+\d{1,4}$", "", line)
        # Strip leading page numbers
        line = re.sub(r"^\d{1,4}\s+", "", line)
        return line.strip()

    def _looks_like_question_or_exercise(self, line: str) -> bool:
        """
        Detects questions, exercises, and instructional text that should NOT become topics.
        Supports English and Arabic.
        """
        if not line:
            return False
        lower = line.lower().strip()

        # Question marks in any language
        if line.endswith("?") or line.endswith("؟"):
            return True

        # English question/exercise starters
        en_bad_prefixes = [
            "what is", "what are", "what does", "what will", "what can",
            "which of", "which one", "why is", "why are", "why does",
            "how do", "how does", "how can", "how would", "how to",
            "show the output", "show the result", "show that", "show how",
            "identify and fix", "identify the", "find the", "find out",
            "write a program", "write an expression", "write a statement", "write code",
            "translate the following", "evaluate the following", "calculate the",
            "suppose ", "if you ", "if the ", "consider the", "given the",
            "true or false", "give examples", "answer the quiz", "answer the following",
            "discuss the", "explain the", "describe the", "compare the",
            "prove that", "derive the", "list the", "define the",
        ]

        # Arabic question/exercise starters
        ar_bad_prefixes = [
            "ما هو", "ما هي", "ما ال", "متى", "أين", "من هو", "من هي",
            "لماذا", "كيف", "أي من", "هل يمكن", "هل ",
            "قارن", "اشرح", "وضح", "بيّن", "فسّر", "عرّف",
            "صح أم خطأ", "صح او خطأ", "تمرين", "سؤال", "سؤال ",
            "أوجد", "احسب", "استخرج", "حلّل", "ناقش", "اذكر",
            "ما الفرق", "ما الفرق", "شرح", "مثال",
        ]

        for prefix in en_bad_prefixes + ar_bad_prefixes:
            if lower.startswith(prefix):
                return True

        # Arabic: contains question words anywhere
        ar_question_words = ["ماذا", "متى", "أين", "كيف", "لماذا", "هل", "من", "ما"]
        for word in ar_question_words:
            if word in line:
                return True

        return False

    def _is_noise_or_non_structure(self, line: str) -> bool:
        """
        Aggressively filters out non-structural content.
        Handles: academic years, math variables, page numbers, citations, notes, etc.
        """
        if not line:
            return True

        lower = line.lower().strip()

        # Length filters
        if len(line) < self.HEADING_MIN_LENGTH or len(line) > self.HEADING_MAX_LENGTH:
            return True

        # 1. Pure numbers or version-like patterns
        if re.fullmatch(r"\d+", line):
            return True
        if re.fullmatch(r"\d+(?:\.\d+)+", line):
            return True
        if re.fullmatch(r"\d{4}\s*-\s*\d{4}", line):  # "2023-2024"
            return True
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", line):  # Dates
            return True

        # 2. Lowercase start (sentence fragment, not a heading)
        # Exception: Arabic text doesn't have case
        if line[0].islower() and re.search(r"[a-zA-Z]", line):
            return True

        # 3. Math/Code expressions
        if re.search(r"[=+*/<>$]", line):
            return True
        # High symbol-to-alpha ratio
        alpha_count = sum(1 for c in line if c.isalpha())
        if alpha_count > 0 and (alpha_count / max(len(line), 1)) < 0.4:
            return True
        # Decimal numbering without text
        if re.match(r"^\d+\.\d+\s*$", line):
            return True

        # 4. URLs, emails, citations
        if "http" in lower or "www." in lower or "@" in line:
            return True
        if lower.startswith("isbn") or lower.startswith("issn"):
            return True

        # 5. Noise keywords (English + Arabic)
        noise_keywords = [
            "point", "key point", "checkpoint", "check point", "note", "tip", "caution",
            "pedagogical note", "videonote", "figure", "table ", "listing", "chapter summary",
            "key terms", "quiz", "programming exercises", "supplement", "source:",
            "copyright", "all rights reserved", "problems", "problem set", "learning objectives",
            "references", "bibliography", "acknowledgements", "preface", "foreword",
            "dr.", "prof.", "professor", "faculty of", "university", "college of", "department of",
            "spring", "fall", "summer", "winter", "semester", "lecture", "session",
            "thank you", "questions?", "q&a", "any questions", "agenda",
            # --- Arabic Noise ---
            "ملاحظة", "تنبيه", "شكل", "جدول", "حقوق الطبع", "ملخص", "مراجع",
            "شكر", "تمهيد", "إهداء", "غلاف", "دكتور", "أستاذ", "كلية", "جامعة", "قسم", "شكرا", "الأسئلة"
        ]

        for keyword in noise_keywords:
            if keyword in lower or lower.startswith(keyword):
                return True

        # 6. Questions/exercises
        if self._looks_like_question_or_exercise(line):
            return True

        # 7. Colon trap: Short instructional text ending in colon
        # "To describe this object you use:" - but allow "Chapter 1:"
        if line.endswith(":"):
            words = line.split()
            if len(words) <= 5:
                # Allow if starts with chapter/unit markers
                if not re.match(r"^(chapter|section|part|unit|topic|الفصل|الباب|الوحدة|الدرس)\s", line, re.IGNORECASE):
                    return True

        # 8. All caps short fragments (likely labels, not headings)
        if line.isupper() and len(line) < 10 and " " not in line:
            return True

        return False

    def _extract_headings_only(self, text: str, doc_type: str) -> str:
        """
        Extracts only valid headings from text, filtering out body content and noise.
        Uses spatial awareness (blank lines) for lectures, and strict numbering for books.
        """
        lines = text.split("\n")
        heading_lines = []
        seen = set()
        consecutive_non_headings = 0
        
        # Bring back spatial awareness from the old version!
        prev_blank = True

        for i, raw in enumerate(lines):
            line = self._normalize_line(raw)
            
            # Look ahead to see if the next line is empty
            next_blank = (i + 1 >= len(lines)) or (self._normalize_line(lines[i + 1]) == "")

            if not line:
                consecutive_non_headings = 0
                prev_blank = True
                continue

            if self._is_noise_or_non_structure(line):
                consecutive_non_headings += 1
                if consecutive_non_headings > 3:
                    prev_blank = False
                    continue

            is_heading = False

            # === STRICT: Book-style numbered headings ===
            if re.match(r"^(chapter|section|part|unit|topic|module|lecture|الفصل|الباب|الوحدة|الدرس|الجزء)\s+[\d\wأ-ي]+", line, re.IGNORECASE):
                is_heading = True
            elif re.match(r"^\d+(\.\d+)*\.?\s+[A-Zأ-ي]", line):
                is_heading = True
            elif re.match(r"^[IVXLCDM]+\.\s+[A-Z]", line):
                is_heading = True
            elif re.match(r"^[\u0660-\u0669]+[\.\)]\s+[أ-ي]", line):
                is_heading = True

            # === SPATIAL AWARENESS: Lecture slide titles ===
            elif doc_type == "lecture":
                # Old Code Magic 1: The Isolated Line. 
                # If it's short and surrounded by empty space, it's a slide title.
                if prev_blank and next_blank and len(line) <= 80:
                    is_heading = True
                
                # Old Code Magic 2: Title Case Density
                words = line.split()
                if not is_heading and len(words) >= 2 and len(line) <= 80:
                    capitalized = sum(1 for w in words if w and (w[0].isupper() or w.isupper()))
                    if capitalized / len(words) >= 0.6:
                        is_heading = True
                        
                # Old Code Magic 3: Short Colon Endings
                if not is_heading and line.endswith(":") and len(line) <= 80:
                    is_heading = True

            # === FINAL FILTERS ===
            if is_heading:
                if len(line.split()) > self.MAX_HEADING_WORDS:
                    is_heading = False
                if line.endswith(".") and not re.match(r"^\d", line):
                    is_heading = False
                if line.startswith("•") or line.startswith("-") or line.startswith("*"):
                    is_heading = False

            if is_heading:
                key = line.lower()
                if key not in seen:
                    seen.add(key)
                    heading_lines.append(line)
                    consecutive_non_headings = 0

            # Update the blank tracker for the next loop iteration
            prev_blank = False

        return "\n".join(heading_lines)

    # =============================================================
    # SCENARIO-SPECIFIC PROMPTS
    # =============================================================

    def _build_toc_prompt(self, text: str, max_topics: int = None) -> str:
        """
        Prompt optimized for exact Table of Contents parsing.
        Handles English and Arabic ToC entries.
        """
        max_constraint = f"\n- LIMIT: Extract at most {max_topics} top-level topics." if max_topics else ""
        return f"""You are parsing a Table of Contents from an academic document. Extract the COMPLETE hierarchical structure.

CRITICAL RULES:
1. DO NOT STOP EARLY - process EVERY entry from start to finish
2. Preserve exact title wording (do not paraphrase)
3. Support both English and Arabic entries
4. Maximum 2 levels: topics and subtitles only
5. Write a brief 1-sentence description for each item
6. Output MUST be valid JSON matching the schema exactly
{max_constraint}

TABLE OF CONTENTS:
{text}

OUTPUT JSON SCHEMA (strict - no extra fields):
{{
  "topics": [
    {{
      "title": "exact title from ToC",
      "description": "one sentence description",
      "order": 0,
      "subtitles": [
        {{"title": "subtitle", "description": "one sentence", "order": 0}}
      ]
    }}
  ]
}}

Return ONLY the JSON object, no markdown, no explanations."""

    def _build_book_headings_prompt(self, text: str, max_topics: int = None) -> str:
        """
        Prompt optimized for reconstructing book structure from extracted headings.
        Handles chapter/section numbering and bilingual content.
        """
        max_constraint = f"\n- LIMIT: Extract at most {max_topics} top-level topics." if max_topics else ""
        return f"""You are reconstructing a textbook's hierarchical structure from extracted headings.

CRITICAL RULES:
1. DO NOT STOP EARLY - this is a full book, read ALL headings to the end
2. Group sub-numbered headings (1.1, 1.2, 2.1, etc.) as subtitles under their parent chapter
3. Ignore: citations, academic years (2023-2024), standalone variables, page numbers
4. Preserve exact title wording in original language (English or Arabic)
5. Write a brief 1-sentence description for each item
6. Output MUST be valid JSON matching the schema exactly
{max_constraint}

EXTRACTED HEADINGS:
{text}

OUTPUT JSON SCHEMA (strict - no extra fields):
{{
  "topics": [
    {{
      "title": "chapter or main section title",
      "description": "one sentence description",
      "order": 0,
      "subtitles": [
        {{"title": "subsection title", "description": "one sentence", "order": 0}}
      ]
    }}
  ]
}}

Return ONLY the JSON object, no markdown, no explanations."""

    def _build_lecture_prompt(self, text: str, max_topics: int = None) -> str:
        """
        Prompt optimized for sparse, messy lecture presentation slides.
        Handles flat structures and fragmented content.
        """
        max_constraint = f"\n- LIMIT: Extract at most {max_topics} top-level topics." if max_topics else ""
        return f"""You are reconstructing a lecture presentation structure from slide titles.

CRITICAL RULES:
1. Extract main concepts as top-level topics, sub-concepts as subtitles
2. If structure is flat, return all titles as top-level topics with empty subtitles []
3. IGNORE: citations, academic years (2023-2024), math variables (Px, Py), URLs, instructions
4. Do NOT invent topics - only use what appears in the text
5. Support both English and Arabic content
6. Write a brief 1-sentence description for each item
7. Output MUST be valid JSON matching the schema exactly
{max_constraint}

SLIDE TITLES:
{text}

OUTPUT JSON SCHEMA (strict - no extra fields):
{{
  "topics": [
    {{
      "title": "main concept or slide title",
      "description": "one sentence description",
      "order": 0,
      "subtitles": [
        {{"title": "sub-concept", "description": "one sentence", "order": 0}}
      ]
    }}
  ]
}}

Return ONLY the JSON object, no markdown, no explanations."""

    # =============================================================
    # PARSER & NORMALIZER
    # =============================================================

    def _parse_structure_response(self, response: str) -> dict:
        """
        Parses LLM response with robust JSON extraction and repair.
        Handles markdown wrappers, truncated JSON, and common malformations.
        """
        if not response:
            return self._create_fallback_structure()

        response = response.strip()


        # >>> ADD THESE THREE LINES TO TURN THE LIGHTS BACK ON <<<
        print("\n================ LLM FULL RESPONSE ================\n")
        print(response)
        print("\n==================================================\n")

        
        # Extract JSON from markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            parts = response.split("```")
            response = parts[1].strip() if len(parts) >= 3 else response.replace("```", "").strip()

        # Find JSON boundaries
        first_brace, last_brace = response.find("{"), response.rfind("}")
        first_bracket, last_bracket = response.find("["), response.rfind("]")

        # Determine if object or array root
        if first_brace == -1 and first_bracket == -1:
            self.logger.warning("No JSON structure found in response")
            return self._create_fallback_structure()

        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            json_str = response[first_brace:last_brace + 1] if last_brace > first_brace else response[first_brace:]
        else:
            json_str = response[first_bracket:last_bracket + 1] if last_bracket > first_bracket else response[first_bracket:]

        # Try parsing with repair attempts
        structure = self._try_parse_json(json_str)
        if not structure:
            return self._create_fallback_structure()

        # Wrap array in object
        if isinstance(structure, list):
            structure = {"topics": structure}

        if "topics" not in structure:
            self.logger.warning("No 'topics' key in parsed structure")
            return self._create_fallback_structure()

        # Validate and clean topics
        valid_topics = self._validate_topics(structure["topics"])
        if not valid_topics:
            return self._create_fallback_structure()

        structure["topics"] = valid_topics
        return structure

    def _try_parse_json(self, json_str: str) -> Optional[Any]:
        """
        Attempts to parse JSON with multiple repair strategies.
        """
        json_str = json_str.strip()

        # Direct parse attempt
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Strategy 1: Add missing closing brackets
        for suffix in ["]}]}", "]}", "}]}", "]}", "}"]:
            try:
                return json.loads(json_str + suffix)
            except json.JSONDecodeError:
                continue

        # Strategy 2: Truncate at last complete object
        last_complete = json_str.rfind("},")
        if last_complete > 0:
            try:
                return json.loads(json_str[:last_complete + 1] + "]}")
            except json.JSONDecodeError:
                pass

        # Strategy 3: Remove trailing commas
        cleaned = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 4: Fix common quote issues (single to double)
        try:
            return json.loads(json_str.replace("'", '"'))
        except json.JSONDecodeError:
            pass

        self.logger.warning(f"All JSON repair attempts failed for: {json_str[:100]}...")
        return None

    def _validate_topics(self, topics: Any) -> List[dict]:
        """
        Validates and cleans topics list, filtering out invalid entries.
        """
        if not isinstance(topics, list):
            return []

        valid_topics = []
        bad_titles = {"untitled", "none", "null", "document content", "main section", "unknown", "title"}

        for i, topic in enumerate(topics):
            if not isinstance(topic, dict):
                continue

            title = str(topic.get("title", "")).strip()
            if not title or title.lower() in bad_titles:
                continue

            # Skip duplicates
            if title.lower() in {t["title"].lower() for t in valid_topics}:
                continue

            # Process subtitles
            subtitles = []
            for j, sub in enumerate(topic.get("subtitles", [])):
                if not isinstance(sub, dict):
                    continue
                sub_title = str(sub.get("title", "")).strip()
                if not sub_title or sub_title.lower() in bad_titles:
                    continue
                subtitles.append({
                    "title": sub_title,
                    "description": str(sub.get("description", "")).strip() or f"Subtopic of {title}",
                    "order": sub.get("order", j) if isinstance(sub.get("order"), int) else j
                })

            valid_topics.append({
                "title": title,
                "description": str(topic.get("description", "")).strip() or f"This section covers {title}",
                "order": topic.get("order", i) if isinstance(topic.get("order"), int) else i,
                "subtitles": subtitles
            })

        return valid_topics[:50]  # Cap at 50 topics to prevent runaway
    

    def _attempt_json_repair(self, json_str: str) -> Optional[dict]:
        """Legacy compatibility wrapper - now delegates to _try_parse_json."""
        return self._try_parse_json(json_str)

    def _create_fallback_structure(self) -> dict:
        """Returns a safe fallback structure when all extraction attempts fail."""
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

    def normalize_structure(self, raw_structure: dict) -> List[Dict[str, Any]]:
        """
        Converts LLM structure to flat backend schema with temp_ids for database insertion.
        Output format: [{temp_id, title, description, order_index, parent_temp_id}, ...]
        """
        normalized = []

        if not raw_structure or "topics" not in raw_structure:
            return normalized

        topic_counter = 0

        for topic in raw_structure["topics"]:
            title = topic.get("title", "").strip()
            if not title:
                continue

            topic_counter += 1
            topic_temp_id = f"topic_{topic_counter}"

            normalized.append({
                "temp_id": topic_temp_id,
                "title": title,
                "description": topic.get("description", "").strip() or f"This section covers {title}",
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
                    "description": subtitle.get("description", "").strip() or f"Subtopic under {title}",
                    "order_index": topic_counter - 1,
                    "parent_temp_id": topic_temp_id
                })

        return normalized