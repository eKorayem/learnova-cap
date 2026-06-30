from models.ChunkModel import ChunkModel
from controllers.BaseController import BaseController
from typing import List, Optional, Dict, Any, Tuple
import json
import re
import logging
import time

class StructureController(BaseController):
    """
    Universal Parser for academic document structure extraction.
    Handles: Textbooks (with/without ToC), Lecture Slides, Bilingual (EN/AR) content.
    """

    def __init__(self, generation_client=None):
        super().__init__()
        self.generation_client = generation_client
        self.logger = logging.getLogger('uvicorn.error')
        self.logger.setLevel(logging.INFO)

        # Document type detection thresholds
        self.BOOK_CHUNK_THRESHOLD = 30
        self.BOOK_CHAR_THRESHOLD = 30000
        self.TOC_SCAN_LIMIT = 25000
        self.TOC_BLOCK_SIZE = 6000
        
        # ==========================================
        # GROQ SAFETY FIX 1: Clamp Input Characters
        # Reduced to 15,000 to guarantee it stays under ~4,000 input tokens
        # ==========================================
        self.MAX_LLM_INPUT_CHARS = 15000 
        
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
        asset_id: str = None, 
        use_all_chunks: bool = False
    ) -> dict:
        
        start_time = time.time()

        self.logger.info(f"========== [STARTED] STRUCTURE EXTRACTION FOR PROJECT {project_id} ==========")
        
        # 1. Fetch chunks
        chunks = await chunk_model.get_chunks_by_project_id(
            project_id=project_id, page_no=1, page_size=3000, chunk_type="structure", asset_id=asset_id
        )

        if not chunks:
            self.logger.error("DEBUG: 0 Chunks found. Aborting.")
            return self._create_fallback_structure()

        chunks = sorted(chunks, key=lambda c: c.chunk_order)
        total_chunks = len(chunks)

        # Inject the physical page number into the text stream so the LLM can see it
        text_blocks = []
        for c in chunks:
            page_num = c.chunk_metadata.get('page')
            if page_num is not None:
                text_blocks.append(f"--- [PAGE {page_num + 1}] ---\n{c.chunk_text}")
            else:
                text_blocks.append(c.chunk_text)
                
        full_text = "\n\n".join(text_blocks)
        
        self.logger.info(f"DEBUG: Fetched {total_chunks} chunks. Total raw characters: {len(full_text)}")

        # 2. Universal Detection
        doc_type = self._detect_document_type(full_text, total_chunks)
        toc_text = self._extract_potential_toc(full_text)
        has_toc = toc_text is not None

        # 3. Dynamic Extraction Routing
        if doc_type == "lecture" or doc_type in ["slide", "presentation"]:
            strategy = "Slide Reading (Full Document)"
            
            llm_input_lines = []
            for c in chunks:
                page_num = c.chunk_metadata.get('page', 0) + 1
                text_snippet = str(c.chunk_text)[:400].strip()
                if text_snippet:
                    llm_input_lines.append(f"--- [PAGE {page_num}] ---\n{text_snippet}")
            
            llm_input = "\n\n".join(llm_input_lines)
            prompt = self._build_lecture_prompt(llm_input, max_topics)
            
        elif has_toc:
            llm_input = toc_text
            prompt = self._build_toc_prompt(llm_input, max_topics)
            strategy = "Table of Contents"
            
        else:
            llm_input = self._extract_headings_only(full_text, doc_type)
            
            if len(llm_input.strip()) < 1000:
                self.logger.warning("Book strategy yielded too little text. Falling back to lecture strategy.")
                strategy = "Slide Reading (Fallback)"
                llm_input_lines = []
                for c in chunks:
                    page_num = c.chunk_metadata.get('page', 0) + 1
                    text_snippet = str(c.chunk_text)[:400].strip()
                    if text_snippet:
                        llm_input_lines.append(f"--- [PAGE {page_num}] ---\n{text_snippet}")
                llm_input = "\n\n".join(llm_input_lines)
                prompt = self._build_lecture_prompt(llm_input, max_topics)
            else:
                prompt = self._build_book_headings_prompt(llm_input, max_topics)
                strategy = "Book Headings"

        # Hard cap input to prevent Groq from panicking on giant PDFs
        if len(llm_input) > self.MAX_LLM_INPUT_CHARS:
            llm_input = llm_input[:self.MAX_LLM_INPUT_CHARS]

        approx_in_tokens = len(prompt) // 4

        # 4. Generate AI Prompt
        try:
            llm_start_time = time.time()
            
            # ==========================================
            # GROQ SAFETY FIX 2: Clamp Output Tokens
            # We hardcode max_output_tokens to 2500 so Groq doesn't reserve 8192+ upfront!
            # ==========================================
            response = self.generation_client.generate_text(
                prompt=prompt,
                temperature=self.app_settings.STRUCTURE_TEMPERATURE,
                max_output_tokens=2500 # <--- THIS FIXES THE CRASH
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
        asset_id: str = None,
        max_topics: int = None,
        use_all_chunks: bool = False
    ) -> tuple:
        raw_structure = await self.analyze_lecture_structure(
            chunk_model=chunk_model, project_id=project_id, asset_id=asset_id,
            max_topics=max_topics, use_all_chunks=use_all_chunks
        )
        normalized = self.normalize_structure(raw_structure)
        return normalized, "completed" if normalized else "failed"

    # =============================================================
    # UNIVERSAL DETECTION
    # =============================================================

    def _detect_document_type(self, text: str, total_chunks: int) -> str:
        text_lower = text.lower()
        
        book_patterns = [
            r"\bchapter\s+\d+\b", r"\bchapter\s+[ivxlcdm]+\b", r"\bpart\s+[ivxlcdm\d]+\b",
            r"\bunit\s+\d+\b", r"\btable\s+of\s+contents?\b",
            r"\bالفصل\s+[\dأ-ي]+\b", r"\bالباب\s+[\dأ-ي]+\b", r"\bالجزء\s+[\dأ-ي]+\b"
        ]
        book_hits = sum(1 for p in book_patterns if re.search(p, text_lower, re.IGNORECASE))

        lecture_hits = len(re.findall(r"\blecture\b|\bslide\b|\bpresentation\b", text_lower[:10000]))
        
        if lecture_hits > 3:
            return "lecture" 

        if book_hits >= 2 or (total_chunks > 200 and len(text) > 200000):
            return "book"
            
        return "lecture"

    def _extract_potential_toc(self, text: str) -> Optional[str]:
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
            numbered = re.findall(r"^\s*\d+[\.\)]\s+[A-Zأ-ي][^\n]{3,80}$", scan, re.MULTILINE)
            if len(numbered) >= 5:
                return "\n".join(numbered[:100])
            return None

        remaining = scan[toc_header_end:]
        toc_block = remaining[:15000]

        toc_lines = []
        for line in toc_block.split("\n"):
            line = line.strip()
            if re.match(r"^---\s*\[PAGE\s+(\d+)\]\s*---$", line, re.IGNORECASE):
                continue
            if line and len(line) > 2:
                toc_lines.append(line)

        result = "\n".join(toc_lines)
        return result if len(result) > 50 else None

    # =============================================================
    # HEADINGS SCRUBBER (Used for Books Only)
    # =============================================================

    def _normalize_line(self, line: str) -> str:
        line = line.replace("\u00a0", " ").replace("\u2002", " ").replace("\u2003", " ").replace("\u2009", " ")
        line = line.replace("\ufeff", " ")
        line = re.sub(r"\s+", " ", line.strip())
        line = re.sub(r"^\d{1,4}\s+", "", line)
        return line.strip()

    def _looks_like_question_or_exercise(self, line: str) -> bool:
        if not line:
            return False
        lower = line.lower().strip()

        if line.endswith("?") or line.endswith("؟"):
            return True

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

        ar_question_words = ["ماذا", "متى", "أين", "كيف", "لماذا", "هل", "من", "ما"]
        for word in ar_question_words:
            if word in line:
                return True

        return False

    def _is_noise_or_non_structure(self, line: str) -> bool:
        if not line:
            return True

        lower = line.lower().strip()

        if len(line) < self.HEADING_MIN_LENGTH or len(line) > self.HEADING_MAX_LENGTH:
            return True

        if re.fullmatch(r"\d+", line):
            return True
        if re.fullmatch(r"\d+(?:\.\d+)+", line):
            return True
        if re.fullmatch(r"\d{4}\s*-\s*\d{4}", line): 
            return True
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", line): 
            return True

        if len(line) > 0 and 'a' <= line[0] <= 'z':
            return True

        if re.search(r"[=+\-*/<>$≈≠≤≥∑∫∞]", line):
            return True
        if re.match(r"^([A-Za-zأ-ي]\s*,\s*)+[A-Za-zأ-ي]$", line):
            return True
            
        alpha_count = sum(1 for c in line if c.isalpha())
        if len(line) > 0 and (alpha_count / len(line)) < 0.5:
            return True
        if re.match(r"^\d+\.\d+\s*$", line):
            return True

        if "http" in lower or "www." in lower or "@" in line:
            return True
        if lower.startswith("isbn") or lower.startswith("issn"):
            return True

        noise_keywords = [
            "point", "key point", "checkpoint", "check point", "note", "tip", "caution",
            "pedagogical note", "videonote", "figure", "table ", "listing", "chapter summary",
            "key terms", "quiz", "programming exercises", "supplement", "source:",
            "copyright", "all rights reserved", "problems", "problem set", "learning objectives",
            "references", "bibliography", "acknowledgements", "preface", "foreword",
            "dr.", "prof.", "professor", "faculty of", "university", "college of", "department of",
            "spring", "fall", "summer", "winter", "semester", "lecture", "session",
            "thank you", "questions?", "q&a", "any questions", "agenda", "dr.", "dr ",
            "prof.", "professor", "faculty of", "university", "college of", "thank you",
            "ملاحظة", "تنبيه", "شكل", "جدول", "حقوق الطبع", "ملخص", "مراجع",
            "شكر", "تمهيد", "إهداء", "غلاف", "دكتور", "أستاذ", "كلية", "جامعة", "قسم", "شكرا", "الأسئلة"
        ]

        for keyword in noise_keywords:
            if keyword in lower or lower.startswith(keyword):
                return True

        if self._looks_like_question_or_exercise(line):
            return True

        if line.endswith(":"):
            words = line.split()
            if len(words) > 5:
                return True

        if line.isupper() and len(line) < 10 and " " not in line:
            return True

        return False

    def _extract_headings_only(self, text: str, doc_type: str) -> str:
        lines = text.split("\n")
        heading_lines = []
        seen = set()
        consecutive_non_headings = 0
        prev_blank = True
        current_page = None

        for i, raw in enumerate(lines):
            line = self._normalize_line(raw)
            next_blank = (i + 1 >= len(lines)) or (self._normalize_line(lines[i + 1]) == "")

            page_match = re.match(r"^---\s*\[PAGE\s+(\d+)\]\s*---$", line, re.IGNORECASE)
            if page_match:
                current_page = page_match.group(1)
                continue

            if not line:
                consecutive_non_headings = 0
                prev_blank = True
                continue

            if self._is_noise_or_non_structure(line):
                consecutive_non_headings += 1
                prev_blank = False 
                continue

            is_heading = False

            if re.match(r"^(chapter|section|part|unit|topic|module|lecture|الفصل|الباب|الوحدة|الدرس|الجزء)\s+[\d\wأ-ي]+", line, re.IGNORECASE):
                is_heading = True
            elif re.match(r"^\d+(\.\d+)*\.?\s+[A-Zأ-ي]", line):
                is_heading = True
            elif re.match(r"^[IVXLCDM]+\.\s+[A-Z]", line):
                is_heading = True
            elif re.match(r"^[\u0660-\u0669]+[\.\)]\s+[أ-ي]", line):
                is_heading = True
            elif doc_type == "lecture":
                if prev_blank and next_blank and len(line) <= 80:
                    is_heading = True
                words = line.split()
                if not is_heading and len(words) >= 2 and len(line) <= 80:
                    capitalized = sum(1 for w in words if w and (w[0].isupper() or w.isupper()))
                    if capitalized / len(words) >= 0.6:
                        is_heading = True
                if not is_heading and line.endswith(":") and len(line) <= 80:
                    is_heading = True

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
                    if current_page:
                        heading_lines.append(f"{line} (Page {current_page})")
                    else:
                        heading_lines.append(line)
                    consecutive_non_headings = 0
            
            prev_blank = False

        return "\n".join(heading_lines)

    # =============================================================
    # SCENARIO-SPECIFIC PROMPTS
    # =============================================================

    def _build_toc_prompt(self, text: str, max_topics: int = None) -> str:
        max_constraint = f"\n- LIMIT: Extract at most {max_topics} top-level topics." if max_topics else ""
        return f"""You are parsing a Table of Contents from an academic document. Extract the COMPLETE hierarchical structure.

CRITICAL RULES:
1. DO NOT STOP EARLY - process EVERY entry from start to finish
2. Preserve exact title wording (do not paraphrase)
3. Support both English and Arabic entries
4. Maximum 2 levels: topics and subtitles only
5. Write a brief 1-sentence description for each item
6. Output MUST be valid JSON matching the schema exactly
7. Extract 'page_start' and 'page_end' integers if visible in the text (e.g., from ToC numbers or "(Page X)" labels).
8. 'page_end' is typically the page before the next topic starts. If unknown, output null.
{max_constraint}

TABLE OF CONTENTS:
{text}

OUTPUT JSON SCHEMA (strict - no extra fields):
{{
  "topics": [
    {{
      "title": "exact title",
      "description": "one sentence description",
      "order": 0,
      "page_start": 1,
      "page_end": 5,
      "subtitles": [
        {{"title": "subtitle", "description": "one sentence", "order": 0, "page_start": 1, "page_end": 5}}
      ]
    }}
  ]
}}

Return ONLY the JSON object, no markdown, no explanations."""

    def _build_book_headings_prompt(self, text: str, max_topics: int = None) -> str:
        max_constraint = f"\n- LIMIT: Extract at most {max_topics} top-level topics." if max_topics else ""
        return f"""You are reconstructing a textbook's hierarchical structure from extracted headings.

CRITICAL RULES:
1. DO NOT STOP EARLY - this is a full book, read ALL headings to the end
2. Group sub-numbered headings (1.1, 1.2, 2.1, etc.) as subtitles under their parent chapter
3. Ignore: citations, academic years (2023-2024), standalone variables, page numbers
4. Preserve exact title wording in original language (English or Arabic)
5. Write a brief 1-sentence description for each item
6. Output MUST be valid JSON matching the schema exactly
7. Extract 'page_start' and 'page_end' integers if visible in the text (e.g., from ToC numbers or "(Page X)" labels).
8. 'page_end' is typically the page before the next topic starts. If unknown, output null.
{max_constraint}

EXTRACTED HEADINGS:
{text}

OUTPUT JSON SCHEMA (strict - no extra fields):
{{
  "topics": [
    {{
      "title": "exact title",
      "description": "one sentence description",
      "order": 0,
      "page_start": 1,
      "page_end": 5,
      "subtitles": [
        {{"title": "subtitle", "description": "one sentence", "order": 0, "page_start": 1, "page_end": 5}}
      ]
    }}
  ]
}}

Return ONLY the JSON object, no markdown, no explanations."""

    def _build_lecture_prompt(self, text: str, max_topics: int = None) -> str:
        max_constraint = f"\n- LIMIT: Extract at most {max_topics} top-level topics." if max_topics else ""
        return f"""You are analyzing an academic Lecture/Slide Deck. Extract the main topics.

CRITICAL RULES:
1. Read the provided slides. Each slide snippet is marked with a --- [PAGE X] --- tag.
2. Group the slides into logical overarching topics (and subtitles if applicable).
3. You MUST use the numbers inside the --- [PAGE X] --- tags to determine the 'page_start' and 'page_end' for each topic!
4. 'page_end' is the slide number right before the next topic begins.
5. Output MUST be valid JSON matching the schema exactly.
{max_constraint}

LECTURE SLIDES:
{text}

OUTPUT JSON SCHEMA (strict - no extra fields):
{{
  "topics": [
    {{
      "title": "exact title",
      "description": "one sentence description",
      "order": 0,
      "page_start": 1,
      "page_end": 5,
      "subtitles": [
        {{"title": "subtitle", "description": "one sentence", "order": 0, "page_start": 1, "page_end": 5}}
      ]
    }}
  ]
}}

Return ONLY the JSON object, no markdown, no explanations."""
    
    # =============================================================
    # PARSER & NORMALIZER
    # =============================================================

    def _parse_structure_response(self, response: str) -> dict:
        if not response:
            return self._create_fallback_structure()

        response = response.strip()

        print("\n================ LLM FULL RESPONSE ================\n")
        print(response)
        print("\n==================================================\n")

        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            parts = response.split("```")
            response = parts[1].strip() if len(parts) >= 3 else response.replace("```", "").strip()

        first_brace, last_brace = response.find("{"), response.rfind("}")
        first_bracket, last_bracket = response.find("["), response.rfind("]")

        if first_brace == -1 and first_bracket == -1:
            self.logger.warning("No JSON structure found in response")
            return self._create_fallback_structure()

        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            json_str = response[first_brace:last_brace + 1] if last_brace > first_brace else response[first_brace:]
        else:
            json_str = response[first_bracket:last_bracket + 1] if last_bracket > first_bracket else response[first_bracket:]

        structure = self._try_parse_json(json_str)
        if not structure:
            return self._create_fallback_structure()

        if isinstance(structure, list):
            structure = {"topics": structure}

        if "topics" not in structure:
            self.logger.warning("No 'topics' key in parsed structure")
            return self._create_fallback_structure()

        valid_topics = self._validate_topics(structure["topics"])
        if not valid_topics:
            return self._create_fallback_structure()

        structure["topics"] = valid_topics
        return structure

    def _try_parse_json(self, json_str: str) -> Optional[Any]:
        json_str = json_str.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        for suffix in ["]}]}", "]}", "}]}", "]}", "}"]:
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

        cleaned = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        try:
            return json.loads(json_str.replace("'", '"'))
        except json.JSONDecodeError:
            pass

        self.logger.warning(f"All JSON repair attempts failed for: {json_str[:100]}...")
        return None

    def _validate_topics(self, topics: Any) -> List[dict]:
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

            if title.lower() in {t["title"].lower() for t in valid_topics}:
                continue

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
                    "order": sub.get("order", j) if isinstance(sub.get("order"), int) else j,
                    "page_start": sub.get("page_start"),  
                    "page_end": sub.get("page_end")       
                })

            valid_topics.append({
                "title": title,
                "description": str(topic.get("description", "")).strip() or f"This section covers {title}",
                "order": topic.get("order", i) if isinstance(topic.get("order"), int) else i,
                "subtitles": subtitles,
                "page_start": topic.get("page_start"),  
                "page_end": topic.get("page_end")       
            })

        return valid_topics[:50]  

    def _attempt_json_repair(self, json_str: str) -> Optional[dict]:
        return self._try_parse_json(json_str)

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

    def normalize_structure(self, raw_structure: dict) -> List[Dict[str, Any]]:
        normalized = []
        if not raw_structure or "topics" not in raw_structure:
            return self._get_safe_fallback()

        topic_counter = 0

        for topic in raw_structure["topics"]:
            title = str(topic.get("title", "")).strip()
            if not title:
                continue

            # DB SANITIZER: Prevent 500 Errors
            title = title[:250]

            t_start = topic.get("page_start")
            t_end = topic.get("page_end")
            
            try: t_start = int(t_start) if t_start is not None else None
            except: t_start = None
            try: t_end = int(t_end) if t_end is not None else None
            except: t_end = None

            if t_start is not None and t_start <= 0: t_start = None
            if t_end is not None and t_end <= 0: t_end = None

            if t_start is None or t_end is None or t_start > t_end:
                t_start, t_end = None, None

            has_sub_pages = False
            valid_subtitles = []
            
            for sub in topic.get("subtitles", []):
                s_title = str(sub.get("title", "")).strip()
                if not s_title: continue
                
                # DB SANITIZER: Prevent 500 Errors
                s_title = s_title[:250]
                
                s_start = sub.get("page_start")
                s_end = sub.get("page_end")
                try: s_start = int(s_start) if s_start is not None else None
                except: s_start = None
                try: s_end = int(s_end) if s_end is not None else None
                except: s_end = None
                
                if s_start is not None and s_start <= 0: s_start = None
                if s_end is not None and s_end <= 0: s_end = None

                if s_start is None or s_end is None or s_start > s_end:
                    s_start, s_end = None, None
                else:
                    has_sub_pages = True
                    
                sub["_clean_start"] = s_start
                sub["_clean_end"] = s_end
                sub["_clean_title"] = s_title
                valid_subtitles.append(sub)

            if has_sub_pages:
                min_start = min([s["_clean_start"] for s in valid_subtitles if s["_clean_start"] is not None], default=t_start)
                max_end = max([s["_clean_end"] for s in valid_subtitles if s["_clean_end"] is not None], default=t_end)
                
                t_start = min(t_start, min_start) if t_start is not None else min_start
                t_end = max(t_end, max_end) if t_end is not None else max_end
                
            if t_start is None or t_end is None:
                t_start, t_end = None, None
                for sub in valid_subtitles:
                    sub["_clean_start"] = None
                    sub["_clean_end"] = None

            topic_counter += 1
            topic_temp_id = f"topic_{topic_counter}"

            # DB SANITIZER: Prevent Description 500 Errors
            desc = str(topic.get("description", "")).strip() or f"This section covers {title}"
            desc = desc[:500] 

            normalized.append({
                "temp_id": topic_temp_id,
                "title": title,
                "description": desc,
                "order_index": int(topic_counter - 1),
                "parent_temp_id": None,
                "page_start": t_start,
                "page_end": t_end
            })

            for subtitle in valid_subtitles:
                topic_counter += 1
                
                # DB SANITIZER: Prevent Description 500 Errors
                sub_desc = str(subtitle.get("description", "")).strip() or f"Subtopic under {title}"
                sub_desc = sub_desc[:500]
                
                normalized.append({
                    "temp_id": f"topic_{topic_counter}",
                    "title": subtitle["_clean_title"],
                    "description": sub_desc,
                    "order_index": int(topic_counter - 1),
                    "parent_temp_id": topic_temp_id,
                    "page_start": subtitle["_clean_start"],
                    "page_end": subtitle["_clean_end"]
                })

        if not normalized:
             return self._get_safe_fallback()

        return normalized

    def _get_safe_fallback(self) -> List[Dict[str, Any]]:
        return [{
            "temp_id": "topic_1",
            "title": "Document Content",
            "description": "Content extracted from the uploaded document.",
            "order_index": 0,
            "parent_temp_id": None,
            "page_start": None,
            "page_end": None
        }]