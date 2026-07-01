import logging
import asyncio
import re
from fastapi import APIRouter, Request, BackgroundTasks, status, Depends
from fastapi.responses import JSONResponse

from routes.schemas.extraction import ExtractionWebhookPayload
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel
from models.AssetModel import AssetModel
from models.enums.AssetTypeEnum import AssetTypeEnum
from controllers.ExtractionController import ExtractionController
from core.security.dependencies import verify_backend_signature
from core.security.callback import send_webhook_callback

logger = logging.getLogger('uvicorn.error')

extraction_router = APIRouter(
    prefix="/api/v1/courses",
    tags=["api_v1", "extraction"]
)

def _contains_questions(text: str) -> bool:
    """Heuristic filter to determine if a chunk likely contains questions/exercises.

    NOTE: this used to trigger on a single stray '?' anywhere in the chunk,
    which matches almost any real document (rhetorical questions in body
    text, section headers, etc.) and defeated the point of filtering.
    We now require either a strong keyword/pattern signal, OR multiple
    question marks close together (a much better proxy for an actual
    question list than a single '?' in normal prose).
    """
    lower_text = text.lower()

    en_keywords = ["multiple choice", "true/false", "true / false", "short answer",
                   "review questions", "chapter quiz", "knowledge check", "exercises",
                   "practice problems", "test your knowledge", "fill in the blank", "quiz:", "practice"]
    if any(kw in lower_text for kw in en_keywords): return True

    ar_keywords = ["سؤال", "أسئلة", "تمرين", "تمارين", "اختر الإجابة", "صح أم خطأ", "اختبار"]
    if any(kw in lower_text for kw in ar_keywords): return True

    mcq_pattern = r"(?m)^\s*[\(]?[a-eA-E][\)\.]\s+[A-Za-z]"
    if len(re.findall(mcq_pattern, text)) >= 2: return True

    # A numbered line ending in a question mark is a strong "quiz item" signal.
    numbered_question_pattern = r"(?m)^\s*\d+[\.\)]\s+.*[?؟]\s*$"
    if re.search(numbered_question_pattern, text): return True

    # Multiple question marks in one chunk is a much stronger signal than a
    # single one — real question/exercise blocks tend to cluster several
    # questions together, while ordinary prose rarely has more than one.
    question_mark_count = text.count('?') + text.count('؟')
    if question_mark_count >= 3: return True

    if "___" in text: return True

    return False


async def _extract_questions_background(app, project, payload: ExtractionWebhookPayload):
    logger.info(f"Background Task: Extracting native questions for material {payload.body.material_id}")
    
    try:
        chunk_model = await ChunkModel.create_instance(db_client=app.db_client)
        asset_model = await AssetModel.create_instance(db_client=app.db_client)
        
        project_files = await asset_model.get_all_projects_assets(
            asset_project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value
        )
        
        target_asset_id = None
        for asset in project_files:
            if asset.asset_config and asset.asset_config.get("material_id") == payload.body.material_id:
                target_asset_id = asset.id
                break

        chunks = await chunk_model.get_chunks_by_project_id(
            project_id=project.project_id,
            page_no=1,
            page_size=10000, 
            chunk_type="question",
            asset_id=target_asset_id
        )

        if not chunks:
            raise ValueError("No text chunks found for this material.")

        chunks = sorted(chunks, key=lambda c: c.chunk_order)

        # ==========================================
        # SMART CHUNK FILTERING
        # ==========================================
        candidate_indices = set()
        for i, c in enumerate(chunks):
            if _contains_questions(c.chunk_text):
                candidate_indices.add(i)
                if i < len(chunks) - 1: candidate_indices.add(i + 1)
                
        filtered_chunks = [chunks[i] for i in sorted(list(candidate_indices))]
        logger.info(f"SMART FILTER: Reduced document from {len(chunks)} chunks down to just {len(filtered_chunks)} relevant chunks!")

        if not filtered_chunks:
            logger.info("No questions detected. Bypassing LLM to save tokens.")
            await send_webhook_callback(
                request_id=payload.request_id,
                course_id=payload.course_id,
                operation_type=payload.operation_type,
                status="success",
                message="Processed successfully. Document contains no questions.",
                module_id=payload.body.module_id,
                material_id=payload.body.material_id,
                data={
                    "extracted_questions": [],
                    "destination": getattr(payload.body, "destination", None) # <--- ADDED DESTINATION
                } 
            )
            return

        # ==========================================
        # GROQ-SAFE SEQUENTIAL EXTRACTION
        # ==========================================
        controller = ExtractionController(generation_client=app.qgen_client)
        
        CHUNKS_PER_BATCH = 15  
        batches = [filtered_chunks[i:i + CHUNKS_PER_BATCH] for i in range(0, len(filtered_chunks), CHUNKS_PER_BATCH)]
        logger.info(f"Executing LLM Extraction in {len(batches)} sequential Groq-safe batches...")

        all_extracted_questions = []

        for idx, batch_chunks in enumerate(batches):
            batch_text = "\n\n".join([c.chunk_text for c in batch_chunks])
            
            for attempt in range(3):
                try:
                    res = await controller.extract_native_questions(
                        document_text=batch_text,
                        topics=payload.body.topics
                    )
                    if isinstance(res, dict) and "extracted_questions" in res:
                        extracted = res["extracted_questions"]
                        if len(extracted) == 0:
                            logger.info(f"Batch {idx + 1}/{len(batches)} complete. 0 questions found.")
                        else:
                            logger.info(f"✅ Batch {idx + 1}/{len(batches)} complete. EXTRACTED {len(extracted)} QUESTIONS.")
                        all_extracted_questions.extend(extracted)
                        break # Success!
                except Exception as e:
                    logger.error(f"Batch {idx + 1} failed on attempt {attempt + 1}: {e}")
                    await asyncio.sleep(5)
            
            if idx < len(batches) - 1:
                logger.info("Sleeping 15s to reset Groq's TPM Token Window...")
                await asyncio.sleep(15)

        # ==========================================
        # THE STRICT SANITIZER
        # ==========================================
        valid_topic_ids = [t.id for t in payload.body.topics]
        fallback_topic = valid_topic_ids[0] if valid_topic_ids else 0

        sanitized_questions = []
        for q in all_extracted_questions:
            try:
                tid = q.get("topic_id")
                try: tid = int(tid)
                except: tid = fallback_topic
                if tid not in valid_topic_ids: tid = fallback_topic
                q["topic_id"] = tid

                q_type = str(q.get("type", "short_answer")).lower().replace(" ", "_")
                if "multiple" in q_type or "mcq" in q_type: q["type"] = "multiple_choice"
                elif "true" in q_type or "tf" in q_type: q["type"] = "true_false"
                elif "essay" in q_type: q["type"] = "essay"
                else: q["type"] = "short_answer"

                diff = str(q.get("difficulty", "medium")).lower()
                if diff not in ["easy", "medium", "hard"]: diff = "medium"
                q["difficulty"] = diff

                if not q.get("expected_answer"):
                    q["expected_answer"] = "Please refer to rubric or course material."
                q["expected_answer"] = str(q["expected_answer"])

                if q["type"] in ["multiple_choice", "true_false"]:
                    q["grading_rubric"] = None
                    if not isinstance(q.get("options"), list) or len(q["options"]) == 0:
                        q["options"] = [{"id": "A", "text": str(q["expected_answer"])}]
                else:
                    q["options"] = None
                    if not isinstance(q.get("grading_rubric"), dict):
                        q["grading_rubric"] = {"key_points": ["Review the topic material."]}

                if not q.get("explanation"):
                    q["explanation"] = None

                sanitized_questions.append(q)
            except Exception as e:
                logger.warning(f"Dropped malformed question due to parsing issue: {e}")
                continue

        # Send Success Callback
        await send_webhook_callback(
            request_id=payload.request_id,
            course_id=payload.course_id,
            operation_type=payload.operation_type,
            status="success",
            message=f"Successfully extracted {len(sanitized_questions)} native questions.",
            module_id=payload.body.module_id,
            material_id=payload.body.material_id,
            data={
                "extracted_questions": sanitized_questions,
                "destination": getattr(payload.body, "destination", None)  # <--- ADDED DESTINATION
            } 
        )

    except Exception as e:
        logger.error(f"Native extraction failed: {e}")
        await send_webhook_callback(
            request_id=payload.request_id,
            course_id=payload.course_id,
            operation_type=payload.operation_type,
            status="success", 
            message=f"Extraction failed: {str(e)[:100]}",
            module_id=payload.body.module_id,
            material_id=payload.body.material_id,
            data={
                "extracted_questions": [],
                "destination": getattr(payload.body, "destination", None) # <--- ADDED DESTINATION
            } 
        )

@extraction_router.post("/extraction/questions")
async def extract_questions_endpoint(
    request: Request,
    payload: ExtractionWebhookPayload,
    background_tasks: BackgroundTasks,
    secure_request_id: str = Depends(verify_backend_signature)
):
    project_id = str(payload.course_id)
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    background_tasks.add_task(
        _extract_questions_background,
        app=request.app, project=project, payload=payload
    )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "processing_started",
            "request_id": secure_request_id,
            "message": "Native question extraction queued successfully."
        }
    )