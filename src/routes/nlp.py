import logging
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from routes.schemas.nlp import PushRequest, SearchRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.enums.ResponseEnums import ResponseSignal
from controllers import NLPController
from models import ResponseSignal

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/courses",
    tags=["api_v1", "courses"]
)


@nlp_router.post("/{project_id}/rag/index")
async def index_project(request: Request,
                        project_id: str,
                        push_request: PushRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client,
    )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client,
    )

    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value,
                "message": f"Project with id {project_id} not found."
            }
        )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )

    # ── STEP 1: Fetch ALL rag chunks at once ──────────────────────────────
    # No pagination here — we need all chunks in one list so embed_batch
    # can process them in one GPU call instead of many small calls.
    logger.info(f"Fetching all RAG chunks for project {project_id}")

    all_chunks = await chunk_model.get_chunks_by_project_id(
        project_id=project.project_id,
        page_no=1,
        page_size=10000,        # large enough to get everything
        chunk_type="rag"
    )

    if not all_chunks or len(all_chunks) == 0:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value,
                "message": f"No RAG chunks found for project {project_id}. "
                           f"Please process the document first."
            }
        )

    logger.info(f"Found {len(all_chunks)} chunks — starting batch embedding")

    # ── STEP 2: Embed ALL chunks in one single GPU batch call ─────────────
    # This is the key fix — one call to embed_batch instead of
    # one call per page. GPU processes all chunks in batches of 64.
    chunks_ids = list(range(0, len(all_chunks)))

    is_inserted = await nlp_controller.index_into_vector_db(
        project=project,
        chunks=all_chunks,
        chunks_ids=chunks_ids,
        do_reset=push_request.do_reset
    )

    if not is_inserted:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value,
            }
        )

    logger.info(f"Successfully indexed {len(all_chunks)} chunks for project {project_id}")

    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": len(all_chunks)
        }
    )


@nlp_router.get("/{project_id}/rag/index")
async def get_project_index_info(request: Request, project_id: str):
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client,
    )

    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )

    collection_info = nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED_SUCCESS.value,
            "collection_info": collection_info
        }
    )


@nlp_router.post("/{project_id}/rag/search")
async def search_index(request: Request, project_id: str, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client,
    )

    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )

    results = await nlp_controller.search_vector_db_collection(
        project=project, text=search_request.text, limit=search_request.limit
    )

    if not results:
        return JSONResponse(
            content={
                "signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value,
            }
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "content": [r.dict() for r in results]
        }
    )


@nlp_router.post("/{project_id}/rag/answer")
async def answer_question(request: Request, project_id: str, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client,
    )

    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )

    answer, full_prompt, chat_history = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit
    )

    if not answer:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.RAG_ANSWER_ERROR.value
            }
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history
        }
    )