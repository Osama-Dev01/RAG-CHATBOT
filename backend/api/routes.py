import logging
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from backend.config.settings import settings
from backend.models.schemas import (
    UploadResponse,
    FilesListResponse,
    FileInfo,
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    ChatMessage,
    HealthResponse,
)
from backend.services.document_processor import document_processor
from backend.services.vector_store import vector_store_service
from backend.services.rag_service import rag_service

logger = logging.getLogger(__name__)

router = APIRouter()




# ── File Upload ────

@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOCX, TXT).
    Extracts text, chunks it, embeds and stores in ChromaDB.
    """
    print(f"Received file: {file.filename}")
    print(f"Content type: {file.content_type}")
   
    
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {settings.allowed_extensions_list}",
        )

    target_file_path = Path(settings.upload_dir) / file.filename

    if target_file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The file '{file.filename}' has already been uploaded and processed in the system.",
        )


   
    save_path = Path(settings.upload_dir) / file.filename
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved: {save_path}")
    except Exception as e:
        logger.error(f"File save error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

  
    try:
        chunks = document_processor.process(str(save_path))
        num_chunks = vector_store_service.add_documents(chunks)
    except Exception as e:
        logger.error(f"Processing error for {file.filename}: {e}")
        # Clean up saved file on failure
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    return UploadResponse(
        success=True,
        filename=file.filename,
        message=f"File processed and indexed successfully.",
        chunks_created=num_chunks,
    )






# ── Chat ────



@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required.")

    try:
        answer, raw_sources = rag_service.chat(
            question=request.question,
            session_id=request.session_id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
  
    cleaned_sources = []
    if raw_sources:
    
        top_source_obj = raw_sources[0]
        raw_path = top_source_obj.get("source", "Unknown Source")
        
       
        file_name = os.path.basename(raw_path)
        
       
        clean_name, _ = os.path.splitext(file_name)
        
        cleaned_sources.append(clean_name)

    return ChatResponse(
        answer=answer,
        sources=cleaned_sources, 
        session_id=request.session_id
    )

# ── History ─────

@router.get("/history/{session_id}", response_model=HistoryResponse, tags=["Chat"])
async def get_history(session_id: str):
    """Return full conversation history for a session."""
    messages_raw = rag_service.get_history(session_id)
    messages = [
        ChatMessage(
            role=m["role"],
            content=m["content"],
            timestamp=m["timestamp"],
        )
        for m in messages_raw
    ]
    return HistoryResponse(session_id=session_id, messages=messages)


