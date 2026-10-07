import os
import os
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.audio import AudioUploadResponse
from app.services import audio_service
from app.models.audio import AudioFile

router = APIRouter()

@router.get("/{audio_id}/file")
async def get_audio_file(audio_id: UUID, db: AsyncSession = Depends(get_db)):
    audio_record = await db.get(AudioFile, audio_id)
    if not audio_record or not os.path.isfile(audio_record.storage_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        audio_record.storage_path,
        media_type=audio_record.mime_type or "application/octet-stream",
        filename=audio_record.original_filename,
    )

@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    audio_record = await audio_service.save_upload_file(file, db)
    return AudioUploadResponse(
        id=audio_record.id,
        filename=audio_record.original_filename,
        duration=audio_record.duration,
        status="UPLOADED"
    )

@router.get("/")
async def list_audio(db: AsyncSession = Depends(get_db)):
    # Mock implementation
    return []

@router.get("/{audio_id}")
async def get_audio(audio_id: str, db: AsyncSession = Depends(get_db)):
    # Mock implementation
    return {}

@router.delete("/{audio_id}")
async def delete_audio(audio_id: str, db: AsyncSession = Depends(get_db)):
    # Mock implementation
    return {"message": "deleted"}
