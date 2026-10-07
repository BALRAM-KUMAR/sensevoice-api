import os
import uuid
import aiofiles
from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.audio import AudioFile
from app.core.config import settings

async def save_upload_file(upload_file: UploadFile, session: AsyncSession) -> AudioFile:
    # MediaRecorder commonly produces WebM in Chromium browsers.
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".webm"}
    ext = os.path.splitext(upload_file.filename)[1].lower()
    
    if ext not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise HTTPException(status_code=400, detail=f"Unsupported file extension. Allowed: {allowed}")

    os.makedirs(settings.STORAGE_PATH, exist_ok=True)
    
    file_id = uuid.uuid4()
    storage_path = os.path.join(settings.STORAGE_PATH, f"{file_id}{ext}")
    
    file_size = 0
    async with aiofiles.open(storage_path, 'wb') as out_file:
        while content := await upload_file.read(1024 * 1024):  # 1MB chunks
            await out_file.write(content)
            file_size += len(content)
            if file_size > settings.MAX_AUDIO_SIZE_MB * 1024 * 1024:
                os.remove(storage_path)
                raise HTTPException(status_code=400, detail=f"File exceeds maximum size of {settings.MAX_AUDIO_SIZE_MB}MB")
                
    audio_record = AudioFile(
        id=file_id,
        original_filename=upload_file.filename,
        storage_path=storage_path,
        mime_type=upload_file.content_type or "audio/unknown",
        file_size=file_size,
        duration=120.0, # Mock duration
        sample_rate=44100,
        channels=2
    )
    
    session.add(audio_record)
    await session.commit()
    await session.refresh(audio_record)
    
    return audio_record
