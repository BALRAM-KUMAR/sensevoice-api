from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime

class AudioUploadResponse(BaseModel):
    id: UUID
    filename: str
    duration: Optional[float] = None
    status: str

class AudioFile(BaseModel):
    id: UUID
    original_filename: str
    storage_path: str
    mime_type: str
    file_size: int
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True
