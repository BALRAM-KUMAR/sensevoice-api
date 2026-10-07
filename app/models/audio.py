import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime
from sqlalchemy.types import Uuid as UUID
from app.core.database import Base

class AudioFile(Base):
    __tablename__ = "audio_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    mime_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    duration = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
