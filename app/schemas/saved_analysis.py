from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID
from datetime import datetime

class SavedAnalysisCreate(BaseModel):
    analysis_id: UUID
    name: str
    notes: Optional[str] = None
    tags: List[str] = []

class SavedAnalysisUpdate(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None

class SavedAnalysisResponse(BaseModel):
    id: UUID
    analysis_id: UUID
    name: str
    notes: Optional[str] = None
    tags: List[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
