from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from app.schemas.audio import AudioFile
from app.schemas.ml_model import MLModel
from app.schemas.prediction import PredictionResponse

class AnalysisCreate(BaseModel):
    audio_id: UUID
    model_ids: List[str]
    hybrid_mode: bool = False

class AnalysisResponse(BaseModel):
    id: UUID
    status: str
    audio: dict # simplified for response
    models: List[dict]
    predictions: List[PredictionResponse]
    timeline: List[dict]
    transcript: dict
    hybrid: Optional[dict] = None

    class Config:
        from_attributes = True
