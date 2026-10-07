from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal
from uuid import UUID
from datetime import datetime
from app.schemas.audio import AudioFile
from app.schemas.ml_model import MLModel
from app.schemas.prediction import PredictionResponse

class AnalysisCreate(BaseModel):
    audio_id: UUID
    model_ids: List[str]
    hybrid_mode: bool = False
    combine_strategy: Literal["weighted", "audio_first", "text_first", "majority_vote"] = "weighted"

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
