from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID

class PredictionScoreResponse(BaseModel):
    label: str
    score: float
    score_type: str

    class Config:
        from_attributes = True

class SegmentResponse(BaseModel):
    start_time: float
    end_time: float
    sentiment: Optional[str] = None
    emotion: Optional[str] = None
    confidence: Optional[float] = None
    transcript_text: Optional[str] = None

    class Config:
        from_attributes = True

class PredictionResponse(BaseModel):
    model_id: str
    sentiment: Optional[str] = None
    emotion: Optional[str] = None
    confidence: Optional[float] = None
    scores: List[PredictionScoreResponse] = []
    status: Optional[str] = None
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True

class DetailedPredictionResponse(PredictionResponse):
    processing_time_ms: Optional[int] = None
    status: str
    error_message: Optional[str] = None
    segments: List[SegmentResponse] = []
