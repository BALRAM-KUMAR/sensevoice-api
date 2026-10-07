from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from uuid import UUID

class ComparisonRequest(BaseModel):
    analysis_id: UUID
    model_ids: List[str]

class ComparisonResponse(BaseModel):
    analysis_id: UUID
    models: List[str]
    sentiment_comparison: Dict[str, Any]
    emotion_comparison: Dict[str, Any]
    confidence_comparison: Dict[str, Any]
    processing_time_comparison: Dict[str, Any]
    agreement: Dict[str, Any]
    segment_differences: List[Dict[str, Any]]
