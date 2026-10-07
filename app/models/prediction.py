import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey
from sqlalchemy.types import Uuid as UUID
from app.core.database import Base

class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey('analyses.id'), nullable=False)
    model_id = Column(String, nullable=False)
    sentiment = Column(String, nullable=True)
    dominant_emotion = Column(String, nullable=True)
    sentiment_confidence = Column(Float, nullable=True)
    emotion_confidence = Column(Float, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    status = Column(String, default="SUCCESS")
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class PredictionScore(Base):
    __tablename__ = "prediction_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey('model_predictions.id'), nullable=False)
    label = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    score_type = Column(String, nullable=False) # e.g., 'emotion', 'sentiment'

class AnalysisSegment(Base):
    __tablename__ = "analysis_segments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey('analyses.id'), nullable=False)
    model_id = Column(String, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    sentiment = Column(String, nullable=True)
    emotion = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    transcript_text = Column(String, nullable=True)

class Transcript(Base):
    __tablename__ = "transcripts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey('analyses.id'), nullable=False)
    language = Column(String, nullable=True)
    full_text = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
