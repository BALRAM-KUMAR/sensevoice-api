from fastapi import APIRouter, Depends
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.analysis import AnalysisCreate, AnalysisResponse
from app.schemas.prediction import DetailedPredictionResponse
from app.services import analysis_service

router = APIRouter()

@router.get("/")
async def list_analyses(db: AsyncSession = Depends(get_db)):
    return await analysis_service.list_analyses(db)

@router.post("/", response_model=AnalysisResponse)
async def create_analysis(request: AnalysisCreate, db: AsyncSession = Depends(get_db)):
    return await analysis_service.run_analysis(request, db)

@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: UUID, db: AsyncSession = Depends(get_db)):
    return await analysis_service.get_analysis(analysis_id, db)

@router.get("/{analysis_id}/models/{model_id}", response_model=DetailedPredictionResponse)
async def get_model_result(analysis_id: UUID, model_id: str, db: AsyncSession = Depends(get_db)):
    return await analysis_service.get_model_result(analysis_id, model_id, db)
