from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.comparison import ComparisonRequest, ComparisonResponse
from app.services import comparison_service

router = APIRouter()

@router.post("/", response_model=ComparisonResponse)
async def compare_models(request: ComparisonRequest, db: AsyncSession = Depends(get_db)):
    return await comparison_service.compare_models(request, db)

@router.get("/")
async def list_comparisons():
    return []

@router.get("/{id}")
async def get_comparison(id: str):
    return {}
