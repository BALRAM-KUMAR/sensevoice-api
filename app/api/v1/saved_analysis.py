from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List
from app.core.database import get_db
from app.schemas.saved_analysis import SavedAnalysisCreate, SavedAnalysisResponse, SavedAnalysisUpdate
from app.models.saved_analysis import SavedAnalysis

router = APIRouter()

@router.post("/", response_model=SavedAnalysisResponse)
async def create_saved_analysis(request: SavedAnalysisCreate, db: AsyncSession = Depends(get_db)):
    saved = SavedAnalysis(**request.model_dump())
    db.add(saved)
    await db.commit()
    await db.refresh(saved)
    return saved

@router.get("/", response_model=List[SavedAnalysisResponse])
async def list_saved_analyses(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(SavedAnalysis))
    return result.scalars().all()

@router.get("/{id}", response_model=SavedAnalysisResponse)
async def get_saved_analysis(id: str, db: AsyncSession = Depends(get_db)):
    saved = await db.get(SavedAnalysis, id)
    if not saved:
        raise HTTPException(status_code=404, detail="Saved analysis not found")
    return saved

@router.patch("/{id}", response_model=SavedAnalysisResponse)
async def update_saved_analysis(id: str, request: SavedAnalysisUpdate, db: AsyncSession = Depends(get_db)):
    saved = await db.get(SavedAnalysis, id)
    if not saved:
        raise HTTPException(status_code=404, detail="Saved analysis not found")
    
    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(saved, key, value)
        
    await db.commit()
    await db.refresh(saved)
    return saved

@router.delete("/{id}")
async def delete_saved_analysis(id: str, db: AsyncSession = Depends(get_db)):
    saved = await db.get(SavedAnalysis, id)
    if not saved:
        raise HTTPException(status_code=404, detail="Saved analysis not found")
    await db.delete(saved)
    await db.commit()
    return {"message": "Deleted successfully"}
