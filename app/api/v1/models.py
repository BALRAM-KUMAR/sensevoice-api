from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.ml_model import MLModel
from app.services import model_service

router = APIRouter()

@router.get("/", response_model=List[MLModel])
async def list_models():
    return model_service.get_all_models()

@router.get("/{model_id}", response_model=MLModel)
async def get_model(model_id: str):
    model = model_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model
