from app.ml.registry import registry
from app.schemas.ml_model import MLModel
from typing import List

def get_all_models() -> List[MLModel]:
    models = registry.get_all_models()
    return [MLModel(**m.metadata()) for m in models]

def get_model(model_id: str) -> MLModel:
    model = registry.get_model(model_id)
    if model:
        return MLModel(**model.metadata())
    return None
