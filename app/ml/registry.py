from typing import Dict
from app.ml.base import BaseAudioModel

class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, BaseAudioModel] = {}

    def register(self, model: BaseAudioModel):
        meta = model.metadata()
        self._models[meta["id"]] = model

    def get_model(self, model_id: str) -> BaseAudioModel:
        return self._models.get(model_id)

    def get_all_models(self) -> list[BaseAudioModel]:
        return list(self._models.values())

registry = ModelRegistry()
