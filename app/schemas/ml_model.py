from pydantic import BaseModel
from typing import List

class MLModel(BaseModel):
    id: str
    name: str
    version: str
    type: str
    description: str
    supported_tasks: List[str]
    languages: List[str]
    is_active: bool
