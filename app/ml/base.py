from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAudioModel(ABC):
    @abstractmethod
    async def predict(self, audio_path: str) -> Dict[str, Any]:
        """Run prediction on the audio file."""
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return model metadata."""
        pass
