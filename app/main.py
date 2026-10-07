from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.database import init_db
from app.ml.registry import registry
from app.ml.models.kimi_model import KimiAudioModel
from app.ml.models.qwen_model import QwenOmniModel
from app.ml.models.model_b import SentimentModelB
from app.ml.models.model_c import CombinedModelC
from app.ml.models.sensevoice_model import SenseVoiceModel
from app.ml.models.emotion2vec_model import Emotion2VecModel

from app.api.v1 import audio, models, analysis, saved_analysis, comparison

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Database
    await init_db()
    
    # Register Mock Models
    registry.register(KimiAudioModel())
    registry.register(QwenOmniModel())
    registry.register(CombinedModelC())
    registry.register(SenseVoiceModel())
    registry.register(Emotion2VecModel())
    
    yield

app = FastAPI(title="AudioSense API POC", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audio.router, prefix="/api/v1/audio", tags=["Audio"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(saved_analysis.router, prefix="/api/v1/saved-analyses", tags=["Saved Analysis"])
app.include_router(comparison.router, prefix="/api/v1/comparisons", tags=["Comparison"])

@app.get("/")
def root():
    return {"message": "Welcome to AudioSense API"}
