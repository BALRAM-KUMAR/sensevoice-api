import asyncio
import random
from app.ml.base import BaseAudioModel

class SentimentModelB(BaseAudioModel):
    async def predict(self, audio_path: str):
        await asyncio.sleep(random.uniform(0.3, 1.0))
        
        sentiments = ["positive", "neutral", "negative"]
        dominant = random.choice(sentiments)
        
        return {
            "sentiment": dominant,
            "dominant_emotion": None,
            "sentiment_confidence": round(random.uniform(0.7, 0.99), 2),
            "emotion_confidence": None,
            "processing_time_ms": random.randint(300, 1000),
            "scores": [
                {"label": s, "score": 0.85 if s == dominant else 0.07, "score_type": "sentiment"}
                for s in sentiments
            ],
            "segments": [
                {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "sentiment": dominant,
                    "confidence": round(random.uniform(0.7, 0.99), 2)
                }
            ]
        }

    def metadata(self):
        return {
            "id": "model-b",
            "name": "SentimentNet",
            "version": "1.0",
            "type": "sentiment",
            "description": "Audio sentiment analysis model",
            "supported_tasks": ["sentiment"],
            "languages": ["en"],
            "is_active": True
        }
