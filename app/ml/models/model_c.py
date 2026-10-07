import asyncio
import random
from app.ml.base import BaseAudioModel

class CombinedModelC(BaseAudioModel):
    async def predict(self, audio_path: str):
        await asyncio.sleep(random.uniform(1.0, 2.0))
        
        emotions = ["happy", "sad", "angry", "calm", "fearful"]
        sentiments = ["positive", "neutral", "negative"]
        
        dom_emo = random.choice(emotions)
        dom_sent = random.choice(sentiments)
        
        scores = []
        for e in emotions:
            scores.append({"label": e, "score": 0.8 if e == dom_emo else 0.05, "score_type": "emotion"})
        for s in sentiments:
            scores.append({"label": s, "score": 0.9 if s == dom_sent else 0.05, "score_type": "sentiment"})
            
        return {
            "sentiment": dom_sent,
            "dominant_emotion": dom_emo,
            "sentiment_confidence": round(random.uniform(0.8, 0.99), 2),
            "emotion_confidence": round(random.uniform(0.7, 0.99), 2),
            "processing_time_ms": random.randint(1000, 2000),
            "scores": scores,
            "segments": [
                {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "sentiment": dom_sent,
                    "emotion": dom_emo,
                    "confidence": round(random.uniform(0.75, 0.99), 2)
                }
            ]
        }

    def metadata(self):
        return {
            "id": "model-c",
            "name": "OmniAudio",
            "version": "2.0",
            "type": "combined",
            "description": "Combined emotion and sentiment analysis model",
            "supported_tasks": ["emotion", "sentiment"],
            "languages": ["en", "es"],
            "is_active": True
        }
