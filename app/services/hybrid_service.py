"""Speech transcription, transcript sentiment, and audio/text fusion."""
import asyncio
from collections import Counter
from pathlib import Path

from fastapi import HTTPException

from app.core.config import settings


def _transcribe_sync(audio_path: str) -> dict:
    if not settings.ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Hybrid mode needs ELEVENLABS_API_KEY configured on the API server.",
        )
    try:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        with Path(audio_path).open("rb") as audio_file:
            result = client.speech_to_text.convert(
                file=audio_file,
                model_id=settings.ELEVENLABS_STT_MODEL_ID,
                tag_audio_events=True,
                diarize=True,
            )
        return {
            "text": getattr(result, "text", "") or "",
            "language_code": getattr(result, "language_code", None),
            "words": [
                {"text": getattr(word, "text", ""), "start": getattr(word, "start", None),
                 "end": getattr(word, "end", None), "speaker_id": getattr(word, "speaker_id", None)}
                for word in (getattr(result, "words", None) or [])
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ElevenLabs transcription failed: {exc}") from exc


async def transcribe(audio_path: str) -> dict:
    return await asyncio.to_thread(_transcribe_sync, audio_path)


def analyze_text(text: str) -> dict:
    """Small deterministic lexical baseline; returns transparent evidence scores."""
    positive = {"good", "great", "excellent", "love", "like", "happy", "wonderful", "thanks", "thankful", "hope", "pleased", "enjoy", "best", "awesome", "calm", "confident"}
    negative = {"bad", "terrible", "awful", "hate", "angry", "sad", "upset", "worried", "fear", "afraid", "disappointed", "worst", "pain", "problem", "frustrated", "sorry"}
    tokens = [token.strip(".,!?;:()[]{}\"'“”‘’").lower() for token in text.split()]
    pos = sum(token in positive for token in tokens)
    neg = sum(token in negative for token in tokens)
    total = pos + neg
    sentiment = "neutral" if not total or pos == neg else "positive" if pos > neg else "negative"
    confidence = 0.5 if not total else max(pos, neg) / total
    return {"sentiment": sentiment, "confidence": confidence,
            "scores": {"positive": pos / total if total else 0.0,
                       "neutral": 1.0 if not total else 0.0,
                       "negative": neg / total if total else 0.0},
            "method": "lexicon"}


def _sentiment_from_emotion(emotion: str | None) -> str | None:
    if not emotion:
        return None
    value = emotion.lower()
    if value in {"happy", "surprised", "excited", "calm", "positive"}:
        return "positive"
    if value in {"sad", "angry", "fearful", "disgusted", "frustrated", "negative"}:
        return "negative"
    if value in {"neutral", "calm"}:
        return "neutral"
    return None


def combine(audio_sentiments: list[str], text_sentiment: str, strategy: str) -> dict:
    allowed = {"weighted", "audio_first", "text_first", "majority_vote"}
    if strategy not in allowed:
        raise HTTPException(status_code=422, detail=f"combine_strategy must be one of: {', '.join(sorted(allowed))}")
    counts = Counter(s for s in audio_sentiments if s in {"positive", "neutral", "negative"})
    audio_sentiment = counts.most_common(1)[0][0] if counts else None
    if strategy == "audio_first":
        final = audio_sentiment or text_sentiment
    elif strategy == "text_first":
        final = text_sentiment or audio_sentiment
    elif strategy == "majority_vote":
        votes = Counter(([audio_sentiment] if audio_sentiment else []) + [text_sentiment])
        final = votes.most_common(1)[0][0] if votes and (len(votes) == 1 or votes.most_common()[0][1] > votes.most_common()[-1][1]) else "neutral"
    elif not audio_sentiment:
        final = text_sentiment
    elif audio_sentiment == text_sentiment:
        final = audio_sentiment
    else:
        audio_votes = counts[audio_sentiment]
        text_votes = 1
        final = audio_sentiment if audio_votes > text_votes else text_sentiment if text_votes > audio_votes else "neutral"
    return {"sentiment": final, "audio_sentiment": audio_sentiment,
            "text_sentiment": text_sentiment, "strategy": strategy}
