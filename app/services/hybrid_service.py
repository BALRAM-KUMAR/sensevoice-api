"""Speech transcription, transcript sentiment, and audio/text fusion."""
import asyncio
import math
from pathlib import Path

from fastapi import HTTPException

from app.core.config import settings
from app.services.text_sentiment_service import analyze_text


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
            "language_probability": getattr(result, "language_probability", None),
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


def _sentiment_from_emotion(emotion: str | None) -> str | None:
    if not emotion:
        return None
    value = emotion.lower()
    if value in {"happy", "excited", "positive"}:
        return "positive"
    if value in {"sad", "angry", "fearful", "disgusted", "frustrated", "negative"}:
        return "negative"
    if value in {"neutral", "calm", "surprised"}:
        return "neutral"
    return None


LABELS = ("positive", "neutral", "negative")
EMOTION_TO_SENTIMENT = {
    "happy": "positive", "happiness": "positive", "joy": "positive",
    "excited": "positive", "love": "positive",
    "sad": "negative", "sadness": "negative", "angry": "negative",
    "anger": "negative", "fearful": "negative", "fear": "negative",
    "disgusted": "negative", "disgust": "negative", "frustrated": "negative",
    "neutral": "neutral", "calm": "neutral", "surprised": "neutral",
    "surprise": "neutral", "other": "neutral",
}


def _normalized_distribution(values: dict[str, float]) -> dict[str, float] | None:
    total = sum(max(0.0, float(values.get(label, 0.0))) for label in LABELS)
    if total <= 0:
        return None
    return {label: max(0.0, float(values.get(label, 0.0))) / total for label in LABELS}


def _audio_model_distribution(prediction: dict) -> dict[str, float] | None:
    values = {label: 0.0 for label in LABELS}
    for score in prediction.get("scores", []):
        label = str(score.get("label", "")).lower().replace(" ", "_")
        score_type = str(score.get("score_type", "")).lower()
        target = label if label in LABELS else EMOTION_TO_SENTIMENT.get(label) if score_type == "emotion" else None
        if target:
            values[target] += max(0.0, float(score.get("score", 0.0)))
    distribution = _normalized_distribution(values)
    if distribution:
        return distribution
    sentiment = str(prediction.get("sentiment") or "").lower()
    sentiment = sentiment if sentiment in LABELS else EMOTION_TO_SENTIMENT.get(str(prediction.get("emotion") or "").lower())
    return ({label: float(label == sentiment) for label in LABELS} if sentiment else None)


def _uncertainty(distribution: dict[str, float]) -> float:
    entropy = -sum(p * math.log(p) for p in distribution.values() if p > 0)
    return min(1.0, entropy / math.log(len(LABELS)))


def audio_distribution(predictions: list[dict]) -> dict[str, float] | None:
    distributions = [
        distribution for prediction in predictions
        if (distribution := _audio_model_distribution(prediction)) is not None
    ]
    if not distributions:
        return None
    return _normalized_distribution({
        label: sum(distribution[label] for distribution in distributions) / len(distributions)
        for label in LABELS
    })


def combine(audio_scores: dict[str, float] | None, text_result: dict,
            language_probability: float | None = None) -> dict:
    """Confidence-aware late fusion; keep disagreement visible for review."""
    text_scores = _normalized_distribution(text_result.get("scores", {}))
    if not text_scores:
        raise HTTPException(status_code=502, detail="Text sentiment model returned invalid scores.")
    audio_scores = _normalized_distribution(audio_scores or {})

    text_reliability = 1.0 - _uncertainty(text_scores)
    if language_probability is not None:
        text_reliability *= max(0.0, min(1.0, language_probability))
    audio_reliability = 1.0 - _uncertainty(audio_scores) if audio_scores else 0.0
    if audio_scores and audio_reliability + text_reliability <= 1e-8:
        audio_reliability = text_reliability = 1.0
    total_weight = audio_reliability + text_reliability
    if audio_scores:
        fused = {
            label: (audio_scores[label] * audio_reliability + text_scores[label] * text_reliability) / total_weight
            for label in LABELS
        }
    else:
        fused = text_scores

    audio_label = max(audio_scores, key=audio_scores.get) if audio_scores else None
    text_label = max(text_scores, key=text_scores.get)
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    disagreement = audio_label is not None and audio_label != text_label
    confidence = ordered[0][1]
    margin = confidence - (ordered[1][1] if len(ordered) > 1 else 0.0)
    return {
        "sentiment": ordered[0][0],
        "audio_sentiment": audio_label,
        "text_sentiment": text_label,
        "strategy": "confidence_weighted_late_fusion",
        "scores": fused,
        "confidence": confidence,
        "audio_reliability": audio_reliability / total_weight if total_weight else 0.0,
        "text_reliability": text_reliability / total_weight if total_weight else 1.0,
        "disagreement": disagreement,
        "needs_review": disagreement or confidence < 0.55 or margin < 0.15,
        "confidence_calibrated": False,
    }
