"""Offline multilingual transcript sentiment inference."""
import asyncio
import math
from pathlib import Path
from threading import Lock

from fastapi import HTTPException

MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MODEL_DIR = Path(__file__).resolve().parents[2] / ".model_cache" / "text-sentiment" / "cardiffnlp-twitter-xlm-roberta-base-sentiment"
MAX_TOKENS_PER_CHUNK = 384


class TextSentimentAnalyzer:
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._load_lock = Lock()
        self._async_lock = asyncio.Lock()

    def _load(self):
        if not MODEL_DIR.is_dir():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Transcript sentiment model is not cached. Run "
                    "python -m app.scripts.download_text_sentiment_model once while online."
                ),
            )
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(MODEL_DIR), local_files_only=True
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(MODEL_DIR), local_files_only=True
            )
            self._model.eval()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Could not load the cached transcript sentiment model: {exc}",
            ) from exc

    def _ensure_loaded(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    self._load()

    def _predict(self, text: str) -> dict:
        import torch

        self._ensure_loaded()
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
                "model": MODEL_ID,
                "method": "transformer",
            }

        chunks = [token_ids[i:i + MAX_TOKENS_PER_CHUNK]
                  for i in range(0, len(token_ids), MAX_TOKENS_PER_CHUNK)]
        probability_sum = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        token_total = 0
        label_map = {str(index): label.lower() for index, label in self._model.config.id2label.items()}
        with torch.inference_mode():
            for chunk in chunks:
                inputs = self._tokenizer.prepare_for_model(
                    chunk, add_special_tokens=True, return_tensors="pt", truncation=True
                )
                probabilities = torch.softmax(self._model(**inputs).logits[0], dim=-1).tolist()
                chunk_size = len(chunk)
                token_total += chunk_size
                for index, probability in enumerate(probabilities):
                    label = label_map.get(str(index), "").replace("label_", "")
                    if label in probability_sum:
                        probability_sum[label] += float(probability) * chunk_size

        scores = {label: value / token_total for label, value in probability_sum.items()}
        total = sum(scores.values())
        if total:
            scores = {label: value / total for label, value in scores.items()}
        sentiment = max(scores, key=scores.get)
        entropy = -sum(p * math.log(p) for p in scores.values() if p > 0)
        confidence = max(scores.values())
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "uncertainty": entropy / math.log(3),
            "scores": scores,
            "model": MODEL_ID,
            "method": "transformer",
        }

    async def analyze(self, text: str) -> dict:
        if not text.strip():
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "uncertainty": 1.0,
                "scores": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
                "model": MODEL_ID,
                "method": "transformer",
            }
        async with self._async_lock:
            return await asyncio.to_thread(self._predict, text)


analyzer = TextSentimentAnalyzer()


async def analyze_text(text: str) -> dict:
    return await analyzer.analyze(text)
