"""Segment-level speech emotion recognition with emotion2vec Plus Large."""

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict

import soundfile as sf

from app.ml.base import BaseAudioModel


class Emotion2VecModel(BaseAudioModel):
    MODEL_ID = "iic/emotion2vec_plus_large"
    VAD_MODEL_ID = "fsmn-vad"

    def __init__(self, device: str = "cpu", cache_dir: str | None = None):
        self.device = device
        project_cache = Path(__file__).resolve().parents[3] / ".model_cache"
        self.cache_dir = Path(cache_dir or project_cache).resolve()
        self._emotion_model = None
        self._vad_model = None
        self._model_lock = asyncio.Lock()

    def _load_models(self):
        # FunASR/ModelScope otherwise defaults to a per-user home cache.
        ms_cache = self.cache_dir / "modelscope"
        ms_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MODELSCOPE_CACHE"] = str(ms_cache)
        vad_dir = ms_cache / "models" / "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch" / "snapshots" / "master"
        emotion_dir = ms_cache / "models" / "iic--emotion2vec_plus_large" / "snapshots" / "master"

        from funasr import AutoModel

        vad_model = AutoModel(
            model=str(vad_dir),
            device=self.device,
            hub="ms",
            disable_update=True,
            vad_kwargs={"max_single_segment_time": 30000},
        )
        emotion_model = AutoModel(
            model=str(emotion_dir),
            device=self.device,
            hub="ms",
            disable_update=True,
        )
        return vad_model, emotion_model

    async def _get_models(self):
        if self._emotion_model is None:
            async with self._model_lock:
                if self._emotion_model is None:
                    self._vad_model, self._emotion_model = await asyncio.to_thread(
                        self._load_models
                    )
        return self._vad_model, self._emotion_model

    async def predict(self, audio_path: str) -> Dict[str, Any]:
        vad_model, emotion_model = await self._get_models()
        started = time.perf_counter()
        results = await asyncio.to_thread(
            self._predict_segments, audio_path, vad_model, emotion_model
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000)

        segments = results
        if segments:
            labels = sorted({score["label"] for segment in segments for score in segment["scores"]})
            scores = [
                {
                    "label": label,
                    "score": sum(
                        next((item["score"] for item in segment["scores"] if item["label"] == label), 0.0)
                        for segment in segments
                    ) / len(segments),
                    "score_type": "emotion",
                }
                for label in labels
            ]
            dominant = max(scores, key=lambda item: item["score"])["label"]
            confidence = max(scores, key=lambda item: item["score"])["score"]
        else:
            scores, dominant, confidence = [], None, None

        return {
            "sentiment": None,
            "dominant_emotion": dominant,
            "sentiment_confidence": None,
            "emotion_confidence": confidence,
            "processing_time_ms": elapsed_ms,
            "scores": scores,
            "segments": [
                {
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "emotion": segment["emotion"],
                    "confidence": segment["confidence"],
                }
                for segment in segments
            ],
        }

    def _predict_segments(self, audio_path: str, vad_model: Any, emotion_model: Any) -> list[dict]:
        vad_results = vad_model.generate(input=audio_path)
        ranges = self._vad_ranges(vad_results)
        classified = []

        with sf.SoundFile(audio_path) as audio:
            sample_rate = audio.samplerate
            frame_count = len(audio)
            for start_ms, end_ms in ranges:
                start_frame = max(0, int(start_ms * sample_rate / 1000))
                end_frame = min(frame_count, int(end_ms * sample_rate / 1000))
                if end_frame <= start_frame:
                    continue
                audio.seek(start_frame)
                chunk = audio.read(
                    frames=end_frame - start_frame,
                    dtype="float32",
                    always_2d=True,
                )
                mono_chunk = chunk.mean(axis=1)
                output = emotion_model.generate(
                    input=mono_chunk,
                    fs=sample_rate,
                    granularity="utterance",
                    extract_embedding=False,
                )
                prediction = output[0] if isinstance(output, list) and output else {}
                labels = prediction.get("labels", [])
                values = prediction.get("scores", [])
                emotion_scores = []
                for label, value in zip(labels, values):
                    normalized = self._normalize_label(str(label))
                    if normalized:
                        emotion_scores.append({
                            "label": normalized,
                            "score": float(value),
                            "score_type": "emotion",
                        })
                if not emotion_scores:
                    continue
                best = max(emotion_scores, key=lambda item: item["score"])
                classified.append({
                    "start_time": start_frame / sample_rate,
                    "end_time": end_frame / sample_rate,
                    "emotion": best["label"],
                    "confidence": best["score"],
                    "scores": emotion_scores,
                })
        return classified

    @staticmethod
    def _vad_ranges(results: Any) -> list[tuple[float, float]]:
        ranges = []
        if not isinstance(results, list):
            results = [results]
        for result in results:
            if not isinstance(result, dict):
                continue
            value = result.get("value", [])
            if isinstance(value, dict):
                value = value.get("segments", [])
            for interval in value or []:
                if isinstance(interval, (list, tuple)) and len(interval) >= 2:
                    start, end = float(interval[0]), float(interval[1])
                    while end - start > 30000:
                        ranges.append((start, start + 30000))
                        start += 30000
                    if end > start:
                        ranges.append((start, end))
        return sorted(ranges)

    @staticmethod
    def _normalize_label(label: str) -> str:
        label = label.strip().replace("<|", "").replace("|>", "")
        if not label or label.lower() in {"<unk>", "unk", "unuse"} or label.lower().startswith("unuse"):
            return ""
        return label.lower()

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": "emotion2vec-plus-large",
            "name": "Emotion2Vec Plus Large",
            "version": self.MODEL_ID,
            "type": "emotion",
            "description": "Segment-level speech emotion recognition using FunASR emotion2vec Plus Large",
            "supported_tasks": ["emotion"],
            "languages": ["multilingual"],
            "is_active": True,
        }
