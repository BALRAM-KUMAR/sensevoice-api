"""SenseVoiceSmall adapter for speech emotion recognition via FunASR."""

import asyncio
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import soundfile as sf

from app.ml.base import BaseAudioModel


class SenseVoiceModel(BaseAudioModel):
    """Run FunAudioLLM/SenseVoiceSmall and extract its inline emotion tags."""

    MODEL_ID = "FunAudioLLM/SenseVoiceSmall"
    EMOTION_TAGS = {
        "HAPPY": "happy",
        "SAD": "sad",
        "ANGRY": "angry",
        "NEUTRAL": "neutral",
        "FEARFUL": "fearful",
        "DISGUSTED": "disgusted",
        "SURPRISED": "surprised",
    }
    TAG_PATTERN = re.compile(r"<\|([A-Z_]+)\|>")

    def __init__(self, device: str = "cpu", cache_dir: str | None = None):
        self.device = device
        project_cache = Path(__file__).resolve().parents[3] / ".model_cache"
        self.cache_dir = Path(cache_dir or project_cache).resolve()
        self._model = None
        self._model_lock = asyncio.Lock()

    async def _get_model(self):
        if self._model is None:
            async with self._model_lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._load_model)
        return self._model

    def _load_model(self):
        # Keep model imports and checkpoint download out of API startup, and
        # direct both hub clients to a cache inside the project working tree.
        hf_cache = self.cache_dir / "huggingface"
        ms_cache = self.cache_dir / "modelscope"
        hf_cache.mkdir(parents=True, exist_ok=True)
        ms_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache / "hub")
        # Avoid Xet CAS reconstruction failures on some networks; the hub
        # client falls back to its standard HTTP file download path.
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        # These weights are already stored in the project cache. Do not query
        # the hub or fetch them again when the API process starts.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["MODELSCOPE_CACHE"] = str(ms_cache)

        from huggingface_hub import snapshot_download
        from funasr import AutoModel

        model_dir = snapshot_download(
            repo_id=self.MODEL_ID,
            cache_dir=str(hf_cache),
            local_files_only=True,
        )
        vad_dir = ms_cache / "models" / "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch" / "snapshots" / "master"
        vad_model = AutoModel(
            model=str(vad_dir),
            device=self.device,
            hub="ms",
            disable_update=True,
            vad_kwargs={"max_single_segment_time": 30000},
        )
        sensevoice_model = AutoModel(
            model=model_dir,
            device=self.device,
            hub="hf",
            disable_update=True,
        )
        return vad_model, sensevoice_model

    async def predict(self, audio_path: str) -> Dict[str, Any]:
        vad_model, model = await self._get_model()
        started = time.perf_counter()
        segments = await asyncio.to_thread(
            self._predict_segments, audio_path, vad_model, model
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000)

        counts = Counter(segment["emotion"] for segment in segments if segment["emotion"])
        dominant_emotion = counts.most_common(1)[0][0] if counts else None
        total = sum(counts.values())

        # These are observed tag frequencies across VAD speech chunks, not
        # model probabilities.
        scores = [
            {"label": label, "score": count / total, "score_type": "emotion"}
            for label, count in sorted(counts.items())
        ]
        return {
            "sentiment": None,
            "dominant_emotion": dominant_emotion,
            "sentiment_confidence": None,
            "emotion_confidence": None,
            "processing_time_ms": elapsed_ms,
            "scores": scores,
            "segments": segments,
        }

    def _predict_segments(self, audio_path: str, vad_model: Any, model: Any) -> list[dict[str, Any]]:
        vad_results = vad_model.generate(input=audio_path)
        ranges = []
        if not isinstance(vad_results, list):
            vad_results = [vad_results]
        for result in vad_results:
            if not isinstance(result, dict):
                continue
            for interval in result.get("value", []) or []:
                if isinstance(interval, (list, tuple)) and len(interval) >= 2:
                    start, end = float(interval[0]), float(interval[1])
                    while end - start > 30000:
                        ranges.append((start, start + 30000))
                        start += 30000
                    if end > start:
                        ranges.append((start, end))

        segments = []
        with sf.SoundFile(audio_path) as audio:
            sample_rate = audio.samplerate
            frame_count = len(audio)
            for start_ms, end_ms in sorted(ranges):
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
                output = model.generate(
                    input=chunk.mean(axis=1),
                    fs=sample_rate,
                    cache={},
                    language="auto",
                    use_itn=True,
                )
                text = output[0].get("text", "") if output and isinstance(output[0], dict) else ""
                emotion = next(
                    (self.EMOTION_TAGS[tag] for tag in self.TAG_PATTERN.findall(text)
                     if tag in self.EMOTION_TAGS),
                    None,
                )
                segments.append({
                    "start_time": start_frame / sample_rate,
                    "end_time": end_frame / sample_rate,
                    "emotion": emotion,
                    "confidence": None,
                })
        return segments

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": "sensevoice-small",
            "name": "SenseVoiceSmall",
            "version": "FunAudioLLM/SenseVoiceSmall",
            "type": "emotion",
            "description": "Speech emotion recognition using SenseVoiceSmall via FunASR",
            "supported_tasks": ["emotion"],
            "languages": ["zh", "en", "ja", "ko", "yue"],
            "is_active": True,
        }
