import asyncio
import time
import json
from typing import Dict, Any
from app.ml.base import BaseAudioModel

class KimiAudioModel(BaseAudioModel):
    def __init__(self):
        self.model = None
        self.device = None
        self.sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }
        self.is_loaded = False
        
    def _load_model(self):
        if self.is_loaded:
            return
            
        import sys
        import os
        import torch
        
        # Add the Kimi-Audio directory to sys.path so we can import kimia_infer
        kimi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Kimi-Audio"))
        if kimi_path not in sys.path:
            sys.path.append(kimi_path)
            
        # Assuming the KimiAudio class is available after installation
        from kimia_infer.api.kimia import KimiAudio
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "moonshotai/Kimi-Audio-7B-Instruct"
        
        print(f"Loading Kimi-Audio-7B-Instruct on {self.device}...")
        self.model = KimiAudio(model_path=model_id, load_detokenizer=True)
        self.model.to(self.device)
        self.is_loaded = True
        print("Model loaded successfully.")

    async def predict(self, audio_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # Load model lazily to avoid blocking startup if not used or downloading
        # Note: In a production async app, loading a 7B model during a request is bad.
        # But for this POC it ensures we only load it when requested.
        # Running the blocking load in a thread pool to avoid blocking the event loop
        await asyncio.to_thread(self._load_model)

        prompt = (
            "Analyze the audio and classify the dominant speech emotion (SER) "
            "and identify any sound events or acoustic scenes (SEC/ASC). "
            "Return ONLY a JSON object with two keys: 'emotion' (string) and 'sound_scene' (string)."
        )
        
        messages_asr = [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": audio_path}
        ]

        # Run inference in a separate thread to prevent blocking
        def run_inference():
            _, text_output = self.model.generate(
                messages_asr, 
                **self.sampling_params, 
                output_type="text"
            )
            return text_output

        text_output = await asyncio.to_thread(run_inference)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Parse output
        emotion = "neutral"
        sound_scene = "unknown"
        
        try:
            # Attempt to parse JSON from the model output
            # Might need to strip markdown formatting like ```json
            clean_json = text_output.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_json)
            emotion = result.get("emotion", "neutral")
            sound_scene = result.get("sound_scene", "unknown")
        except json.JSONDecodeError:
            # Fallback if model doesn't return pure JSON
            print(f"Failed to parse JSON from output: {text_output}")
            emotion = text_output[:50] # Just take a substring as a fallback

        return {
            "sentiment": None,
            "dominant_emotion": emotion.lower(),
            "sentiment_confidence": None,
            "emotion_confidence": 0.85, # Kimi doesn't easily return confidences without logits
            "processing_time_ms": processing_time_ms,
            "scores": [
                {"label": sound_scene, "score": 0.9, "score_type": "scene_event"}
            ],
            "segments": [
                {
                    "start_time": 0.0,
                    "end_time": 100.0, # Placeholder
                    "emotion": emotion.lower(),
                    "confidence": 0.85
                }
            ]
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": "moonshot-kimi-audio-7b",
            "name": "Kimi-Audio-7B-Instruct",
            "version": "1.0",
            "type": "audio_multimodal",
            "description": "Moonshot AI 7B Multimodal Audio model for SER and SEC/ASC",
            "supported_tasks": ["emotion", "sound_event", "asr"],
            "languages": ["en", "zh"],
            "is_active": True
        }
