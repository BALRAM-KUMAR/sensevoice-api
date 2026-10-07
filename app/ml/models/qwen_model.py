import os
import asyncio
import time
import json
from typing import Dict, Any
from app.ml.base import BaseAudioModel

class QwenOmniModel(BaseAudioModel):
    def __init__(self):
        self.llm = None
        self.processor = None
        self.is_loaded = False
        
    def _load_model(self):
        if self.is_loaded:
            return
            
        import torch
        import sys
        
        # Monkeypatch uvloop for Windows since vLLM requires it but it's Linux-only
        if sys.platform == 'win32':
            import asyncio
            sys.modules['uvloop'] = asyncio
            
        from vllm import LLM, SamplingParams
        from transformers import Qwen3OmniMoeProcessor
        
        print("Loading Qwen3-Omni-30B-A3B-Captioner...")
        os.environ['VLLM_USE_V1'] = '0'
        MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Captioner"
        
        self.llm = LLM(
            model=MODEL_PATH, 
            trust_remote_code=True, 
            gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count() or 1,
            limit_mm_per_prompt={'audio': 1},
            max_num_seqs=8,
            max_model_len=32768,
            seed=1234,
        )
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=1024, # Reduced for faster classification
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
        self.is_loaded = True
        print("Qwen Model loaded successfully.")

    async def predict(self, audio_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        await asyncio.to_thread(self._load_model)

        # Run inference in a separate thread
        def run_inference():
            try:
                from app.ml.models.qwen_omni_utils import process_mm_info
            except ImportError:
                raise ImportError("qwen_omni_utils not found. Please ensure it is downloaded and in your path.")

            # Modify prompt for emotion and sentiment
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze the audio and classify the dominant speech emotion (SER) and identify any sound events or acoustic scenes (SEC/ASC). Return ONLY a JSON object with two keys: 'emotion' (string) and 'sound_scene' (string)."},
                        {"type": "audio", "audio": audio_path}
                    ], 
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            audios, _, _ = process_mm_info(messages, use_audio_in_video=False)

            inputs = {
                'prompt': text,
                'multi_modal_data': {},
            }

            if audios is not None:
                inputs['multi_modal_data']['audio'] = audios

            outputs = self.llm.generate([inputs], sampling_params=self.sampling_params)
            return outputs[0].outputs[0].text

        text_output = await asyncio.to_thread(run_inference)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        emotion = "neutral"
        sound_scene = "unknown"
        
        try:
            clean_json = text_output.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_json)
            emotion = result.get("emotion", "neutral")
            sound_scene = result.get("sound_scene", "unknown")
        except Exception:
            emotion = text_output[:50] 

        return {
            "sentiment": None,
            "dominant_emotion": emotion.lower(),
            "sentiment_confidence": None,
            "emotion_confidence": 0.85,
            "processing_time_ms": processing_time_ms,
            "scores": [
                {"label": sound_scene, "score": 0.9, "score_type": "scene_event"}
            ],
            "segments": [
                {
                    "start_time": 0.0,
                    "end_time": 100.0,
                    "emotion": emotion.lower(),
                    "confidence": 0.85
                }
            ]
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": "qwen-omni-30b",
            "name": "Qwen3-Omni-30B",
            "version": "1.0",
            "type": "audio_multimodal",
            "description": "Qwen3 Omni 30B MoE for audio captioning and analysis",
            "supported_tasks": ["emotion", "sound_event"],
            "languages": ["en", "zh"],
            "is_active": True
        }
