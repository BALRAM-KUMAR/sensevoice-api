import os

def process_mm_info(messages, use_audio_in_video=False):
    # Dummy implementation for qwen_omni_utils based on vllm usage
    # Extracts audio data from messages and returns it for vllm inputs
    audios = []
    for msg in messages:
        if "content" in msg and isinstance(msg["content"], list):
            for item in msg["content"]:
                if item.get("type") == "audio" and "audio" in item:
                    audios.append(item["audio"])
    
    if not audios:
        return None, None, None
        
    # vLLM expects a list or tuple of audio paths/tensors depending on version
    return audios, None, None
