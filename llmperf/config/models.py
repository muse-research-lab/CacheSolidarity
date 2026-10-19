import os

from dataclasses import dataclass
from typing import LiteralString, Optional, Union

from llmperf.constants import MODELS_DIR

@dataclass
class Model:
    name: str
    path: Union[str,LiteralString]
    max_model_len: int
    alias: str
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    audio_token_index: Optional[int] = None

    def __hash__(self):
        return hash((self.name, self.alias))

    def __eq__(self, other):
        if isinstance(other, Model):
            return self.name == other.name and self.alias == other.alias
        return False

MODELS = {
    Model(
        name="Mistral-7b",
        path=os.path.join(MODELS_DIR, "Mistral-7B-Instruct-v0.2"),
        max_model_len=32768,
        alias="text-mistral"
    ),
    Model(
        name="LLaVA 1.6 (Mistral-7b)",
        path="llava-hf/llava-v1.6-mistral-7b-hf",
        max_model_len=32768,
        alias="image-mistral",
        image_token_index=32000
    ),
    Model(
        name="LLaVA-Next-Video (Mistral-7b)",
        path="llava-hf/LLaVA-NeXT-Video-7B-32K-hf",
        max_model_len=32768,
        alias="video-mistral",
        image_token_index=32001,
        video_token_index=32000
    ),
    Model(
        name="Qwen2-Audio-7b",
        path="Qwen/Qwen2-Audio-7B-Instruct",
        max_model_len=8192,
        alias="audio-qwen",
        audio_token_index=151646
    ),
    Model(
        name="LLaVA-Onevision-7b",
        path="llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
        max_model_len=32768,
        alias="llava-ov",
        image_token_index=151646,
        video_token_index=151647
    ),
    Model(
        name="LLaVA-Onevision-500M",
        path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        max_model_len=32768,
        alias="llava-ov-small",
        image_token_index=151646,
        video_token_index=151647
    ),
    Model(
        name="Qwen2.5-7B",
        path="Qwen/Qwen2.5-VL-7B-Instruct",
        max_model_len=128000,
        alias="qwen-2.5",
        image_token_index=151655,
        video_token_index=151656
    ),
    Model(
        name="Qwen2.5-3B",
        path="Qwen/Qwen2.5-VL-3B-Instruct",
        max_model_len=128000,
        alias="qwen-2.5-small",
        image_token_index=151655,
        video_token_index=151656
    ),
    Model(
        name="Qwen2-7B",
        path="Qwen/Qwen2-VL-7B-Instruct",
        max_model_len=128000,
        alias="qwen-2",
        image_token_index=151655,
        video_token_index=151656
    ),
    Model(
        name="Qwen2-2B",
        path="Qwen/Qwen2-VL-2B-Instruct",
        max_model_len=32768,
        alias="qwen-2-small",
        image_token_index=151655,
        video_token_index=151656
    ),
}

def get_model_by_name(name: str) -> Union[None, Model]:
    for model in MODELS:
        if getattr(model, "name", None) == name:
            return model
    return None

def get_model_by_alias(alias: str) -> Union[None, Model]:
    for model in MODELS:
        if getattr(model, "alias", None) == alias:
            return model
    return None