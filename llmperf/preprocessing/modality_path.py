import os

from typing import Dict, LiteralString, Union

def get_llava_image_path(dir: Union[str,LiteralString], record: Dict) -> LiteralString:
    return os.path.join(dir, "images", record["image"])

def get_llava_video_path(dir: Union[str,LiteralString], record: Dict) -> LiteralString:
    return os.path.join(dir, "videos", record["video"])

def get_clotho_audio_path(dir: Union[str,LiteralString], record: Dict) -> LiteralString:
    return os.path.join(dir, "audios", record["audio_id"])