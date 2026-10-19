from typing import Dict, Optional

def get_sharegpt_output(record: Dict) -> Optional[str]:
    if not (record["conversations"][0]["from"] == "human" and record["conversations"][1]["from"] == "gpt"):
        return None
    return record["conversations"][1]["value"]

def get_llava_image_reasoning_output(record: Dict) -> Optional[str]:
    if not (record["conversations"][0]["from"] == "human" and record["conversations"][1]["from"] == "gpt"):
        return None
    return record["conversations"][1]["value"]

def get_llava_video_description_output(record: Dict) -> Optional[str]:
    if not (record["conversations"][0]["from"] == "human" and record["conversations"][1]["from"] == "gpt"):
        return None
    return record["conversations"][1]["value"]

def get_clotho_output(record: Dict) -> Optional[str]:
    return record["caption"]