from typing import Dict, Optional

def get_sharegpt_input(record: Dict) -> Optional[str]:
    if not (record["conversations"][0]["from"] == "human" and record["conversations"][1]["from"] == "gpt"):
        return None
    return record["conversations"][0]["value"]

def get_llava_image_reasoning_input(record: Dict) -> Optional[str]:
    if not (record["conversations"][0]["from"] == "human" and record["conversations"][1]["from"] == "gpt"):
        return None
    return record["conversations"][0]["value"].replace("<image>\n", "").replace("\n<image>", "")

def get_llava_video_description_input(record: Dict) -> Optional[str]:
    if not (record["conversations"][0]["from"] == "human" and record["conversations"][1]["from"] == "gpt"):
        return None
    return record["conversations"][0]["value"].replace("<image>\n", "").replace("\n<image>", "")

def get_clotho_input() -> Optional[str]:
    return "Please describe this audio."