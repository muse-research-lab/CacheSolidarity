import json
import os

from dataclasses import dataclass, field
from typing import Callable, Dict, List, LiteralString, Optional, Tuple, Union

from llmperf.constants import DATASETS_DIR
from llmperf.preprocessing.input import get_clotho_input, get_llava_image_reasoning_input, get_llava_video_description_input, get_sharegpt_input
from llmperf.preprocessing.output import get_clotho_output, get_llava_image_reasoning_output, get_llava_video_description_output, get_sharegpt_output
from llmperf.preprocessing.modality_path import get_clotho_audio_path, get_llava_image_path, get_llava_video_path
from llmperf.preprocessing.modality_size import get_audio_size, get_image_size, get_video_size

@dataclass
class Dataset:
    name: str
    path: Union[str,LiteralString]
    file: str
    alias: str
    color: Optional[str] = None

    get_input: Callable[[Optional[Dict]], Optional[str]] = field(default=None)
    get_output: Callable[[Dict], Optional[str]] = field(default=None)
    _get_modality_path: Callable[[Union[str,LiteralString],Dict], Union[str,LiteralString]] = field(default=None)
    _get_modality_size: Callable[[Union[str,LiteralString],Dict], Dict] = field(default=None)

    def get_modality_path(self, record: Dict) -> Union[str,LiteralString]:
        return self._get_modality_path(self.path, record)

    def get_modality_size(self, record: Dict) -> Dict:
        return self._get_modality_size(self.path, record)

    def __hash__(self):
        return hash((self.name, self.alias))
    
    def __eq__(self, other):
        if isinstance(other, Dataset):
            return self.name == other.name and self.alias == other.alias
        return False

    def load(self) -> List[dict]:
        with open(os.path.join(self.path, self.file)) as f:
            data = []
            for line in f:
                data.append(json.loads(line))
            return data

DATASETS = {
    Dataset(
        name="Text Conversations",
        path=os.path.join(DATASETS_DIR, "ShareGPT"),
        file="sharegpt.jsonl",
        alias="text-conv",
        color="#577FBC",
        get_input=get_sharegpt_input,
        get_output=get_sharegpt_output
    ),
    Dataset(
        name="Image Reasoning",
        path=os.path.join(DATASETS_DIR, "LLaVA-Instruct-150K"),
        file="complex_reasoning.jsonl",
        alias="img-reason",
        color="#57B593",
        get_input=get_llava_image_reasoning_input,
        get_output=get_llava_image_reasoning_output,
        _get_modality_path=get_llava_image_path,
        _get_modality_size=get_image_size
    ),
    Dataset(
        name="Video Description",
        path=os.path.join(DATASETS_DIR, "LLaVA-Video"),
        file="description.jsonl",
        alias="vid-desc",
        color="#F8DE4B",
        get_input=get_llava_video_description_input,
        get_output=get_llava_video_description_output,
        _get_modality_path=get_llava_video_path,
        _get_modality_size=get_video_size
    ),
    Dataset(
        name="Audio Captioning",
        path=os.path.join(DATASETS_DIR, "Clotho"),
        file="captioning.jsonl",
        alias="audio-cap",
        color="#E16F65",
        get_input=get_clotho_input,
        get_output=get_clotho_output,
        _get_modality_path=get_clotho_audio_path,
        _get_modality_size=get_audio_size
    )
}

def get_dataset_by_name(name: str) -> Union[None, Dataset]:
    for dataset in DATASETS:
        if getattr(dataset, "name", None) == name:
            return dataset
    return None

def get_dataset_by_alias(alias: str) -> Union[None, Dataset]:
    for dataset in DATASETS:
        if getattr(dataset, "alias", None) == alias:
            return dataset
    return None