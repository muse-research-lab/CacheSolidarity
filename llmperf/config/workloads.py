import hashlib
import json
import os

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, LiteralString, Optional, Union

from llmperf.constants import WORKLOADS_DIR

@dataclass
class Request:
    input: str
    output: str
    id: str = None
    modality_path: Optional[Union[str,LiteralString]] = None
    modality_size: Optional[Dict] = field(default_factory=dict)

    def __post_init__(self): 
        if not self.id:
            combined = f"{self.input}|{self.output}|{self.modality_path or ''}"
            # ! Not collision free, but sufficient for the number of our requests
            hash_object = hashlib.sha256(combined.encode('utf-8'))
            self.id = hash_object.hexdigest()[:8]

@dataclass
class Workload:
    name: str
    path: Union[str,LiteralString]
    alias: str
    modalities: Union[str, List[str]]
    modality_pct: Union[float, List[float]]
    modality_dist: Optional[Literal["uniform", "categorical"]] = None
    arrival_dist: Optional[Literal["poisson", "bursgpt", "gamma"]] = None

    requests: Optional[List[Request]] = field(default_factory=list)
    timestamps: Optional[List[float]] = field(default_factory=list)

    def __hash__(self):
        return hash((self.name, self.alias))

    def __eq__(self, other):
        if isinstance(other, Workload):
            return self.name == other.name and self.alias == other.alias
        return False
    
    def save(self):
        path = os.path.join(self.path, f"{self.alias}.jsonl")
        with open(path, "w", encoding="utf-8") as file:
            for i, request in enumerate(self.requests):
                entry = {"request": asdict(request)}
                if self.timestamps:
                    entry["timestamp"] = self.timestamps[i]
                file.write(json.dumps(entry) + "\n")

    def load(self):
        path = os.path.join(self.path, f"{self.alias}.jsonl")
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                self.requests.append(Request(**entry["request"]))
                if "timestamp" in entry:
                    self.timestamps.append(entry["timestamp"])
        if self.timestamps:
            assert len(self.timestamps) == len(self.requests)

_WORKLOADS_STATIC = {
    Workload(
        name="Text Conversations",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="text-static",
        modalities="text",
        modality_pct=1.0
    ),
    Workload(
        name="Image Reasoning",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="image-static",
        modalities="image",
        modality_pct=1.0
    ),
    Workload(
        name="Video Description",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="video-static",
        modalities="video",
        modality_pct=1.0
    ),
    Workload(
        name="Audio Captioning",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="audio-static",
        modalities="audio",
        modality_pct=1.0
    ),
}

# Text Conversations with Poisson | Varying request rate
_WORKLOADS_TEXT_POISSON = {
    Workload(
        name=f"Text Conversations with Poisson {rate}",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias=f"text-poisson-{rate}",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    )
    for rate in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
}

# Mixed Modalities with Poisson | Varying request rate | Top 15% replaced
_WORKLOADS_MIX_POISSON_15 = {
    Workload(
        name=f"Mixed Modalities with Poisson {rate} 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias=f"mix-poisson-{rate}-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    )
    for rate in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
}

# Mixed Modalities with Poisson | Varying request rate | Top 30% replaced
_WORKLOADS_MIX_POISSON_30 = {
    Workload(
        name=f"Mixed Modalities with Poisson {rate} 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias=f"mix-poisson-{rate}-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    )
    for rate in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
}

# Mixed Modalities with Poisson | Varying request rate | Top 45% replaced
_WORKLOADS_MIX_POISSON_45 = {
    Workload(
        name=f"Mixed Modalities with Poisson {rate} 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias=f"mix-poisson-{rate}-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    )
    for rate in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
}

# Text Conversations with Gamma | Varying request rate
_WORKLOADS_TEXT_GAMMA = {
    Workload(
        name=f"Text Conversations with Gamma {rate}",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias=f"text-gamma-{rate}",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    )
    for rate in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
}

# Mixed Modalities with Gamma | Varying request rate | Top 15% replaced
_WORKLOADS_MIX_GAMMA_15 = {
    Workload(
        name=f"Mixed Modalities with Gamma {rate} 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias=f"mix-gamma-{rate}-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    )
    for rate in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
}

# Mixed Modalities with Gamma | Varying request rate | Top 30% replaced
_WORKLOADS_MIX_GAMMA_30 = {
    Workload(
        name=f"Mixed Modalities with Gamma {rate} 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias=f"mix-gamma-{rate}-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    )
    for rate in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
}

# Mixed Modalities with Gamma | Varying request rate | Top 45% replaced
_WORKLOADS_MIX_GAMMA_45 = {
        Workload(
        name=f"Mixed Modalities with Gamma {rate} 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias=f"mix-gamma-{rate}-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    )
    for rate in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
}

# Rocks Pebbles Sand with Poisson | Varying request rate | 60% - 30% - 10%
_WORKLOADS_RPS_POISSON_60_30_10 = {
    Workload(
        name=f"Rock - Pebbles - Sand with Poisson {rate} 60%-30%-10%",
        path=os.path.join(WORKLOADS_DIR, "rps-poisson-60-30-10"),
        alias=f"rps-poisson-{rate}-60-30-10",
        arrival_dist="poisson",
        modalities=["text", "image", "video"],
        modality_pct=[0.6, 0.3, 0.1],
        modality_dist="categorical"
    )
    for rate in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
}

# Rocks Pebbles Sand with Poisson | Varying request rate | 45% - 35% - 20%
_WORKLOADS_RPS_POISSON_45_35_20 = {
    Workload(
        name=f"Rock - Pebbles - Sand with Poisson {rate} 45%-35%-20%",
        path=os.path.join(WORKLOADS_DIR, "rps-poisson-45-35-20"),
        alias=f"rps-poisson-{rate}-45-35-20",
        arrival_dist="poisson",
        modalities=["text", "image", "video"],
        modality_pct=[0.6, 0.3, 0.1],
        modality_dist="categorical"
    )
    for rate in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
}

WORKLOADS = _WORKLOADS_STATIC | \
    _WORKLOADS_TEXT_POISSON | \
    _WORKLOADS_MIX_POISSON_15 | \
    _WORKLOADS_MIX_POISSON_30 | \
    _WORKLOADS_MIX_POISSON_45 | \
    _WORKLOADS_TEXT_GAMMA | \
    _WORKLOADS_MIX_GAMMA_15 | \
    _WORKLOADS_MIX_GAMMA_30 | \
    _WORKLOADS_MIX_GAMMA_45 | \
    _WORKLOADS_RPS_POISSON_60_30_10 | \
    _WORKLOADS_RPS_POISSON_45_35_20

def get_workload_by_name(name: str) -> Union[None, Workload]:
    for workload in WORKLOADS:
        if getattr(workload, "name", None) == name:
            return workload
    return None

def get_workload_by_alias(alias: str) -> Union[None, Workload]:
    for workload in WORKLOADS:
        if getattr(workload, "alias", None) == alias:
            return workload
    return None