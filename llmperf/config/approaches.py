from dataclasses import dataclass
from typing import Literal, Union

@dataclass
class Approach:
    name: str
    alias: str
    scheduling_policy: Literal["fcfs", "priority"] = "fcfs"
    enable_custom_scheduler: bool = False
    enable_chunked_prefill: bool = False

    def __hash__(self):
        return hash((self.name, self.alias))
    
    def __eq__(self, other):
        if isinstance(other, Approach):
            return self.name == other.name and self.alias == other.alias
        return False

APPROACHES = {
    Approach(
        name="Isolation",
        alias="iso"
    ),
    Approach(
        name="Vanilla vLLM",
        alias="vllm"
    ),
}

def get_approach_by_name(name: str) -> Union[None, Approach]:
    for approach in APPROACHES:
        if getattr(approach, "name", None) == name:
            return approach
    return None

def get_approach_by_alias(alias: str) -> Union[None, Approach]:
    for approach in APPROACHES:
        if getattr(approach, "alias", None) == alias:
            return approach
    return None