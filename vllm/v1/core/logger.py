import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import Request

@dataclass
class BlockRecord:
    block_id: int
    event: str  # "hit" or "new"


class Detector:
    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        output_dir: Optional[str] = None,
        filepath: Optional[str] = None,
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.original_block_owner: Dict[int, str] = {}

        if output_dir is None:
            output_dir = os.environ.get("VLLM_BLOCK_LOG_DIR") or "./temp_results/"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = filepath or os.path.join(output_dir, f"block_{timestamp}.jsonl")

    # ----------------------------------------------------
    # Log each block independently
    # ----------------------------------------------------
    def log_request(
        self,
        request: Request,
        new_blocks: List[KVCacheBlock],
        computed_blocks: List[KVCacheBlock],
    ) -> None:

        request_id = request.request_id
        user_id = request.user_md.uid

        # HIT BLOCKS
        for block in computed_blocks:
            self._write_block_log(block.block_id, request_id, user_id, "hit")

        # NEW BLOCKS
        for block in new_blocks:
            self.original_block_owner[block.block_id] = user_id
            self._write_block_log(block.block_id, request_id, user_id, "new")


    # ----------------------------------------------------
    # Write a single block entry to the log file
    # ----------------------------------------------------
    def _write_block_log(self, block_id: int, request_id: str, user_id: str, event: str):
        log_entry = {
            "block_id": block_id,
            "request_id": request_id,
            "user_id": user_id,
            "event": event
        }

        

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

