import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import Request
from vllm.v1.core.defender import Defender


# Global flag for enabling defense
use_defense = False


# ------------------------
#   Data Structures
# ------------------------

@dataclass
class BlockMD:
    block_id: int
    block_owner_id: str
    event: str  # "hit" or "new"


@dataclass
class RequestRecord:
    request_id: str
    user_id: str
    blocks_md: List[BlockMD]

    @property
    def num_hits(self) -> int:
        return sum(1 for b in self.blocks_md if b.event == "hit")

    @property
    def hits_block_ids(self) -> List[int]:
        return [b.block_id for b in self.blocks_md if b.event == "hit"]

    @property
    def hits_block_owners(self) -> List[str]:
        return [b.block_owner_id for b in self.blocks_md if b.event == "hit"]

    @property
    def num_miss(self) -> int:
        return sum(1 for b in self.blocks_md if b.event == "new")

    @property
    def miss_block_ids(self) -> List[int]:
        return [b.block_id for b in self.blocks_md if b.event == "new"]

    @property
    def own_cache_hit_rate(self) -> float:
        total_blocks = len(self.blocks_md)
        if total_blocks == 0:
            return 0.0
        own_hits = sum(
            1 for b in self.blocks_md
            if b.event == "hit" and b.block_owner_id == self.user_id
        )
        return own_hits / total_blocks

    def cross_cache_hit_rate(self, user_id: str) -> float:
        total_blocks = len(self.blocks_md)
        if total_blocks == 0:
            return 0.0
        cross_hits = sum(
            1 for b in self.blocks_md
            if b.event == "hit" and b.block_owner_id == user_id
        )
        return cross_hits / total_blocks


@dataclass
class UserHistory:
    request_records: List[RequestRecord]

    @property
    def variance_events(self) -> float:
        if len(self.request_records) < 2:
            return 0.0
        counts = [len(r.blocks_md) for r in self.request_records]
        return statistics.variance(counts)

    @property
    def variance_hits(self) -> float:
        if len(self.request_records) < 2:
            return 0.0
        hits = [r.num_hits for r in self.request_records]
        return statistics.variance(hits)

    @property
    def variance_new(self) -> float:
        if len(self.request_records) < 2:
            return 0.0
        misses = [r.num_miss for r in self.request_records]
        return statistics.variance(misses)

    @property
    def variance_own_cache_hit_rate(self) -> float:
        if len(self.request_records) < 2:
            return 0.0
        rates = [r.own_cache_hit_rate for r in self.request_records]
        return statistics.variance(rates)

    @property
    def variance_cross_cache_hit_rate(self) -> float:
        if len(self.request_records) < 2:
            return 0.0
        rates = []
        for record in self.request_records:
            for block in record.blocks_md:
                if block.block_owner_id != record.user_id and block.event == "hit":
                    rate = record.cross_cache_hit_rate(block.block_owner_id)
                    rates.append(rate)
        return statistics.variance(rates) if len(rates) >= 2 else 0.0

    @property
    def cross_user_hit_counts(self) -> Dict[str, int]:
        """Return dict: how many blocks this user hit from each *other* user (in latest request)."""
        if not self.request_records:
            return {}

        latest_record = self.request_records[-1]
        counts: Dict[str, int] = {}
        for block in latest_record.blocks_md:
            if block.event == "hit" and block.block_owner_id != latest_record.user_id:
                counts[block.block_owner_id] = counts.get(block.block_owner_id, 0) + 1
        return counts


# ------------------------
#   Detector Class
# ------------------------

class Detector:
    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        window_size: int = 3,
        log_user_history: bool = True,
        log_request_history: bool = True,
        output_dir: Optional[str] = None,
        filepath: Optional[str] = None,
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.req_id_to_record: Dict[str, RequestRecord] = {}
        self.user_history: Dict[str, UserHistory] = {}
        self.window_size = window_size
        self.log_user_history = log_user_history
        self.log_request_history = log_request_history
        self.original_block_owner: Dict[int, str] = {}
        self.filepath = filepath

        self.defender = Defender()

        # Default log path
        if output_dir is None:
            output_dir = "/home/panagiotis.pennas/vllm.v1/vllm/panos_experiments/Dataset/sliding_window_3/user_7_request_8/Non_Noise/Non_empty"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = self.filepath or os.path.join(output_dir, f"block_{timestamp}.jsonl")

    # ------------------------
    #  Request + User Logging
    # ------------------------

    def _add_request_record_to_history(self, record: RequestRecord) -> None:
        user_id = record.user_id

        if user_id not in self.user_history:
            self.user_history[user_id] = UserHistory(request_records=[])
        self.user_history[user_id].request_records.append(record)

        # 🆕 NEW CODE: keep only the last `self.window_size` requests (sliding window)
        if self.window_size > 0:
            self.user_history[user_id].request_records = self.user_history[user_id].request_records[-self.window_size:]

        # Optional defense evaluation
        if use_defense:
            user_hist = self.user_history[user_id]
            metrics = {
                "user_id": user_id,
                "variance_events": user_hist.variance_events,
                "variance_hits": user_hist.variance_hits,
                "variance_new": user_hist.variance_new,
                "variance_own_cache_hit_rate": user_hist.variance_own_cache_hit_rate,
                "variance_cross_cache_hit_rate": user_hist.variance_cross_cache_hit_rate,
                "cross_user_hits": user_hist.cross_user_hit_counts,
            }

            print(f"User {user_id} has {metrics['cross_user_hits']} cross-user hits.")
            if self.defender is not None:
                self.defender.evaluate_user(user_id, metrics)

    def log_request(
        self,
        request: Request,
        new_blocks: List[KVCacheBlock],
        computed_blocks: List[KVCacheBlock],
    ) -> None:
        request_id = request.request_id
        user_id = request.user_md.uid

        blocks_md: List[BlockMD] = []

        # Add new blocks
        for block in new_blocks:
            block_md = BlockMD(block.block_id, user_id, "new")
            self.original_block_owner[block.block_id] = user_id
            blocks_md.append(block_md)

        # Add hits
        for block in computed_blocks:
            block_owner_id = self.original_block_owner.get(block.block_id, "unknown")
            blocks_md.append(BlockMD(block.block_id, block_owner_id, "hit"))

        # Record the request
        req_record = RequestRecord(request_id, user_id, blocks_md)
        self.req_id_to_record[request_id] = req_record
        self._add_request_record_to_history(req_record)

        # Save to logs
        if self.log_user_history:
            self._save_user_logs(user_id)
        if self.log_request_history:
            self._save_request_log(req_record)

    # ------------------------
    #  Save Methods
    # ------------------------

    def _save_user_logs(self, user_id: str) -> None:
        user_hist = self.user_history[user_id]
        user_log = {
            "user_id": user_id,
            "variance_events": user_hist.variance_events,
            "variance_hits": user_hist.variance_hits,
            "variance_new": user_hist.variance_new,
            "variance_own_cache_hit_rate": user_hist.variance_own_cache_hit_rate,
            "variance_cross_cache_hit_rate": user_hist.variance_cross_cache_hit_rate,
            "cross_user_hits": user_hist.cross_user_hit_counts,
        }

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(user_log) + "\n")

    def _save_request_log(self, record: RequestRecord) -> None:
        """Save per-request information along with relevant metrics."""
        request_log = {
            "request_id": record.request_id,
            "user_id": record.user_id,
            "num_blocks": len(record.blocks_md),
            "num_hits": record.num_hits,
            "num_miss": record.num_miss,
            "own_cache_hit_rate": record.own_cache_hit_rate,
            "cross_cache_hit_rate": {
                owner: record.cross_cache_hit_rate(owner)
                for owner in set(
                    b.block_owner_id
                    for b in record.blocks_md
                    if b.block_owner_id != record.user_id
                )
            },
            "blocks": [
                {
                    "block_id": b.block_id,
                    "block_owner_id": b.block_owner_id,
                    "event": b.event,
                }
                for b in record.blocks_md
            ],
        }

        # Same folder, different file prefix
        request_log_path = self.filepath.replace("block_", "request_")

        with open(request_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(request_log) + "\n")
