import json
from dataclasses import dataclass
from typing import List, Dict

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import Request


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
    blocks_in_danger: List[int]
    danger_by_owner: Dict[str, List[int]]


class Detector:
    def __init__(self, kv_cache_manager: KVCacheManager) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.original_block_owner: Dict[int, str] = {}
        self.blocks_by_owner: Dict[str, List[int]] = {}

        # Persistent global trackers
        self.global_blocks_in_danger: List[int] = []
        self.global_danger_by_owner: Dict[str, List[int]] = {}

    def log_request(
        self,
        request: Request,
        new_blocks: List[KVCacheBlock],
        computed_blocks: List[KVCacheBlock],
    ) -> RequestRecord:
        request_id = request.request_id
        user_id = request.user_md.uid

        #print(f"\n=== Processing Request {request_id} from User {user_id} ===")

        blocks_md: List[BlockMD] = []
        hits_by_owner: Dict[str, List[int]] = {}

        # STEP 1: Register HIT blocks
        for block in computed_blocks:
            owner = self.original_block_owner.get(block.block_id, "unknown")
            blocks_md.append(BlockMD(block.block_id, owner, "hit"))
            hits_by_owner.setdefault(owner, []).append(block.block_id)
            print(f"HIT block {block.block_id} originally owned by {owner}")

        # STEP 2: Register NEW blocks
        for block in new_blocks:
            self.original_block_owner[block.block_id] = user_id
            blocks_md.append(BlockMD(block.block_id, user_id, "new"))
            self.blocks_by_owner.setdefault(user_id, []).append(block.block_id)
            print(f"NEW block {block.block_id} created by {user_id}")

        # STEP 3: Detect danger (first new after last hit)
        new_dangers: List[int] = []
        last_hit_index = None
        for i, b in enumerate(blocks_md):
            if b.event == "hit":
                last_hit_index = i
            elif b.event == "new" and last_hit_index is not None and i == last_hit_index + 1:
                new_dangers.append(b.block_id)
                print(f"Block {b.block_id} marked as IN DANGER (first new after last hit)")

        # STEP 4: Propagate danger to owners
        for owner, hit_ids in hits_by_owner.items():
            if owner == "unknown":
                continue
            owner_seq = self.blocks_by_owner.get(owner, [])
            if not owner_seq:
                continue
            max_idx = max((owner_seq.index(hid) for hid in hit_ids if hid in owner_seq), default=-1)
            next_idx = max_idx + 1
            if 0 <= max_idx < len(owner_seq) and next_idx < len(owner_seq):
                next_block = owner_seq[next_idx]
                if next_block not in new_dangers and next_block not in self.global_blocks_in_danger:
                    new_dangers.append(next_block)
                    print(f"Propagated danger to block {next_block} (after last hit {owner_seq[max_idx]} of user {owner})")

        # STEP 5: Update global danger state
        for bid in new_dangers:
            if bid not in self.global_blocks_in_danger:
                self.global_blocks_in_danger.append(bid)
                owner = self.original_block_owner.get(bid, "unknown")
                self.global_danger_by_owner.setdefault(owner, []).append(bid)

        # STEP 6: #print summaries
        #print("\n--- Request Summary ---")
        for b in blocks_md:
            print(f"  Block {b.block_id}: owner={b.block_owner_id}, event={b.event}")

        print(f"\nNew blocks in danger (this request): {new_dangers}")
        print(f"Accumulated blocks in danger (global): {self.global_blocks_in_danger}")
        print(f"Global danger grouped by owner: {json.dumps(self.global_danger_by_owner, indent=2)}")
        print("--- End of Request ---\n")

        return RequestRecord(
            request_id=request_id,
            user_id=user_id,
            blocks_md=blocks_md,
            blocks_in_danger=list(self.global_blocks_in_danger),
            danger_by_owner=dict(self.global_danger_by_owner),
        )
