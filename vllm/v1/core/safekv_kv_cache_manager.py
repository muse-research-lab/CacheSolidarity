# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from collections.abc import Iterable
from typing import Optional
import re
import json
import torch
import asyncio
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from vllm.logger import init_logger
from vllm.utils import cdiv, sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashType,
    KVCacheBlock,
    hash_request_tokens,
)
from vllm.v1.core.specialized_manager import (
    get_specialized_manager,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

# ============================================================
# TWO-TIER PII DETECTION CONFIG
# ============================================================

MODEL_PATH = "./models/Llama-3.2-1B-Instruct"

PII_RULES = [
    ("email", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    ("phone", r"\+?\d[\d\s\-\(\)]{7,}\d"),
    ("url", r"https?://[^\s]+"),
    ("ip_address", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("ipv6", r"\b(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}\b"),

    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("passport", r"\b[A-Z0-9]{6,9}\b"),
    ("driver_license", r"\b[A-Z0-9]{5,15}\b"),
    ("tax_id", r"\b\d{2}-\d{7}\b"),
    ("national_id", r"\b\d{8,14}\b"),
    ("date_of_birth", r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b"),

    ("credit_card", r"\b(?:\d[ -]?){13,16}\b"),
    ("iban", r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    ("bank_account", r"\b\d{8,20}\b"),
    ("routing_number", r"\b\d{9}\b"),
    ("swift_bic", r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b"),

    ("api_key", r"\bsk-[a-zA-Z0-9]{10,}\b"),
    ("jwt_token", r"eyJ[a-zA-Z0-9_\-]+=*\.[a-zA-Z0-9_\-]+=*\.?[a-zA-Z0-9_\-+/=]*"),
    ("aws_access_key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("github_token", r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ("private_key", r"-----BEGIN (RSA|DSA|EC|OPENSSH|PGP)? PRIVATE KEY-----"),
    ("bearer_token", r"Bearer\s+[A-Za-z0-9\-._~+/]+=*"),

    ("medical_record_number", r"\bMRN[:\s]?\d{5,15}\b"),
    ("insurance_id", r"\b[A-Z]{2,5}\d{6,12}\b"),

    ("zip_code", r"\b\d{5}(?:-\d{4})?\b"),
    ("postal_code", r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b"),
    ("street_address", r"\b\d{1,5}\s+\w+(\s\w+)*\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b"),

    ("license_plate", r"\b[A-Z0-9]{2,8}-?[A-Z0-9]{1,8}\b"),
    ("vin", r"\b[A-HJ-NPR-Z0-9]{17}\b"),

    ("bitcoin_address", r"\b(?:bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b"),
    ("ethereum_address", r"\b0x[a-fA-F0-9]{40}\b"),

    ("telegram_handle", r"@[A-Za-z0-9_]{5,32}"),
    ("discord_tag", r".+#\d{4}"),
    ("mac_address", r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"),
]

# ============================================================
# DEVICE + MODEL
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

def _get_free_gpu_memory() -> int:
    """Returns free GPU memory in bytes."""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return free

# --- Before loading detection model ---
mem_before_detector = _get_free_gpu_memory()
logger.info(
    "SAFEKV_GPU_MEMORY_BEFORE_DETECTOR | "
    "free_bytes=%d | free_mb=%.1f",
    mem_before_detector, mem_before_detector / (1024**2)
)

_tokenizer = None
_model = None
_detector_stream = torch.cuda.Stream()
def _get_detector():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    mem_before = _get_free_gpu_memory()
    logger.info(
        "SAFEKV_GPU_BEFORE_DETECTOR | free_mb=%.1f",
        mem_before / (1024**2)
    )

    logger.info(f"SAFEKV_USING_DEVICE={device}")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    _model.to(device)
    _model.eval()
   
    mem_after = _get_free_gpu_memory()
    logger.info(
        "SAFEKV_GPU_AFTER_DETECTOR | free_mb=%.1f | detector_used_mb=%.1f",
        mem_after / (1024**2),
        (mem_before - mem_after) / (1024**2)
    )
    
    logger.info("SAFEKV_MODEL_LOADED")
    return _tokenizer, _model

# At module level in safekv_kv_cache_manager.py
_detector_lock = None

def _get_detector_lock():
    global _detector_lock
    if _detector_lock is None:
        _detector_lock = asyncio.Lock()
    return _detector_lock


# ============================================================
# REGEX DETECTOR
# ============================================================

def regex_pii_detector(text):

    entities = []

    for pii_type, pattern in PII_RULES:

        matches = re.findall(pattern, text)

        for match in matches:

            entities.append({
                "type": pii_type,
                "value": match,
            })

    return {
        "contains_pii": len(entities) > 0,
        "entities": entities,
    }

# ============================================================
# LLM DETECTOR
# ============================================================

def llm_pii_detector(text):
    tokenizer, model = _get_detector()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a PII detection API. "
                "Return ONLY valid JSON. "
                "Do not explain anything."
            ),
        },
        {
            "role": "user",
            "content": f"""
Detect PII entities in the following text.

Return ONLY JSON:

{{
  "contains_pii": true,
  "entities": [
    {{
      "type": "name|email|phone|address|id|medical|financial|other",
      "value": "text"
    }}
  ]
}}

Text:
{text}
"""
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.cuda.stream(_detector_stream):

        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.current_stream().wait_stream(_detector_stream)
    generated_tokens = output[0][
        inputs["input_ids"].shape[-1]:
    ]

    decoded = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    # logger.info(
    #     f"SAFEKV_LLM_OUTPUT | output={decoded}"
    # )

    try:

        start = decoded.find("{")
        end = decoded.rfind("}") + 1

        if start != -1 and end != -1:

            json_text = decoded[start:end]

            parsed = json.loads(json_text)

            return parsed

    except Exception as e:

        logger.warning(
            f"SAFEKV_JSON_PARSE_ERROR | error={e}"
        )

    return {
        "contains_pii": False,
        "entities": [],
    }

# ============================================================
# TWO-TIER DETECTOR
# ============================================================

def detect_pii_entities(text):

    regex_result = regex_pii_detector(text)

    if regex_result["contains_pii"]:

        # logger.info(
        #     f"SAFEKV_REGEX_HIT | "
        #     f"entities={regex_result['entities']}"
        # )

        return {
            "contains_pii": False,
            "tier": "regex",
            "entities": regex_result["entities"],
        }

    # llm_result = llm_pii_detector(text)

    # if llm_result["contains_pii"]:

    #     # logger.info(
    #     #     f"SAFEKV_LLM_HIT | "
    #     #     f"entities={llm_result['entities']}"
    #     # )

    #     return {
    #         "contains_pii": False,
    #         "tier": "llm",
    #         "entities": llm_result["entities"],
    #     }

    # # logger.info("SAFEKV_NO_PII")

    return {
        "contains_pii": False,
        "tier": "none",
        "entities": [],
    }

# ============================================================
# MAIN SAFEKV PII CHECK
# ============================================================


def contains_pii(
    text: str,
    block_size: int = 16,
    tokenizer=None,
) -> int:

    if not text:
        return -1

    detection = detect_pii_entities(text)

    if not detection["contains_pii"]:

        # logger.info("NO_PII_DETECTED_IN_PROMPT")

        return -1

    entities = detection["entities"]

    first_pii_char_pos = -1

    for entity in entities:

        value = entity.get("value")

        if not value:
            continue

        pos = text.find(value)

        if pos != -1:

            if (
                first_pii_char_pos == -1
                or pos < first_pii_char_pos
            ):
                first_pii_char_pos = pos

    if first_pii_char_pos == -1:

        # logger.warning(
        #     "PII_FOUND_BUT_POSITION_NOT_FOUND"
        # )

        return -1

    if tokenizer is not None:

        prefix_text = text[:first_pii_char_pos]

        try:

            prefix_tokens = tokenizer.encode(
                prefix_text,
                add_special_tokens=False,
            )

            pii_token_idx = len(prefix_tokens)

        except Exception as e:

            # logger.warning(
            #     f"TOKENIZER_ENCODE_FAILED | "
            #     f"error={e} | "
            #     f"falling back_to_approximation=True"
            # )

            pii_token_idx = first_pii_char_pos // 4

    else:

        # logger.warning(
        #     "NO_TOKENIZER_PROVIDED | "
        #     "using approximation"
        # )

        pii_token_idx = first_pii_char_pos // 4

    pii_block_idx = pii_token_idx // block_size

    # logger.warning(
    #     f"PII_DETECTED_IN_PROMPT_BLOCK | "
    #     f"tier={detection['tier']} | "
    #     f"block_idx={pii_block_idx} | "
    #     f"pii_token_idx={pii_token_idx} | "
    #     f"char_pos={first_pii_char_pos}"
    # )

    return pii_block_idx


class SafeKVCacheManager:
    # pass
    # ============================================================
    # KEEP EVERYTHING BELOW EXACTLY THE SAME
    # ============================================================

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        num_preallocate_tokens: int = 0,
        log_stats: bool = False,
        enable_safekv: bool = False,
        enable_CacheSolidarity: bool = False,
    ) -> None:

        assert len(
            kv_cache_config.kv_cache_groups
        ) == 1

        kv_cache_spec = (
            kv_cache_config
            .kv_cache_groups[0]
            .kv_cache_spec
        )

        self.block_size = (
            kv_cache_spec.block_size
        )

        self.num_gpu_blocks = (
            kv_cache_config.num_blocks
        )

        self.max_model_len = max_model_len

        self.enable_safekv = enable_safekv
        self.enable_CacheSolidarity = enable_CacheSolidarity

        self.max_num_blocks_per_req = cdiv(
            max_model_len,
            self.block_size,
        )

        self.enable_caching = enable_caching

        self.caching_hash_fn = (
            sha256
            if caching_hash_algo == "sha256"
            else hash
        )

        self.log_stats = log_stats

        self.num_preallocate_tokens = (
            num_preallocate_tokens
        )

        self.num_preallocate_blocks = cdiv(
            num_preallocate_tokens,
            self.block_size,
        )

        self.block_pool = BlockPool(
            self.num_gpu_blocks,
            enable_caching,
        )

        self.specialized_manager = (
            get_specialized_manager(
                kv_cache_spec=kv_cache_spec,
                block_pool=self.block_pool,
            )
        )

        self.req_to_blocks: defaultdict[
            str,
            list[KVCacheBlock]
        ] = defaultdict(list)

        self.req_to_block_hashes: defaultdict[
            str,
            list[BlockHashType]
        ] = defaultdict(list)

        self.num_cached_block: dict[
            str,
            int,
        ] = {}

        self.req_to_blocks_snapshot: dict[
            str,
            list[KVCacheBlock]
        ] = {}

        self.req_to_pii_block_idx: dict[
            str,
            int,
        ] = {}

        self.prefix_cache_stats = (
            PrefixCacheStats()
        )

    @property
    def usage(self) -> float:
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(
        self,
    ) -> PrefixCacheStats:

        stats = self.prefix_cache_stats

        self.prefix_cache_stats = (
            PrefixCacheStats()
        )

        return stats
    
    def finalize_request_privacy(
        self,
        request_id: str,
        pii_block_idx: int,
    ) -> None:

        self.req_to_pii_block_idx[request_id] = pii_block_idx

        self.promote_request_blocks_to_public(request_id)

        # NOW cleanup snapshot
        self.req_to_blocks_snapshot.pop(request_id, None)
        self.req_to_pii_block_idx.pop(request_id, None)

        # logger.info(
        #     f"FINALIZE_REQUEST_PRIVACY | "
        #     f"request_id={request_id} | "
        #     f"pii_block_idx={pii_block_idx}"
        # )


    def get_computed_blocks(
        self,
        request: Request,
    ) -> tuple[list[KVCacheBlock], int]:

        if not self.enable_caching:
            return [], 0

        block_hashes = (
            self.req_to_block_hashes[
                request.request_id
            ]
        )

        if not block_hashes:

            block_hashes = (
                hash_request_tokens(
                    self.caching_hash_fn,
                    self.block_size,
                    request,
                )
            )

            self.req_to_block_hashes[
                request.request_id
            ] = block_hashes

        self.prefix_cache_stats.requests += 1

        if (
            request.sampling_params
            .prompt_logprobs
            is not None
        ):
            return [], 0

        if (
            len(block_hashes)
            * self.block_size
            == request.num_tokens
        ):
            last_block_hash = (
                block_hashes.pop()
            )
        else:
            last_block_hash = None

        computed_blocks = (
            self.specialized_manager
            .find_longest_cache_hit(
                block_hashes,
                request,
            )
        )

        request_uid = getattr(
            request.user_md,
            "uid",
            None,
        )

        # ============================================================
        # SECURITY FILTER + REUSE LOGGING
        # Replace your current SECURITY FILTER section inside
        # get_computed_blocks()
        # ============================================================

        filtered_blocks = []
        current_user = request_uid

        for blk in computed_blocks:

            is_private = getattr(blk, "is_private", True)
            owner_id = getattr(blk, "owner_id", None)

            # PUBLIC => everyone can reuse
            if not is_private:
                # logger.info(
                #     "CACHE_REUSE_PUBLIC | "
                #     f"user={current_user} | "
                #     f"owner={owner_id} | "
                #     f"block={blk.block_id}"
                # )
                filtered_blocks.append(blk)
                continue

            # PRIVATE => only owner can reuse
            if owner_id == current_user:
                # logger.info(
                #     "CACHE_REUSE_PRIVATE_OWNER | "
                #     f"user={current_user} | "
                #     f"block={blk.block_id}"
                # )
                filtered_blocks.append(blk)
                continue

            # Different user => stop prefix chain
            # logger.info(
            #     "CACHE_REUSE_DENIED_PRIVATE | "
            #     f"user={current_user} | "
            #     f"owner={owner_id} | "
            #     f"block={blk.block_id}"
            # )
            break

        computed_blocks = filtered_blocks
        # logger.info(
        #     "CACHE_REUSE_SUMMARY | "
        #     f"user={current_user} | "
        #     f"num_hits={len(computed_blocks)} | "
        #     f"blocks={[b.block_id for b in computed_blocks]}"
        # )

        self.prefix_cache_stats.queries += len(
            block_hashes
        )

        self.prefix_cache_stats.hits += len(
            computed_blocks
        )

        if last_block_hash is not None:
            block_hashes.append(
                last_block_hash
            )

        num_computed_tokens = (
            len(computed_blocks)
            * self.block_size
        )

        return (
            computed_blocks,
            num_computed_tokens,
        )

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[
            list[KVCacheBlock]
        ] = None,
        num_lookahead_tokens: int = 0,
    ) -> Optional[list[KVCacheBlock]]:

        if num_tokens == 0:
            raise ValueError(
                "num_tokens must be > 0"
            )

        new_computed_blocks = (
            new_computed_blocks or []
        )

        req_blocks = self.req_to_blocks[
            request.request_id
        ]

        removed_blocks = (
            self.specialized_manager
            .remove_skipped_blocks(
                req_blocks,
                request.num_computed_tokens,
            )
        )

        self.block_pool.free_blocks(
            removed_blocks
        )

        num_computed_tokens = (
            request.num_computed_tokens
            + len(new_computed_blocks)
            * self.block_size
        )

        num_required_blocks = cdiv(
            num_computed_tokens
            + num_tokens
            + num_lookahead_tokens,
            self.block_size,
        )

        num_new_blocks = (
            num_required_blocks
            - len(req_blocks)
            - len(new_computed_blocks)
        )

        num_evictable_computed_blocks = sum(
            1
            for blk in new_computed_blocks
            if blk.ref_cnt == 0
        )

        if (
            num_new_blocks
            > self.block_pool.get_num_free_blocks()
            - num_evictable_computed_blocks
        ):
            return None

        if self.enable_caching:

            self.block_pool.touch(
                new_computed_blocks
            )

        req_blocks.extend(
            new_computed_blocks
        )

        if num_new_blocks <= 0:

            new_blocks = []

        else:

            num_preallocate_blocks = max(
                0,
                self.num_preallocate_blocks
                - num_lookahead_tokens
                // self.block_size,
            )

            num_new_blocks = min(
                num_new_blocks
                + num_preallocate_blocks,

                self.block_pool
                .get_num_free_blocks(),

                self.max_num_blocks_per_req
                - len(req_blocks),
            )

            assert num_new_blocks > 0

            new_blocks = (
                self.block_pool
                .get_new_blocks(
                    num_new_blocks
                )
            )

            request_uid = getattr(
                request.user_md,
                "uid",
                None,
            )

            for block in new_blocks:

                block.owner_id = request_uid

                if self.enable_safekv:

                    block.is_private = True
                    block.scan_pending = True
                    block.pii_scanned = False

                else:

                    block.is_private = False
                    block.scan_pending = False
                    block.pii_scanned = True

                # logger.debug(
                #     f"BLOCK_INIT | "
                #     f"user={request_uid} | "
                #     f"request={request.request_id} | "
                #     f"block={block.block_id}"
                # )

            req_blocks.extend(
                new_blocks
            )

        self.req_to_blocks_snapshot[
            request.request_id
        ] = list(req_blocks)

        if not self.enable_caching:
            return new_blocks

        num_cached_blocks = (
            self.num_cached_block.get(
                request.request_id,
                len(new_computed_blocks),
            )
        )

        num_full_blocks_after_append = (
            num_computed_tokens
            + num_tokens
            - len(request.spec_token_ids)
        ) // self.block_size

        self.block_pool.cache_full_blocks(
            request=request,

            blocks=req_blocks,

            block_hashes=(
                self.req_to_block_hashes[
                    request.request_id
                ]
            ),

            num_cached_blocks=num_cached_blocks,

            num_full_blocks=(
                num_full_blocks_after_append
            ),

            block_size=self.block_size,

            hash_fn=self.caching_hash_fn,
        )

        self.num_cached_block[
            request.request_id
        ] = num_full_blocks_after_append

        return new_blocks

    def free(
        self,
        request: Request,
    ) -> None:

        blocks = self.req_to_blocks.pop(
            request.request_id,
            [],
        )

        ordered_blocks: Iterable[
            KVCacheBlock
        ] = blocks

        if self.enable_caching:
            ordered_blocks = reversed(
                blocks
            )

        self.block_pool.free_blocks(
            ordered_blocks
        )

        self.num_cached_block.pop(
            request.request_id,
            None,
        )

        pass

    def promote_request_blocks_to_public(
    self,
    request_id: str,
    ) -> None:

        blocks = (
            self.req_to_blocks.get(request_id)
            or self.req_to_blocks_snapshot.get(request_id, [])
        )

        if not blocks:
            # logger.info(
            #     f"PROMOTION_SKIPPED_REQUEST_GONE | request_id={request_id}"
            # )
            return

        pii_block_idx = self.req_to_pii_block_idx.get(
            request_id,
            -1,
        )

        promoted = 0

        for idx, block in enumerate(blocks):

            block.scan_pending = False
            block.pii_scanned = True
            block.position_in_request = idx
            block.owner_pii_block_idx = pii_block_idx

            should_promote = (
                pii_block_idx == -1      # no pii => all public
                or idx < pii_block_idx   # safe prefix only
            )

            if should_promote and block.is_private:
                block.is_private = False
                promoted += 1

        # logger.info(
        #     f"PROMOTION_SUMMARY | "
        #     f"request_id={request_id} | "
        #     f"promoted={promoted} | "
        #     f"total={len(blocks)} | "
        #     f"pii_block_idx={pii_block_idx}"
        # )

    def reset_prefix_cache(
        self,
    ) -> bool:

        if (
            self.block_pool
            .reset_prefix_cache()
        ):

            self.prefix_cache_stats.reset = True

            # logger.warning(
            #     "PREFIX_CACHE_RESET"
            # )

            return True

        return False

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> int:

        assert (
            request.status
            == RequestStatus.RUNNING
        )

        blocks = self.req_to_blocks[
            request.request_id
        ]

        num_common_blocks = 0

        for block in blocks:

            if (
                block.ref_cnt
                == num_running_requests
            ):
                num_common_blocks += 1
            else:
                break

        return num_common_blocks

    def free_block_hashes(
        self,
        request: Request,
    ) -> None:

        self.req_to_block_hashes.pop(
            request.request_id,
            None,
        )