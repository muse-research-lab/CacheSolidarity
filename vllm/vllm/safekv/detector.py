import re
import asyncio
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

class PrivacyLabel(Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    UNCERTAIN = "uncertain"

# ── Tier 1: Rule-based ──────────────────────────────────────────────────

PII_PATTERNS = {
    "email":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone":       r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn":         r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address":  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "us_bank":     r"\b\d{8,17}\b",  # simplified
    "passport":    r"\b[A-Z]{1,2}\d{6,9}\b",
    "driver_lic":  r"\b[A-Z]\d{7}\b",
}

# Extend via config file
def load_custom_patterns(config_path: str) -> dict:
    import json
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

class Tier1Detector:
    def __init__(self, custom_config_path: Optional[str] = None):
        self.patterns = {k: re.compile(v) for k, v in PII_PATTERNS.items()}
        if custom_config_path:
            custom = load_custom_patterns(custom_config_path)
            self.patterns.update({k: re.compile(v) for k, v in custom.items()})
        
        # Trie-based blacklist for fixed tokens
        self.blacklist = set()  # populate from config

    def detect(self, text: str) -> PrivacyLabel:
        for name, pattern in self.patterns.items():
            if pattern.search(text):
                return PrivacyLabel.PRIVATE
        for term in self.blacklist:
            if term in text:
                return PrivacyLabel.PRIVATE
        return PrivacyLabel.UNCERTAIN  # pass to Tier 2

# ── Tier 2: Lightweight LLM detector ───────────────────────────────────

class Tier2Detector:
    """
    Uses a small model (e.g. Llama-3.2-1B) loaded separately.
    Requires ~2.6GB GPU HBM.
    """
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B"):
        from transformers import pipeline
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device=0,
            max_new_tokens=10,
        )
        self.prompt_template = (
            "Does the following text contain personally identifiable information "
            "(PII) such as names, emails, phone numbers, IDs, passwords, or "
            "financial data? Answer only YES or NO.\n\nText: {text}\n\nAnswer:"
        )

    def detect(self, text: str) -> PrivacyLabel:
        prompt = self.prompt_template.format(text=text[:512])  # truncate
        result = self.pipe(prompt)[0]["generated_text"]
        answer = result.split("Answer:")[-1].strip().upper()
        if "YES" in answer:
            return PrivacyLabel.PRIVATE
        elif "NO" in answer:
            return PrivacyLabel.PUBLIC
        return PrivacyLabel.UNCERTAIN

# ── Tier 3: Context-aware validation via in-service LLM ────────────────

class Tier3Detector:
    """
    Reuses the serving LLM via the engine's generate() path.
    Only called for ~2-8% of blocks (uncertain after Tier 1+2).
    """
    def __init__(self, llm_engine):
        self.engine = llm_engine  # reference to running vLLM engine

    async def detect(self, text: str, conversation_history: List[str]) -> PrivacyLabel:
        context = "\n".join(conversation_history[-5:])  # last 5 turns
        prompt = (
            f"Given the following conversation context:\n{context}\n\n"
            f"Does this message contain sensitive or private information "
            f"that should not be shared? Message: {text}\n"
            f"Answer YES or NO only."
        )
        # Submit as low-priority internal request
        outputs = await self.engine.generate_async(prompt, max_tokens=5)
        answer = outputs.strip().upper()
        if "YES" in answer:
            return PrivacyLabel.PRIVATE
        return PrivacyLabel.PUBLIC

# ── Pipeline orchestrator ───────────────────────────────────────────────

class HybridPrivacyDetector:
    def __init__(self, llm_engine=None, custom_config_path=None,
                 tier2_model="meta-llama/Llama-3.2-1B"):
        self.tier1 = Tier1Detector(custom_config_path)
        self.tier2 = Tier2Detector(tier2_model)
        self.tier3 = Tier3Detector(llm_engine) if llm_engine else None
        self._detection_queue = asyncio.Queue()

    def detect_sync(self, text: str, history: List[str] = None) -> PrivacyLabel:
        """Synchronous fast path (Tier 1 + 2 only)."""
        result = self.tier1.detect(text)
        if result != PrivacyLabel.UNCERTAIN:
            return result
        return self.tier2.detect(text)

    async def detect_async(self, text: str, history: List[str] = None) -> PrivacyLabel:
        """Full async pipeline including Tier 3."""
        result = self.tier1.detect(text)
        if result != PrivacyLabel.UNCERTAIN:
            return result
        result = self.tier2.detect(text)
        if result != PrivacyLabel.UNCERTAIN:
            return result
        if self.tier3 and history:
            return await self.tier3.detect(text, history)
        return PrivacyLabel.PRIVATE  # conservative fallback