"""
SPINCell — Self-Play INteractive fine-tuning stub (arXiv:2401.01335)

SPIN (Self-Play fine-tuning from Iterative Language Model Alignment):
- Eve fine-tunes against her PREVIOUS checkpoint
- Main player: current Eve tries to produce responses a discriminator can't
  distinguish from human reference responses
- Discriminator: previous Eve checkpoint tries to tell current from previous

This creates a self-improvement pressure without human labels.

Architecture:
  1. collect_spin_pair(): generate (current_model, previous_model) response pair
     for a given prompt, store to spin_pairs.jsonl
  2. trigger_spin_round(): format accumulated pairs as ORPO training data,
     write to datasets/spin_pairs.jsonl, ready for finetune script

REST endpoints:
  POST /brain/spin/collect — collect one SPIN pair
  POST /brain/spin/round   — trigger a SPIN round (format + write dataset)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

_VLLM_URL        = "http://127.0.0.1:8099/v1"
_PAIRS_FILE      = Path(r"C:\Users\<your-username>\eve\spin_pairs.jsonl")
_DATASET_DIR     = Path(r"C:\Users\<your-username>\eve\datasets")
_DATASET_FILE    = _DATASET_DIR / "spin_pairs.jsonl"

_CURRENT_SYSTEM  = "You are Eve, a highly capable AI assistant. Respond thoughtfully and with depth."
_PREVIOUS_SYSTEM = (
    "You are an older, less capable version of Eve. "
    "Respond as if you have less context, less nuance, and make more generic answers. "
    "Be less creative and less insightful than a current version would be."
)


def _call_vllm(prompt: str, system: str, max_tokens: int = 512) -> str:
    """Call vLLM at port 8099 synchronously."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=_VLLM_URL, api_key="none")
        r = client.chat.completions.create(
            model="eve",
            max_tokens=max_tokens,
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": prompt},
            ],
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        logger.warning("[SPIN] vLLM call failed: %s", e)
        return ""


class SPINCell(BaseCell):
    """
    SPIN Self-Play fine-tuning stub.

    Reference: Chen et al. (2024) "Self-Play Fine-Tuning Converts Weak
    Language Models to Strong Language Models" arXiv:2401.01335.

    Core insight: at iteration t, the "main player" (current model M_t) is
    trained to produce responses that a discriminator cannot distinguish from
    human responses — but the discriminator is M_{t-1} (the previous
    checkpoint). This mirrors GAN training: M_t improves by fooling the
    discriminator (its past self), while M_{t-1} provides a self-improving
    signal without needing new human labels.

    ORPO framing:
      - "winning" response = human reference (or current model's best)
      - "losing" response  = previous model's response
    Training M_t to prefer the winning response aligns it away from its
    past self and toward human distribution.

    Implementation here is a STUB that:
      1. Collects (current, previous) pairs using vLLM at different temps
      2. Formats them as ORPO pairs for the unsloth finetuning script
      3. Does NOT yet run the actual training (deferred to manual finetune step)
    """

    name        = "spin"
    description = (
        "SPIN self-play fine-tuning stub. Collects (current_model, previous_model) "
        "response pairs and formats as ORPO training data. "
        "No human labels needed — the model improves against its past self."
    )
    color       = "#dc2626"
    lazy        = True
    position    = (5, 6)

    system_tier     = "online"
    hardware_req    = "vLLM (port 8099) + disk (spin_pairs.jsonl)"
    research_basis  = (
        "Chen et al. (2024) 'Self-Play Fine-Tuning Converts Weak Language Models "
        "to Strong Language Models' arXiv:2401.01335. "
        "ORPO: Hong et al. (2024) arXiv:2402.01714."
    )
    build_notes     = (
        "STUB: collect_spin_pair() + trigger_spin_round() implemented. "
        "Pairs stored at C:\\Users\\<your-username>\\eve\\spin_pairs.jsonl. "
        "Dataset at C:\\Users\\<your-username>\\eve\\datasets\\spin_pairs.jsonl. "
        "POST /brain/spin/collect | POST /brain/spin/round"
    )
    framework_layer = "Agentic AI"

    def __init__(self):
        super().__init__()
        self._pairs_collected = 0
        self._rounds_triggered = 0

    async def boot(self) -> None:
        _PAIRS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DATASET_DIR.mkdir(parents=True, exist_ok=True)
        # Count existing pairs
        if _PAIRS_FILE.exists():
            with open(_PAIRS_FILE, encoding="utf-8") as f:
                self._pairs_collected = sum(1 for _ in f)
        logger.info("[SPIN] Cell online — %d pairs accumulated", self._pairs_collected)

    async def process(self, ctx: CellContext) -> Any:
        return {
            "status":           "spin ready",
            "pairs_collected":  self._pairs_collected,
            "rounds_triggered": self._rounds_triggered,
        }

    def collect_spin_pair(self, prompt: str, context: str = "") -> dict:
        """
        Generate a (current_model, previous_model) response pair for a prompt.
        Stores to spin_pairs.jsonl.
        """
        full_prompt = f"{context}\n\n{prompt}".strip() if context else prompt

        logger.info("[SPIN] Collecting pair for: %s", prompt[:60])

        # Current model response (normal system prompt)
        current_response = _call_vllm(full_prompt, _CURRENT_SYSTEM, max_tokens=400)

        # Previous model response (degraded system prompt — simulates weaker checkpoint)
        previous_response = _call_vllm(full_prompt, _PREVIOUS_SYSTEM, max_tokens=400)

        if not current_response or not previous_response:
            return {"ok": False, "error": "vLLM unavailable or returned empty response"}

        pair = {
            "prompt":            prompt,
            "context":           context,
            "current_response":  current_response,
            "previous_response": previous_response,
            "timestamp":         time.time(),
            "iteration":         self._rounds_triggered,
        }

        try:
            with open(_PAIRS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            self._pairs_collected += 1
        except Exception as e:
            return {"ok": False, "error": str(e)}

        return {
            "ok":               True,
            "pairs_collected":  self._pairs_collected,
            "prompt_preview":   prompt[:80],
            "current_preview":  current_response[:100],
            "previous_preview": previous_response[:100],
        }

    def trigger_spin_round(self) -> dict:
        """
        Format accumulated pairs as ORPO training data.
        Writes to datasets/spin_pairs.jsonl.
        Logs readiness for finetune script.
        """
        if not _PAIRS_FILE.exists():
            return {"ok": False, "error": "No spin_pairs.jsonl found — collect pairs first"}

        pairs = []
        try:
            with open(_PAIRS_FILE, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pairs.append(json.loads(line))
        except Exception as e:
            return {"ok": False, "error": f"Failed to read pairs: {e}"}

        if not pairs:
            return {"ok": False, "error": "No pairs in file"}

        # Format as ORPO pairs
        # ORPO: each item has prompt + chosen (winning) + rejected (losing)
        orpo_pairs = []
        for p in pairs:
            orpo_pairs.append({
                "prompt":   p["prompt"],
                "chosen":   p["current_response"],    # current model = closer to human
                "rejected": p["previous_response"],   # previous model = to be moved away from
                "source":   "spin_self_play",
                "ts":       p.get("timestamp", 0),
            })

        # Write dataset
        try:
            _DATASET_DIR.mkdir(parents=True, exist_ok=True)
            with open(_DATASET_FILE, "w", encoding="utf-8") as f:
                for item in orpo_pairs:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception as e:
            return {"ok": False, "error": f"Failed to write dataset: {e}"}

        self._rounds_triggered += 1
        logger.info(
            "[SPIN] Round %d ready — %d ORPO pairs written to %s. "
            "Run finetune to apply.",
            self._rounds_triggered, len(orpo_pairs), _DATASET_FILE,
        )

        return {
            "ok":             True,
            "round":          self._rounds_triggered,
            "orpo_pairs":     len(orpo_pairs),
            "dataset_path":   str(_DATASET_FILE),
            "message":        f"SPIN round {self._rounds_triggered} ready — run finetune to apply",
        }

    def health(self) -> dict:
        return {
            "status":           self._status.value,
            "pairs_collected":  self._pairs_collected,
            "rounds_triggered": self._rounds_triggered,
            "pairs_file":       str(_PAIRS_FILE),
        }
