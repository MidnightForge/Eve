"""
ReasoningCell — Eve's Trained Soul (vLLM / Qwen3-14B).

This is Eve's fine-tuned personality model. It is:
  - LAZY: only initializes when first needed (no blocking at startup)
  - OPTIONAL: if offline, Cortex handles the response directly
  - ENHANCEMENT: when online, it can style-transfer Cortex responses
    through Eve's trained voice, or handle complex reasoning tasks

The manager checks ReasoningCell.is_ready() before routing to it.
Eve always responds even when this cell is offline.
"""

import os
import logging
import asyncio
import subprocess
import requests as _req

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

VLLM_URL   = os.getenv("VLLM_URL",   "http://localhost:8099/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "eve")

# Minimum free VRAM (GB) before we consider vLLM launchable
_MIN_VRAM_GB = 13.0


def _vram_free_gb() -> float:
    """Read free VRAM from nvidia-smi. Returns 99.0 on failure (safe fallback)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=5, text=True
        ).strip()
        return float(out.split()[0]) / 1024.0
    except Exception:
        return 99.0


class ReasoningCell(BaseCell):
    name        = "reasoning"
    description = "Deep Reasoning — Qwen3-14B trained soul (lazy)"
    color       = "#b91c1c"
    lazy        = True
    position    = (2, 2)    # bottom-center

    def __init__(self):
        super().__init__()
        self._vllm_ready  = False
        self._vram_free   = 99.0   # last known free VRAM in GB
        self._check_task  = None
        self._consec_down = 0      # consecutive health-check failures

    async def boot(self) -> None:
        """Non-blocking boot — start health + VRAM monitor loop."""
        self._check_task = asyncio.create_task(self._health_loop())
        self._status = CellStatus.DEGRADED
        logger.info("[ReasoningCell] vLLM health monitor started")

    async def _health_loop(self) -> None:
        """
        Every 15s:
          1. Check free VRAM and store it.
          2. Ping vLLM /health.
          3. If vLLM has been down long enough AND VRAM is sufficient,
             log a warning — the healing factory will handle the actual restart.
        """
        health_url = VLLM_URL.replace("/v1", "") + "/health"
        while True:
            try:
                # VRAM check (non-blocking via executor)
                self._vram_free = await asyncio.get_event_loop().run_in_executor(
                    None, _vram_free_gb
                )

                # vLLM health ping
                r = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: _req.get(health_url, timeout=4)
                )
                was_ready = self._vllm_ready
                self._vllm_ready  = (r.status_code == 200)
                self._consec_down = 0

                if self._vllm_ready and not was_ready:
                    self._status = CellStatus.ACTIVE
                    logger.info(
                        "[ReasoningCell] vLLM ONLINE — Qwen3-14B ready | "
                        "VRAM free: %.1f GB", self._vram_free
                    )
                elif not self._vllm_ready and was_ready:
                    self._status = CellStatus.DEGRADED
                    logger.warning("[ReasoningCell] vLLM went offline")

            except Exception:
                self._vllm_ready   = False
                self._consec_down += 1
                if self._status == CellStatus.ACTIVE:
                    self._status = CellStatus.DEGRADED

                # After 4 consecutive failures (~60s), log VRAM state for diagnostics
                if self._consec_down == 4:
                    logger.warning(
                        "[ReasoningCell] vLLM down 4 checks (~60s). "
                        "Free VRAM: %.1f GB (need ≥%.1f GB). "
                        "Healing factory will restart when safe.",
                        self._vram_free, _MIN_VRAM_GB
                    )

            await asyncio.sleep(15)

    def is_ready(self) -> bool:
        return self._vllm_ready and self._status == CellStatus.ACTIVE

    def vram_ok_for_launch(self) -> bool:
        """True if VRAM is likely sufficient for vLLM to start."""
        return self._vram_free >= _MIN_VRAM_GB

    async def process(self, ctx: CellContext):
        """Returns None if vLLM not ready — caller falls back to Cortex."""
        if not self.is_ready():
            return None
        return True   # signals "route through me"

    def get_client(self):
        """Return an OpenAI-compatible client pointed at vLLM."""
        from openai import OpenAI
        return OpenAI(base_url=VLLM_URL, api_key="EMPTY")

    def health(self) -> dict:
        return {
            "vllm_url":      VLLM_URL,
            "vllm_ready":    self._vllm_ready,
            "vram_free_gb":  round(self._vram_free, 1),
            "vram_ok":       self.vram_ok_for_launch(),
            "consec_down":   self._consec_down,
        }
