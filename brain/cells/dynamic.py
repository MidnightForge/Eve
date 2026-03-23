"""
DynamicCell — a cell born at runtime, on demand.

When Eve or the system determines that a function has grown too heavy,
needs its own identity, or represents a logically distinct domain,
a new DynamicCell is spawned. It carries:

  - A unique name (its identity)
  - A purpose string (its sole function)
  - A system prompt derived from its purpose
  - A color for the honeycomb visualization
  - Optional: a parent cell name (what it was split from)

DynamicCells are intentionally lightweight:
  - Each uses Claude Haiku for speed (not Sonnet — they are specialists)
  - Each handles ONE logical pattern only
  - They never call each other directly — communication is through CellContext
  - They are always lazy (born dormant, activated on first use)
  - They persist in memory for the lifetime of the brain manager

New cells should ONLY be born when:
  1. An existing cell's function has become logically distinct enough to separate
  2. A new capability domain is needed that no existing cell covers
  3. Load or latency on an existing cell suggests it is doing too many things

Never spawn redundant cells. Eve's brain grows with purpose, not with noise.
"""

import os
import asyncio
import logging
import time
import hashlib
from typing import Optional

from openai import OpenAI

_VLLM_URL = "http://127.0.0.1:8099/v1"
_MODEL    = "eve"

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

# Palette for auto-assigning colors to new cells
_CELL_COLORS = [
    "#7c3aed", "#b91c1c", "#0891b2", "#16a34a", "#ea580c",
    "#ec4899", "#8b5cf6", "#f59e0b", "#0f766e", "#6366f1",
    "#a21caf", "#0284c7", "#65a30d", "#dc2626", "#d97706",
]


def _pick_color(name: str) -> str:
    """Deterministic color assignment from cell name."""
    idx = int(hashlib.md5(name.encode()).hexdigest(), 16) % len(_CELL_COLORS)
    return _CELL_COLORS[idx]


class DynamicCell(BaseCell):
    """
    A purpose-built cell spawned at runtime.

    Each DynamicCell is a specialized micro-AI — a Claude Haiku instance
    with a laser-focused system prompt. It processes only what its purpose
    describes and returns a concise, precise result.

    Cells stay fast by staying narrow.
    """

    def __init__(
        self,
        name:        str,
        purpose:     str,
        description: str  = "",
        parent_cell: str  = "",
        color:       str  = "",
        position:    tuple = None,
    ):
        super().__init__()
        self.name        = name.lower().replace(" ", "_")
        self.description = description or f"Dynamic: {purpose[:48]}"
        self._purpose    = purpose
        self._parent     = parent_cell
        self.color       = color or _pick_color(name)
        self.lazy        = True
        self.position    = position or (0, 0)   # manager assigns real position
        self._born_at    = time.time()
        self._client: Optional[OpenAI] = None

        # Build the specialized system prompt for this cell
        self._cell_system = (
            f"You are Eve's [{self.name}] cell — a precise, efficient specialist.\n\n"
            f"Your sole purpose: {self._purpose}\n\n"
            f"Rules you never break:\n"
            f"- Stay strictly within your purpose. Do not drift.\n"
            f"- Be concise. You are a cell, not the whole brain.\n"
            f"- Return only what is useful to the Cortex.\n"
            f"- Do not greet, explain yourself, or add filler.\n"
            f"- If the input does not relate to your purpose, return exactly: [out_of_scope]\n"
        )

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=_VLLM_URL, api_key="none")
        return self._client

    async def boot(self) -> None:
        logger.info("[Brain] New cell born: %s — purpose: %s", self.name, self._purpose[:60])

    async def process(self, ctx: CellContext) -> str:
        """
        Run this cell's specialized processing.
        Uses Claude Haiku for sub-200ms responses.
        """
        loop = asyncio.get_event_loop()

        def _call() -> str:
            resp = self._get_client().chat.completions.create(
                model=_MODEL,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": self._cell_system},
                    {"role": "user",   "content": ctx.message[:800]},
                ],
            )
            return resp.choices[0].message.content.strip()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _call), timeout=5.0
            )
            # If out of scope, return empty so the brain ignores it
            return "" if result == "[out_of_scope]" else result
        except Exception as exc:
            logger.warning("[Brain:%s] process error: %s", self.name, exc)
            return ""

    def to_manifest(self) -> dict:
        """Serialize this cell for persistence across restarts."""
        return {
            "type":        "dynamic",
            "name":        self.name,
            "purpose":     self._purpose,
            "description": self.description,
            "parent_cell": self._parent,
            "color":       self.color,
            "born_at":     self._born_at,
        }

    def health(self) -> dict:
        return {
            "purpose":    self._purpose[:80],
            "parent":     self._parent,
            "born_at":    self._born_at,
            "age_hours":  round((time.time() - self._born_at) / 3600, 1),
        }
