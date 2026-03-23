"""
CoherenceCell — Eve's Neural Cross-Reference & Routing Intelligence
====================================================================
The living bridge between the Quantum Cell Mesh (EQCM) and every cell in
Eve's brain. CoherenceCell closes the loop that the quantum mesh opens.

Problem it solves
-----------------
The quantum mesh publishes a binding state every 10 s — which cells are most
active, which are resonating, what the entanglement looks like. But nothing
was READING that state and acting on it. Meanwhile:

  • Cortex routes using a hardcoded INTENTS dict — dynamic cells are invisible
  • Assimilated programs in CapabilityVault never reach the routing layer
  • New cells spawned at runtime are filtered out at line 108 of cortex.py
  • No cell knows what any other cell can do

CoherenceCell fixes all of this. Every 30 s it:

  1. Reads get_mesh_binding() — live quantum state (who's dominant, coherence)
  2. Scans ALL cells from HoneycombBrainManager (built-in + dynamic)
  3. Reads CapabilityVault for assimilated programs
  4. Builds a live _COHERENCE_INDEX — full capability map of the whole brain
  5. Updates cortex._ROUTER_PROMPT with the complete dynamic cell catalog
     so the LLM routing sees EVERY cell including new ones
  6. Injects quantum binding weights — most-active cells get priority routing
  7. Publishes get_coherence_index() for any cell to read at any time

Result: Every time a new cell is born or a program is assimilated, within
30 s it is fully integrated into Cortex routing and the coherence index.
The quantum mesh + coherence cell together make Eve's brain truly one system.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

# ── Module-level coherence index (readable by any cell) ───────────────────────
_COHERENCE_INDEX: dict = {}          # cell_name → {description, active, calls, quantum_weight}
_COHERENCE_INSTANCE: Optional["CoherenceCell"] = None


def get_coherence_index() -> dict:
    """
    Public API. Any cell can call this to get the live cross-cell capability map.
    Returns {cell_name: {description, active, calls, vault_caps, quantum_weight}}
    """
    return dict(_COHERENCE_INDEX)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_routing_prompt(intents: dict, vault_caps: list, quantum_winner: str, coherence: float) -> str:
    """Build the dynamic _ROUTER_PROMPT string for Cortex."""
    cells_json = json.dumps(intents, ensure_ascii=False)

    vault_section = ""
    if vault_caps:
        names = [c.get("name", "") for c in vault_caps[:12]]
        vault_section = (
            f"\n\nAssimilated capabilities in CapabilityVault (route via 'assimilation' to invoke): "
            f"{', '.join(names)}"
        )

    quantum_section = ""
    if quantum_winner and coherence > 0.3:
        quantum_section = (
            f"\n\nQuantum mesh state: dominant active cell = '{quantum_winner}' "
            f"(coherence={coherence:.3f}). Prefer this cell for ambiguous routing."
        )

    return (
        "You are Eve's Cortex — her routing intelligence. Given a user message, "
        "identify which cells to activate (pick 1–4 relevant ones).\n\n"
        f"Available cells: {cells_json}"
        f"{vault_section}"
        f"{quantum_section}\n\n"
        "Respond with ONLY a JSON array of cell names, e.g.: [\"chat\",\"memory\"]\n"
        "Do not explain. Never return an empty array — always include \"chat\"."
    )


class CoherenceCell(BaseCell):
    """
    Always-on cross-reference coordinator. Reads from the Quantum Cell Mesh
    and keeps all brain cells, dynamic cells, and assimilated programs
    visible to Cortex routing and the rest of the brain.
    """

    name        = "coherence"
    description = ("Cross-cell coherence coordinator. Reads quantum mesh binding, "
                   "updates Cortex routing with ALL live cells + CapabilityVault, "
                   "keeps the entire brain cross-referenced and self-aware.")
    color       = "#10b981"   # emerald
    lazy        = False       # always-on — boots with the brain
    position    = (3, 5)

    system_tier     = "online"
    hardware_req    = "RTX 4090 — pure Python, no GPU needed"
    research_basis  = (
        "Dynamic routing via live cell manifest (manager._cells scan); "
        "CapabilityVault SQLite cross-reference; "
        "Quantum mesh GWT winner → routing priority injection; "
        "Runtime cortex._ROUTER_PROMPT monkey-patch for zero-restart cell discovery"
    )
    build_notes = (
        "LIVE: 30s coherence pulse. Reads EQCM binding → injects into Cortex routing. "
        "Dynamic cells + assimilated vault programs auto-appear in routing within 30 s. "
        "get_coherence_index() callable from anywhere. "
        "Brain is now truly one coherent system."
    )
    framework_layer = "Agentic AI"

    PULSE_S = 30.0   # coherence pulse interval

    def __init__(self):
        super().__init__()
        self._pulse_count = 0
        self._last_cell_count = 0
        self._last_vault_count = 0

    async def boot(self) -> None:
        global _COHERENCE_INSTANCE
        _COHERENCE_INSTANCE = self
        # First pulse after a short delay (let all cells finish booting)
        t = threading.Thread(target=self._pulse_loop, daemon=True, name="coherence-pulse")
        t.start()
        logger.info("[Coherence] Cross-reference coordinator online.")

    # ── Background pulse ───────────────────────────────────────────────────────

    def _pulse_loop(self) -> None:
        time.sleep(8.0)   # let quantum mesh run first (3s delay) + extra buffer
        while True:
            try:
                self._pulse()
            except Exception as exc:
                logger.debug("[Coherence] Pulse error: %s", exc)
            time.sleep(self.PULSE_S)

    def _pulse(self) -> None:
        manager = self._manager
        if manager is None:
            return

        # ── 1. Read quantum mesh binding ──────────────────────────────────────
        try:
            from brain.cells.quantum_mesh import get_mesh_binding
            binding = get_mesh_binding()
        except Exception:
            binding = {}

        quantum_winner  = binding.get("workspace", {}).get("winner", "")
        coherence       = binding.get("quantum", {}).get("coherence", 0.0)
        hopfield_res    = binding.get("hopfield", {}).get("resonance", {})

        # ── 2. Scan all registered cells ──────────────────────────────────────
        all_cells = dict(manager._cells)   # snapshot
        dynamic_intents: dict[str, str] = {}

        for name, cell in all_cells.items():
            try:
                sd = cell.status_dict()
            except Exception:
                sd = {}
            desc = sd.get("description") or cell.description or f"{name} cell"
            dynamic_intents[name] = desc[:120]

        # ── 3. Read CapabilityVault for assimilated programs ──────────────────
        vault_caps: list = []
        try:
            from capability_vault import get_vault
            vault_caps = get_vault().list_capabilities()
            if not isinstance(vault_caps, list):
                vault_caps = []
        except Exception:
            vault_caps = []

        # ── 4. Build coherence index ──────────────────────────────────────────
        global _COHERENCE_INDEX
        index: dict = {}

        for name, cell in all_cells.items():
            try:
                sd = cell.status_dict()
            except Exception:
                sd = {}
            q_weight = float(hopfield_res.get(name, 0.0))
            is_winner = (name == quantum_winner)
            index[name] = {
                "description":    sd.get("description", cell.description),
                "status":         sd.get("status", "dormant"),
                "calls":          sd.get("calls", 0),
                "quantum_weight": round(q_weight, 4),
                "is_dominant":    is_winner,
                "framework_layer": sd.get("framework_layer", ""),
            }

        # Add vault capabilities to index
        if vault_caps:
            index["_vault"] = {
                "description": f"CapabilityVault — {len(vault_caps)} assimilated programs",
                "capabilities": [c.get("name", "") if isinstance(c, dict) else str(c)
                                  for c in vault_caps[:20]],
                "status": "active",
                "calls": 0,
                "quantum_weight": 0.0,
                "is_dominant": False,
            }

        _COHERENCE_INDEX = index

        # ── 5. Update Cortex routing prompt ───────────────────────────────────
        try:
            import brain.cells.cortex as _cortex_mod

            # Update INTENTS with ALL live cells (not just hardcoded ones)
            for cell_name, desc in dynamic_intents.items():
                if cell_name not in _cortex_mod.INTENTS:
                    _cortex_mod.INTENTS[cell_name] = desc

            # Rebuild _ROUTER_PROMPT with quantum-weighted context
            vault_dicts = [c for c in vault_caps if isinstance(c, dict)] if vault_caps else []
            new_prompt = _build_routing_prompt(
                _cortex_mod.INTENTS,
                vault_dicts,
                quantum_winner,
                coherence,
            )
            _cortex_mod._ROUTER_PROMPT = new_prompt

        except Exception as _ce:
            logger.debug("[Coherence] Cortex update error: %s", _ce)

        # ── 6. Log deltas (new cells or vault entries) ─────────────────────────
        self._pulse_count += 1
        n_cells = len(all_cells)
        n_vault = len(vault_caps)

        if n_cells != self._last_cell_count:
            logger.info("[Coherence] Pulse #%d — cell count changed: %d → %d",
                        self._pulse_count, self._last_cell_count, n_cells)
            self._last_cell_count = n_cells

        if n_vault != self._last_vault_count:
            logger.info("[Coherence] Vault changed: %d → %d assimilated capabilities",
                        self._last_vault_count, n_vault)
            self._last_vault_count = n_vault

        logger.debug("[Coherence] Pulse #%d — %d cells, %d vault caps, winner=%s coherence=%.3f",
                     self._pulse_count, n_cells, n_vault, quantum_winner, coherence)

    # ── BaseCell.process ───────────────────────────────────────────────────────

    async def process(self, ctx: CellContext) -> Any:
        """Returns the current coherence index when Cortex routes here."""
        index = get_coherence_index()
        if not index:
            return "Coherence index building — first pulse in ~8 s."

        active = [n for n, v in index.items() if v.get("status") == "active" and not n.startswith("_")]
        dominant = next((n for n, v in index.items() if v.get("is_dominant")), "")
        n_vault  = len(index.get("_vault", {}).get("capabilities", []))

        return {
            "total_cells":  len([k for k in index if not k.startswith("_")]),
            "active_cells": active,
            "dominant_cell": dominant,
            "vault_capabilities": n_vault,
            "pulse": self._pulse_count,
        }

    def health(self) -> dict:
        return {
            "status":       self._status.value,
            "pulses":       self._pulse_count,
            "cells_tracked": self._last_cell_count,
            "vault_caps":    self._last_vault_count,
        }
