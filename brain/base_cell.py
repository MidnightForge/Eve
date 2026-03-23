"""
BaseCell — abstract contract every honeycomb cell must implement.

A cell is a self-contained unit of Eve's intelligence. It registers
with the brain manager, declares what message types it handles, and
processes requests independently. Cells can call each other through
the manager (inter-cell communication bus) but should never import
each other directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json
import os
import time
import logging
import threading

logger = logging.getLogger(__name__)

# ── Cell Interaction Logger ─────────────────────────────────────────
# Records successful outputs from agentic cells to brain_cell_interactions.jsonl
# The Eve brain factory data harvester reads this file to generate ORPO training pairs.

_INTERACTION_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "brain_cell_interactions.jsonl"
)
_LOG_LOCK = threading.Lock()
_LOG_CELLS = {"planner", "guardian", "rag", "agent", "summarizer", "observability", "persona"}
_MAX_LOG_BYTES = 50 * 1024 * 1024  # 50MB cap — rotate when exceeded


def _log_cell_interaction(
    cell_name: str,
    input_text: str,
    output_text: str,
    quality_score: float = 0.80,
    technique: str = "",
) -> None:
    """Append a successful cell interaction to the training data log.
    Only logs the 7 new agentic cells — not core cells like cortex/memory/vision.
    Non-blocking: called from async context via threading."""
    if cell_name not in _LOG_CELLS:
        return
    if not input_text or not output_text:
        return
    # Only log outputs that look substantive (>50 chars)
    if len(str(output_text)) < 50:
        return

    record = {
        "cell":         cell_name,
        "input_text":   str(input_text)[:1000],   # cap to avoid huge log entries
        "output_text":  str(output_text)[:2000],
        "quality_score": quality_score,
        "technique":    technique,
        "ts":           time.time(),
    }

    def _write():
        try:
            with _LOG_LOCK:
                # Rotate if too large
                try:
                    if os.path.getsize(_INTERACTION_LOG) > _MAX_LOG_BYTES:
                        bak = _INTERACTION_LOG + ".bak"
                        if os.path.exists(bak):
                            os.remove(bak)
                        os.rename(_INTERACTION_LOG, bak)
                except OSError:
                    pass
                with open(_INTERACTION_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass  # never crash the cell for logging

    threading.Thread(target=_write, daemon=True).start()


class CellStatus(str, Enum):
    DORMANT   = "dormant"    # not yet initialized
    BOOTING   = "booting"    # initializing
    ACTIVE    = "active"     # online and healthy
    BUSY      = "busy"       # processing a request
    DEGRADED  = "degraded"   # partially functional
    OFFLINE   = "offline"    # failed / unreachable


@dataclass
class CellContext:
    """Shared context passed between cells during a single request cycle."""
    message:       str
    user_id:       int             = 1
    voice_mode:    bool            = False
    is_complex:    bool            = False
    tone_hint:     Optional[str]   = None
    # Outputs written by each cell — other cells can read these
    memory_injection: str          = ""
    emotion_state:    str          = "neutral"
    iris_context:     str          = ""
    tool_results:     list         = field(default_factory=list)
    active_cells:     list         = field(default_factory=list)
    metadata:         dict         = field(default_factory=dict)
    # GWT Global Workspace broadcast — set by QuantumMeshCell each turn
    # {cell: name, content: str, turn: int}
    gws_broadcast:    Optional[dict] = None


@dataclass
class CellResult:
    """Return value from a cell's process() call."""
    cell_name:  str
    success:    bool
    output:     Any            = None
    error:      str            = ""
    duration_ms: float         = 0.0


class BaseCell(ABC):
    """
    Every cell must subclass this. Required overrides: name, description, process().
    Optional overrides: boot(), shutdown(), health().
    """

    #: Unique identifier — used as the cell's key in the brain registry
    name:        str = "base"
    #: Human-readable description shown in the honeycomb UI
    description: str = "Base cell"
    #: Hex color for the honeycomb visualization
    color:       str = "#7c3aed"
    #: If True, cell only initializes on first use (not on brain startup)
    lazy:        bool = True
    #: Honeycomb grid position [col, row] for UI layout
    position:    tuple = (0, 0)

    # ── Agentic AI Framework Metadata ─────────────────────────────────────
    #: "online"        — active now on RTX 4090
    #: "future_system" — coded & ready, activates when RTX PRO 5000 72GB arrives
    #: "dormant"       — planned, not yet coded
    system_tier:     str = "online"
    #: Hardware / service requirements
    hardware_req:    str = "RTX 4090 24GB"
    #: Research papers and techniques implemented
    research_basis:  str = ""
    #: Current build state and expansion roadmap (shown in UI on click)
    build_notes:     str = ""
    #: Agentic AI framework layer (AI&ML / Deep Learning / Gen AI / AI Agents / Agentic AI)
    framework_layer: str = "AI & ML"

    def __init__(self):
        self._status    = CellStatus.DORMANT
        self._boot_time = None
        self._call_count = 0
        self._error_count = 0
        self._last_duration_ms = 0.0
        self._manager   = None   # injected by HoneycombBrainManager
        self._health_cache: dict = {}    # updated by background thread, never blocking
        self._health_thread: Optional[threading.Thread] = None
        self._start_health_updater()

    def _start_health_updater(self):
        """Start a daemon thread that refreshes _health_cache every 10s."""
        def _loop():
            while True:
                try:
                    self._health_cache = self.health()
                except Exception:
                    pass
                time.sleep(10)
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        self._health_thread = t

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def _boot(self) -> None:
        """Called once by the manager before first use."""
        if self._status in (CellStatus.ACTIVE, CellStatus.BOOTING):
            return
        self._status = CellStatus.BOOTING
        try:
            await self.boot()
            self._status    = CellStatus.ACTIVE
            self._boot_time = time.time()
            logger.info("[Brain:%s] Cell online", self.name)
        except Exception as exc:
            self._status = CellStatus.OFFLINE
            logger.error("[Brain:%s] Boot failed: %s", self.name, exc)
            raise

    async def boot(self) -> None:
        """Override to perform async initialization (connect to services, load models, etc.)."""
        pass

    async def shutdown(self) -> None:
        """Override to cleanly release resources."""
        self._status = CellStatus.DORMANT

    def health(self) -> dict:
        """Override to return richer health info (port checks, etc.)."""
        return {"status": self._status.value}

    # ── Processing ─────────────────────────────────────────────────────────

    async def _run(self, ctx: CellContext) -> CellResult:
        """Called by manager — wraps process() with timing + error handling."""
        if self._status == CellStatus.DORMANT:
            await self._boot()

        t0 = time.perf_counter()
        prev_status = self._status
        self._status = CellStatus.BUSY
        try:
            output = await self.process(ctx)
            self._call_count += 1
            duration = (time.perf_counter() - t0) * 1000
            self._last_duration_ms = duration
            self._status = CellStatus.ACTIVE

            # ── Training data log ────────────────────────────────────────
            # Auto-record agentic cell outputs for ORPO pair harvesting.
            # Only logs the 7 new cells (planner, guardian, rag, agent, etc.)
            if self.name in _LOG_CELLS and output:
                output_str = (
                    json.dumps(output, ensure_ascii=False)
                    if isinstance(output, dict) else str(output)
                )
                _log_cell_interaction(
                    cell_name=self.name,
                    input_text=ctx.message,
                    output_text=output_str,
                    quality_score=0.80,  # baseline; ObservabilityCell upgrades this
                    technique=getattr(self, "research_basis", "")[:80],
                )

            return CellResult(cell_name=self.name, success=True, output=output, duration_ms=duration)
        except Exception as exc:
            self._error_count += 1
            self._status = CellStatus.DEGRADED
            logger.warning("[Brain:%s] process() error: %s", self.name, exc)
            return CellResult(cell_name=self.name, success=False, error=str(exc),
                              duration_ms=(time.perf_counter() - t0) * 1000)

    @abstractmethod
    async def process(self, ctx: CellContext) -> Any:
        """
        Main cell logic. Read ctx for inputs, return your output.
        Do NOT store request-specific state — cells must be reentrant.
        """
        raise NotImplementedError

    # ── Status ─────────────────────────────────────────────────────────────

    def status_dict(self) -> dict:
        return {
            "name":            self.name,
            "description":     self.description,
            "color":           self.color,
            "position":        list(self.position),
            "status":          self._status.value,
            "lazy":            self.lazy,
            "calls":           self._call_count,
            "errors":          self._error_count,
            "last_ms":         round(self._last_duration_ms, 1),
            "uptime_s":        round(time.time() - self._boot_time, 0) if self._boot_time else 0,
            # Agentic AI framework metadata
            "system_tier":     self.system_tier,
            "hardware_req":    self.hardware_req,
            "research_basis":  self.research_basis,
            "build_notes":     self.build_notes,
            "framework_layer": self.framework_layer,
            **self._health_cache,
        }
