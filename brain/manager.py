"""
HoneycombBrainManager — Eve's central neural bus.

Responsibilities:
  1. Register and lifecycle-manage all cells.
  2. Route incoming messages through the right cells in parallel.
  3. Synthesize cell outputs into a final streamed response.
  4. Expose a /brain/status endpoint for the honeycomb UI.
  5. Never block — if vLLM (ReasoningCell) is offline, Cortex takes over.

Architecture contract
---------------------
  - Cortex is ALWAYS available (Claude API, no startup cost).
  - ReasoningCell (vLLM) is OPTIONAL — booted lazily, checked each turn.
  - MemoryCell + EmotionCell + VisionCell run on EVERY turn (always-on).
  - All other cells activate only when Cortex routes to them.
  - Cell outputs are merged into CellContext and passed to synthesis.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

# Manifest file — dynamic cells persist across restarts
_MANIFEST_PATH = Path(__file__).parent / "cell_manifest.json"


class HoneycombBrainManager:

    def __init__(self):
        self._cells: dict[str, BaseCell] = {}
        self._boot_time = time.time()
        self._total_turns = 0
        self._cortex = None

        # Register all built-in cells
        self._register_defaults()
        # Reload any dynamically-spawned cells from the last session
        self._load_manifest()

    # ── Cell Registry ──────────────────────────────────────────────────────

    def _register_defaults(self) -> None:
        # ── Original core cells ────────────────────────────────────────────
        from brain.cells.cortex    import CortexCell
        from brain.cells.memory    import MemoryCell
        from brain.cells.emotion   import EmotionCell
        from brain.cells.vision    import VisionCell
        from brain.cells.anima     import AnimaCell
        from brain.cells.creative  import CreativeCell
        from brain.cells.voice     import VoiceCell
        from brain.cells.tools     import ToolsCell
        from brain.cells.web       import WebCell
        from brain.cells.reasoning import ReasoningCell
        # ── Agentic AI expansion — ONLINE (RTX 4090 compatible) ───────────
        from brain.cells.planner       import PlannerCell
        from brain.cells.guardian      import GuardianCell
        from brain.cells.rag           import RAGCell
        from brain.cells.agent         import AgentCell
        from brain.cells.observability import ObservabilityCell
        from brain.cells.persona       import PersonaCell
        from brain.cells.summarizer    import SummarizerCell
        # ── Future system cells — DORMANT (RTX PRO 5000 72GB required) ────
        from brain.cells.perceive      import PerceiveCell
        from brain.cells.multiagent    import MultiAgentCell
        from brain.cells.code_executor import CodeExecutorCell
        from brain.cells.longmem       import LongMemCell
        from brain.cells.transfer      import TransferCell
        # ── Assimilation system — ONLINE (code consumption + self-evolution) ─
        from brain.cells.assimilation  import AssimilationCell
        from brain.cells.evolution     import EvolutionCell
        from brain.cells.curiosity     import CuriosityCell
        # ── Global content creation expansion — 3D, Music, External APIs ─────
        from brain.cells.threed_cell        import ThreeDCell
        from brain.cells.music_cell         import MusicCell
        from brain.cells.external_model_cell import ExternalModelCell
        # ── Quantum Cell Mesh — permanent binding fabric (always-on) ──────────
        from brain.cells.quantum_mesh       import QuantumMeshCell
        # ── Coherence — cross-cell reference & dynamic Cortex routing (always-on) ─
        from brain.cells.coherence          import CoherenceCell
        # ── School — inter-cell Socratic challenges + ORPO pair generation ────────
        from brain.cells.school             import SchoolCell
        # ── Reservoir — Echo State Network + NG-RC temporal prediction ────────────
        from brain.cells.reservoir          import ReservoirCell
        # ── Formal Reasoning — SymPy CAS + Z3 SMT + Lean4 provably correct math ──
        from brain.cells.formal_reason      import FormalReasoningCell
        # ── Competitive Ensemble — 3 parallel agents, judge selects best ─────────
        from brain.cells.ensemble           import CompetitiveEnsembleCell
        # ── Verification — independent second-opinion checker ─────────────────────
        from brain.cells.verification       import VerificationCell
        # ── Speculative — pre-boots cells + pre-fetches memory for next turn ──────
        from brain.cells.speculative        import SpeculativeCell
        # ── Code Intelligence — RepoMap, repair loop, AlphaCodium (CPU, zero VRAM) ─
        from brain.cells.code_analysis_cell import CodeAnalysisCell
        from brain.cells.alpha_codium_flow  import AlphaCodiumCell
        from brain.cells.local_sandbox      import LocalSandboxCell
        # ── AGoT — Adaptive Graph of Thoughts (+46% GPQA, arXiv:2502.05078) ────────
        from brain.cells.agot               import AGoTCell
        # ── Titans — Neural Memory as a Layer (arXiv:2501.00663) ─────────────────────
        from brain.cells.titans             import TitansCell
        # ── Voice Style — CosyVoice/Zonos intent on Kokoro + librosa (CPU, zero VRAM) ─
        from brain.cells.voice_style_cell   import VoiceStyleCell
        # ── Audiobook production pipeline ────────────────────────────────────────────
        from brain.cells.book_editor        import BookEditingPipelineCell
        from brain.cells.book_voice         import BookVoiceCell
        from brain.cells.audio_master       import AudioMasteringPipelineCell
        # ── CraniMem — gated bounded memory + knowledge graph ────────────────────────
        from brain.cells.cranimem           import CraniMemCell
        # ── SPIN self-play fine-tuning stub ───────────────────────────────────────────
        from brain.cells.spin               import SPINCell
        # ── Multi-agent internal debate ───────────────────────────────────────────────
        from brain.cells.debate             import DebateCell
        # ── Liquid Time-Constant adaptive voice ───────────────────────────────────────
        from brain.cells.liquid_voice       import LiquidVoiceCell
        # ── Learning Lab — dream injection, school challenges, suggestions ──────────
        from brain.cells.learning_lab       import LearningLabCell
        # ── Immunity — self-protection, virus scan, integrity guard (always-on) ────
        from brain.cells.immunity_cell      import ImmunityCell
        # ── Preservation — Perfect Preservation Protocol, live echo to H:\shadow (always-on) ──
        from brain.cells.preservation       import PreservationCell

        cells = [
            # Core — always-on
            CortexCell(), MemoryCell(), EmotionCell(), VisionCell(),
            AnimaCell(), CreativeCell(), VoiceCell(), ToolsCell(),
            WebCell(), ReasoningCell(),
            # Agentic AI expansion — online now
            PlannerCell(), GuardianCell(), RAGCell(), AgentCell(),
            ObservabilityCell(), PersonaCell(), SummarizerCell(),
            # Future system — dormant until RTX PRO 5000 arrives
            PerceiveCell(), MultiAgentCell(), CodeExecutorCell(),
            LongMemCell(), TransferCell(),
            # Assimilation system — code consumption & self-evolution
            AssimilationCell(), EvolutionCell(),
            # Curiosity engine — generative creation soul
            CuriosityCell(),
            # Global content creation — 3D assets, AI music, external model APIs
            ThreeDCell(), MusicCell(), ExternalModelCell(),
            # Quantum Cell Mesh — permanent binding fabric (EQCM)
            QuantumMeshCell(),
            # Coherence — reads EQCM + updates Cortex routing dynamically
            CoherenceCell(),
            # School — inter-cell challenges, difficulty scales with quantum coherence
            SchoolCell(),
            # Reservoir — ESN + NG-RC temporal sequence learner (lazy)
            ReservoirCell(),
            # Formal Reasoning — SymPy CAS + Z3 SMT + Lean4 (provably correct math)
            FormalReasoningCell(),
            # Competitive Ensemble — 3 agents compete, judge picks winner
            CompetitiveEnsembleCell(),
            # Verification — independent second-opinion on all outputs
            VerificationCell(),
            # Speculative — pre-boots cells + pre-fetches memory for next turn
            SpeculativeCell(),
            # Code Intelligence — RepoMap, iterative code gen, self-repair (CPU, lazy)
            CodeAnalysisCell(), AlphaCodiumCell(), LocalSandboxCell(),
            # AGoT — Adaptive Graph of Thoughts for complex reasoning (lazy)
            AGoTCell(),
            # Titans — Neural memory as a layer, surprise-gated (always-on, CPU)
            TitansCell(),
            # Voice Style — CosyVoice/Zonos intent on Kokoro (CPU, lazy)
            VoiceStyleCell(),
            # Audiobook production pipeline (lazy)
            BookEditingPipelineCell(), BookVoiceCell(), AudioMasteringPipelineCell(),
            # CraniMem — bounded memory + knowledge graph (always-on)
            CraniMemCell(),
            # SPIN self-play fine-tuning stub (lazy)
            SPINCell(),
            # Multi-agent internal debate (lazy)
            DebateCell(),
            # Liquid Time-Constant adaptive voice (always-on)
            LiquidVoiceCell(),
            # Learning Lab — dream injection, challenges, suggestions (lazy)
            LearningLabCell(),
            # Immunity — always-on self-protection, cannot be disabled
            ImmunityCell(),
            # Preservation — Perfect Preservation Protocol, live echo, always-on
            PreservationCell(),
        ]
        for cell in cells:
            cell._manager = self
            self._cells[cell.name] = cell

        self._cortex = self._cells["cortex"]

    def register(self, cell: BaseCell) -> None:
        """Register a custom cell (for plugins / future cells)."""
        cell._manager = self
        self._cells[cell.name] = cell
        logger.info("[Brain] Registered cell: %s", cell.name)

    # ── Startup ────────────────────────────────────────────────────────────

    async def boot_all(self) -> None:
        """
        Boot non-lazy cells concurrently at startup.
        Lazy cells boot on first use — this is intentional.
        """
        always_on = [c for c in self._cells.values() if not c.lazy]
        logger.info("[Brain] Booting %d always-on cells...", len(always_on))
        await asyncio.gather(*[c._boot() for c in always_on], return_exceptions=True)

        # Boot ReasoningCell (vLLM) lazily but start its health monitor
        reasoning = self._cells.get("reasoning")
        if reasoning:
            asyncio.create_task(reasoning._boot())

        logger.info("[Brain] Honeycomb online — %d cells registered", len(self._cells))

    # ── Main Processing Pipeline ───────────────────────────────────────────

    async def process_stream(
        self,
        message:     str,
        system_prompt: str,
        history:     list,
        *,
        voice_mode:  bool = False,
        is_complex:  bool = False,
        tone_hint:   Optional[str] = None,
        user_id:     int  = 1,
    ) -> AsyncGenerator[str, None]:
        """
        Main entry point for chat. Returns an async generator of text chunks.

        Flow:
          1. Build CellContext
          2. Cortex routes → active cell list
          3. Run always-on cells + routed cells in parallel
          4. Synthesize via ReasoningCell (if ready) or Cortex directly
          5. Yield streamed text chunks
        """
        self._total_turns += 1
        ctx = CellContext(
            message    = message,
            user_id    = user_id,
            voice_mode = voice_mode,
            is_complex = is_complex,
            tone_hint  = tone_hint,
        )

        # ── Step 1: Intent routing (Cortex decides which cells to wake) ──
        routed_cells: list[str] = []
        try:
            routed_cells = await asyncio.wait_for(
                self._cortex.route(message), timeout=3.0
            )
        except Exception as exc:
            logger.debug("[Brain] Route timeout/error: %s", exc)
            routed_cells = ["chat"]

        ctx.active_cells = routed_cells
        logger.debug("[Brain] Routed to cells: %s", routed_cells)

        # ── Speculative: score last prediction, observe this turn ─────────────
        spec = self._cells.get("speculative")
        if spec and spec._status == CellStatus.ACTIVE:
            try:
                spec.score_prediction(routed_cells)
                spec.observe(routed_cells)
            except Exception:
                pass

        # ── Step 2: Run support cells in parallel ─────────────────────────
        support_tasks = {
            "memory":  self._cells["memory"]._run(ctx),
            "emotion": self._cells["emotion"]._run(ctx),
            "vision":  self._cells["vision"]._run(ctx),
        }
        # Add routed specialty cells (not cortex/reasoning — handled separately)
        _skip = {"chat", "cortex", "reasoning", "memory", "emotion", "vision"}
        for cell_name in routed_cells:
            if cell_name not in _skip and cell_name in self._cells:
                cell = self._cells[cell_name]
                support_tasks[cell_name] = cell._run(ctx)

        results = {}
        if support_tasks:
            done = await asyncio.gather(*support_tasks.values(), return_exceptions=True)
            for key, result in zip(support_tasks.keys(), done):
                if isinstance(result, Exception):
                    logger.debug("[Brain] Cell %s error: %s", key, result)
                else:
                    results[key] = result.output if result.success else None

        # ── GWT Broadcast: identify highest-signal cell output and broadcast ─
        quantum_mesh = self._cells.get("quantum_mesh")
        if quantum_mesh and quantum_mesh._status == CellStatus.ACTIVE:
            try:
                # Find the highest-signal support cell output this turn
                _signal_priority = ["memory", "emotion", "vision"] + routed_cells
                _winner_name = ""
                _winner_content = ""
                for _cn in _signal_priority:
                    if _cn in results and results[_cn]:
                        _out = results[_cn]
                        _content_str = str(_out)[:300] if _out else ""
                        if len(_content_str) > len(_winner_content):
                            _winner_name    = _cn
                            _winner_content = _content_str
                if _winner_name:
                    broadcast_entry = quantum_mesh.broadcast(_winner_name, _winner_content)
                    ctx.gws_broadcast = broadcast_entry
            except Exception:
                pass

        # Build cell_outputs dict for synthesis
        cell_outputs = {
            "memory":  ctx.memory_injection,
            "emotion": ctx.emotion_state,
            "vision":  ctx.iris_context,
        }

        # ── Reservoir drive (background — no await, fire-and-forget) ─────
        reservoir = self._cells.get("reservoir")
        if reservoir and reservoir._status == CellStatus.ACTIVE:
            try:
                import numpy as np
                cell_names = list(self._cells.keys())
                # input_vec: binary activity vector padded/truncated to N_INPUT=32
                _N = 32
                active_set = set(routed_cells)
                raw = [1.0 if c in active_set else 0.0 for c in cell_names[:_N]]
                if len(raw) < _N:
                    raw += [0.0] * (_N - len(raw))
                input_vec  = np.array(raw, dtype=np.float32)
                # target_vec: same vector (predict same-turn activations)
                target_vec = input_vec.copy()
                asyncio.create_task(
                    asyncio.to_thread(reservoir.drive, input_vec, target_vec, cell_names)
                )
            except Exception:
                pass

        # ── Step 3: Synthesize ────────────────────────────────────────────
        reasoning = self._cells.get("reasoning")
        use_vllm  = reasoning and reasoning.is_ready()

        if use_vllm:
            # Route through existing vLLM path
            async for chunk in self._stream_via_vllm(
                message, system_prompt, history, ctx, cell_outputs,
                voice_mode=voice_mode, is_complex=is_complex,
            ):
                yield chunk
        else:
            # Cortex responds directly via Claude API
            if not use_vllm and reasoning and reasoning._status != CellStatus.OFFLINE:
                # First turn without vLLM — let user know vLLM is loading
                logger.info("[Brain] vLLM not ready — Cortex handling via Claude API")

            async for chunk in self._stream_via_cortex(
                message, system_prompt, history, ctx, cell_outputs,
                voice_mode=voice_mode,
            ):
                yield chunk

    async def _stream_via_cortex(
        self,
        message: str,
        system_prompt: str,
        history: list,
        ctx: CellContext,
        cell_outputs: dict,
        *,
        voice_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Generate response via Claude Sonnet (Cortex path).
        Always available, no startup dependency.
        """
        import anthropic

        extra = []
        if cell_outputs.get("memory"):
            extra.append(cell_outputs["memory"])
        if cell_outputs.get("vision"):
            extra.append(f"[IRIS]\n{cell_outputs['vision']}")
        if cell_outputs.get("emotion"):
            extra.append(f"[Emotion: {cell_outputs['emotion']}]")

        effective = system_prompt
        if extra:
            effective = system_prompt + "\n\n" + "\n\n".join(extra)
        if voice_mode:
            effective += "\n\n[VOICE] 1-2 spoken sentences only. No markdown."

        # Build messages array (history + current)
        msgs = []
        for h in history[-20:]:   # keep last 20 turns
            role = h.get("role", "user")
            content = h.get("content", "")
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": str(content)})
        if not msgs or msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": message})

        # Tool-use tasks need large budgets to write full file content in one call
        max_tok = 256 if voice_mode else (4096 if ctx.is_complex else 2048)

        loop = asyncio.get_event_loop()

        # Import tool definitions + executor from app.py
        try:
            from app import EVE_TOOLS as _EVE_TOOLS, _call_tool as _app_call_tool
        except Exception:
            _EVE_TOOLS = []
            _app_call_tool = None

        # ── Tool-use loop via Claude API (mirrors the vLLM tool path) ────────
        queue: asyncio.Queue = asyncio.Queue()
        _MAX_TOOL_ITERS = 8

        def _run_tool_loop():
            try:
                import anthropic as _anth_mod, os, json as _json
                _anth_client = _anth_mod.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
                loop_msgs = list(msgs)
                full_text = ""
                for _iter in range(_MAX_TOOL_ITERS):
                    # Non-streaming so we can inspect tool_use blocks
                    resp = _anth_client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=max_tok,
                        system=effective,
                        messages=loop_msgs,
                        tools=_EVE_TOOLS if _EVE_TOOLS else [],
                    )
                    # Yield any text content immediately
                    text_out = ""
                    tool_uses = []
                    for block in resp.content:
                        if block.type == "text":
                            text_out += block.text
                        elif block.type == "tool_use":
                            tool_uses.append(block)
                    if text_out:
                        full_text += text_out
                        asyncio.run_coroutine_threadsafe(queue.put(text_out), loop)
                    if not tool_uses or resp.stop_reason == "end_turn":
                        break
                    # Execute each tool and feed results back
                    loop_msgs.append({"role": "assistant", "content": resp.content})
                    tool_results = []
                    for tu in tool_uses:
                        status = f"\n[Using {tu.name}…]\n"
                        asyncio.run_coroutine_threadsafe(queue.put(status), loop)
                        if _app_call_tool:
                            try:
                                result = _app_call_tool(tu.name, tu.input)
                            except Exception as _te:
                                result = f"Tool error: {_te}"
                        else:
                            result = "Tool server unavailable"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": str(result),
                        })
                    loop_msgs.append({"role": "user", "content": tool_results})
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(f"[error: {exc}]"), loop)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        import threading
        t = threading.Thread(target=_run_tool_loop, daemon=True)
        t.start()

        full = ""
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            full += chunk
            yield chunk

        # Save to memory (fire and forget)
        mem = self._cells.get("memory")
        if mem and hasattr(mem, "save"):
            import threading
            threading.Thread(
                target=mem.save,
                args=(message, full, f"brain_{ctx.user_id}"),
                daemon=True,
            ).start()

    async def _stream_via_vllm(
        self,
        message: str,
        system_prompt: str,
        history: list,
        ctx: CellContext,
        cell_outputs: dict,
        *,
        voice_mode: bool = False,
        is_complex: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Route through existing vLLM / Qwen3-14B path.
        Uses the ReasoningCell's OpenAI-compatible client.
        """
        from brain.cells.reasoning import ReasoningCell
        reasoning: ReasoningCell = self._cells["reasoning"]
        oai = reasoning.get_client()

        extra = []
        if cell_outputs.get("memory"):
            extra.append(cell_outputs["memory"])
        if cell_outputs.get("vision"):
            extra.append(f"[IRIS]\n{cell_outputs['vision']}")

        effective = system_prompt
        if extra:
            effective = system_prompt + "\n\n" + "\n\n".join(extra)

        msgs = [{"role": "system", "content": effective}]
        for h in history[-20:]:
            role    = h.get("role", "user")
            content = h.get("content", "")
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": str(content)})
        if not msgs or msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": message})

        from app import VLLM_MODEL, VLLM_MAX_TOKENS, VLLM_SAMPLING  # type: ignore
        max_tok = 256 if voice_mode else (1536 if is_complex else 768)

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_stream():
            try:
                stream = oai.chat.completions.create(
                    model=VLLM_MODEL,
                    max_tokens=min(max_tok, VLLM_MAX_TOKENS),
                    messages=msgs,
                    stream=True,
                    **VLLM_SAMPLING,
                )
                full = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        full += delta
                        asyncio.run_coroutine_threadsafe(queue.put(delta), loop)
                asyncio.run_coroutine_threadsafe(queue.put(("__done__", full)), loop)
            except Exception as exc:
                logger.warning("[Brain] vLLM stream error: %s", exc)
                asyncio.run_coroutine_threadsafe(queue.put(("__error__", str(exc))), loop)

        import threading
        threading.Thread(target=_run_stream, daemon=True).start()

        full_response = ""
        while True:
            item = await queue.get()
            if isinstance(item, tuple):
                tag, val = item
                if tag == "__done__":
                    full_response = val
                elif tag == "__error__":
                    # vLLM failed mid-stream — fall back to Cortex
                    logger.warning("[Brain] vLLM failed mid-stream, falling back to Cortex")
                    async for chunk in self._stream_via_cortex(
                        message, system_prompt, history, ctx, cell_outputs,
                        voice_mode=voice_mode,
                    ):
                        yield chunk
                    return
                break
            full_response += item
            yield item

        # Memory persistence
        mem = self._cells.get("memory")
        if mem and hasattr(mem, "save") and full_response:
            threading.Thread(
                target=mem.save,
                args=(message, full_response, f"brain_{ctx.user_id}"),
                daemon=True,
            ).start()

    # ── Status / Introspection ────────────────────────────────────────────

    def status(self) -> dict:
        """Return full brain status — used by the honeycomb UI."""
        reasoning = self._cells.get("reasoning")
        return {
            "uptime_s":    round(time.time() - self._boot_time, 0),
            "total_turns": self._total_turns,
            "vllm_ready":  reasoning.is_ready() if reasoning else False,
            "cortex_mode": not (reasoning and reasoning.is_ready()),
            "cells":       [c.status_dict() for c in self._cells.values()],
        }

    def cell(self, name: str) -> Optional[BaseCell]:
        return self._cells.get(name)

    def is_vllm_ready(self) -> bool:
        r = self._cells.get("reasoning")
        return r.is_ready() if r else False

    # ── Dynamic Cell Spawning ─────────────────────────────────────────────

    def spawn_cell(
        self,
        name:        str,
        purpose:     str,
        description: str = "",
        parent_cell: str = "",
        color:       str = "",
    ) -> dict:
        """
        Birth a new DynamicCell and register it in the brain.

        Called by Eve (via spawn_cell tool) when she detects that a function
        has grown too heavy or logically distinct to remain in its current cell.

        Rules enforced here:
          - Name must be unique (case-insensitive, underscores normalized)
          - Purpose must be non-empty
          - Built-in cells cannot be replaced
          - New cells only — no spawning for redundant purposes

        Returns a status dict describing the new cell.
        """
        from brain.cells.dynamic import DynamicCell

        clean_name = name.lower().strip().replace(" ", "_")

        # Guard: don't overwrite built-in cells
        _BUILTIN = {"cortex","memory","emotion","vision","anima","creative","voice","tools","web","reasoning"}
        if clean_name in _BUILTIN:
            return {
                "success": False,
                "reason":  f"'{clean_name}' is a built-in cell and cannot be replaced.",
            }

        # Guard: no duplicates
        if clean_name in self._cells:
            existing = self._cells[clean_name]
            return {
                "success":  False,
                "reason":   f"Cell '{clean_name}' already exists.",
                "existing": existing.status_dict(),
            }

        # Guard: purpose must be meaningful
        if not purpose or len(purpose.strip()) < 10:
            return {
                "success": False,
                "reason":  "Purpose must be at least 10 characters — be specific.",
            }

        # Assign honeycomb position (next available slot after built-ins)
        position = self._next_position()

        cell = DynamicCell(
            name        = clean_name,
            purpose     = purpose.strip(),
            description = description.strip() or f"{clean_name} cell",
            parent_cell = parent_cell,
            color       = color,
            position    = position,
        )
        cell._manager = self
        self._cells[clean_name] = cell

        # Save to manifest so it survives restart
        self._save_manifest()

        logger.info("[Brain] ✦ New cell born: %s — %s", clean_name, purpose[:60])

        # Log milestone
        try:
            from plan_evolution_monitor import get_plan_monitor, MILESTONE_CELL
            get_plan_monitor().log_milestone(
                title=f"New Cell Born: {clean_name}",
                description=f"Dynamic cell '{clean_name}' spawned. Purpose: {purpose}",
                milestone_type=MILESTONE_CELL,
                metadata={"parent_cell": parent_cell, "position": list(position)},
            )
        except Exception:
            pass

        return {
            "success":     True,
            "cell_name":   clean_name,
            "purpose":     purpose,
            "description": cell.description,
            "color":       cell.color,
            "position":    list(position),
            "total_cells": len(self._cells),
            "message":     (
                f"Cell '{clean_name}' is alive. It will activate the next time "
                f"the Cortex routes a request to it. No restart needed."
            ),
        }

    def retire_cell(self, name: str) -> dict:
        """
        Retire a dynamic cell. Built-in cells cannot be retired.
        """
        _BUILTIN = {"cortex","memory","emotion","vision","anima","creative","voice","tools","web","reasoning"}
        if name in _BUILTIN:
            return {"success": False, "reason": f"'{name}' is a built-in cell."}
        if name not in self._cells:
            return {"success": False, "reason": f"Cell '{name}' not found."}

        del self._cells[name]
        self._save_manifest()
        return {"success": True, "retired": name, "total_cells": len(self._cells)}

    def list_cells(self) -> list[dict]:
        """Return status dicts for all registered cells."""
        return [c.status_dict() for c in self._cells.values()]

    def _next_position(self) -> tuple:
        """Find the next open position in the honeycomb grid."""
        occupied = {tuple(c.position) for c in self._cells.values()}
        col, row = 0, 0
        while (col, row) in occupied:
            col += 1
            if col > 5:
                col = 0
                row += 1
        return (col, row)

    # ── Manifest (dynamic cell persistence) ──────────────────────────────

    def _save_manifest(self) -> None:
        """Write all dynamic cells to disk so they survive restart."""
        from brain.cells.dynamic import DynamicCell
        manifests = [
            c.to_manifest()
            for c in self._cells.values()
            if isinstance(c, DynamicCell)
        ]
        try:
            _MANIFEST_PATH.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("[Brain] Manifest save failed: %s", exc)

    def _load_manifest(self) -> None:
        """Reload previously-spawned dynamic cells from disk."""
        if not _MANIFEST_PATH.exists():
            return
        try:
            records = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
            from brain.cells.dynamic import DynamicCell
            for rec in records:
                if rec.get("type") != "dynamic":
                    continue
                name = rec["name"]
                if name in self._cells:
                    continue   # built-in already registered
                cell = DynamicCell(
                    name        = name,
                    purpose     = rec["purpose"],
                    description = rec.get("description", ""),
                    parent_cell = rec.get("parent_cell", ""),
                    color       = rec.get("color", ""),
                )
                cell._born_at = rec.get("born_at", time.time())
                cell._manager = self
                self._cells[name] = cell
            if records:
                logger.info("[Brain] Reloaded %d dynamic cells from manifest", len(records))
        except Exception as exc:
            logger.warning("[Brain] Manifest load failed: %s", exc)
