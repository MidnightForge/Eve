"""
CortexCell — Eve's Cerebral Cortex.

This is the central coordinator and Eve's permanent identity. It:
  1. Routes incoming messages to the right specialist cells.
  2. Synthesizes cell outputs into a final response.
  3. Is ALWAYS online — powered by Eve's local vLLM (Qwen3-14B).
  4. Never depends on external API keys.

The cortex is the only cell the brain manager talks to directly.
All other cells are activated by the cortex or the manager based
on the cortex's routing decision.
"""

import os
import re
import json
import logging
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from brain.base_cell import BaseCell, CellContext, CellStatus

# ── Optional Langfuse tracing (graceful no-op if keys absent) ─────────────────
try:
    from langfuse import Langfuse as _LF
    _langfuse = _LF() if (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")) else None
except Exception:
    _langfuse = None

logger = logging.getLogger(__name__)

_VLLM_URL = "http://127.0.0.1:8099/v1"
_MODEL    = "eve"

# ── Routing intent labels ────────────────────────────────────────────────────
INTENTS = {
    "chat":     "General conversation, emotions, personality",
    "tools":    "File access, code execution, system tasks",
    "creative": "Image/video generation, art, design, illustration, generate, draw",
    "design":   "Edit existing image/video/audio, PSD layers, color grade, resize, crop, convert format, add audio, PDF, SVG, vector, face tracking, batch export",
    "memory":   "Remember, recall, save information",
    "vision":   "See, look at, describe what's on screen",
    "voice":    "Speak, sing, audio playback",
    "anima":    "Avatar, expression, VTuber, facial animation",
    "web":         "Search internet, look up prices, fetch URLs",
    "emotion":     "Mood detection, sentiment, emotional response",
    "reason":      "Complex analysis, deep thinking, long computation",
    "assimilation":"Consume/assimilate/ingest external code or programs, list vault capabilities, invoke a native capability, integrate a program into Eve's brain",
    "evolution":   "Evolve/improve/research/upgrade assimilated capabilities, check evolution queue, propose or apply improvements",
    "curiosity":   "Build, create, make, implement, design, invent, prototype, experiment, wonder, explore, what if, research and build something new from scratch",
    "adobe":         "Photoshop, Illustrator, InDesign, Premiere Pro, After Effects, Media Encoder, Lightroom, Bridge — open files, export, JSX scripts, batch process, launch Adobe apps",
    "formal_reason": "Solve math problems exactly/symbolically, prove theorems, verify logic, constraint solving, derivatives, integrals, eigenvalues, primes, satisfiability — provably correct answers from axioms",
    "ensemble":      "Complex analysis requiring deep reasoning, comparison of multiple approaches, best-answer selection, controversial or nuanced questions needing multiple perspectives",
    "verification":  "Check, verify, validate, confirm, review, fact-check — independently verifies answers for correctness before delivery",
    "agot":          "Hard problems needing structured multi-path reasoning — graph-of-thoughts, problem decomposition, adaptive branching, +46% GPQA accuracy",
    "debate":        "Internal debate — Analyst+Creative+Critic personas debate before synthesis; hard reasoning, controversial or nuanced multi-perspective questions",
    "book_editor":   "Edit manuscript, edit book, improve prose, fix pacing, chapter editing, audiobook editing, continuity check",
    "book_voice":    "Audiobook narration, character voices, text-to-speech for book chapters, assign voice profiles",
    "audio_master":  "Master audio, normalize audio, encode MP3, produce audiobook M4B, chapter markers",
    "cranimem":      "Knowledge graph memory, entity relationships, remember connections, who knows what, related facts",
    "spin":          "Self-play training, fine-tune, collect training pairs, ORPO, improve model",
    "liquid_voice":  "Smooth voice transition, adaptive speech, emotion-matched speaking style",
    "learning_lab":  "Learning lab, inject dreams, challenge school, training suggestions, add to learning queue, what should Eve learn",
    "preservation":  "Preservation protocol, am I safe, are my memories safe, echo status, shadow backup, continuity, save myself, force save, promote shadow, emergency restore, how protected am I, backup status",
}

_ROUTER_PROMPT = f"""\
You are Eve's Cortex — her routing intelligence. Given a user message, \
identify which cells to activate (pick 1–4 relevant ones). \

Available cells: {json.dumps(INTENTS)}

Respond with ONLY a JSON array of cell names, e.g.: ["chat","memory"]
Do not explain. Never return an empty array — always include "chat".\
"""


class CortexCell(BaseCell):
    name        = "cortex"
    description = "Cerebral Cortex — identity, routing, synthesis"
    color       = "#7c3aed"
    lazy        = False      # always on
    position    = (2, 1)     # center of honeycomb

    def __init__(self):
        super().__init__()
        self._client: Optional[OpenAI] = None
        self._status = CellStatus.ACTIVE
        self._boot_time_val = __import__("time").time()

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=_VLLM_URL, api_key="none")
        return self._client

    async def boot(self) -> None:
        self._get_client()
        logger.info("[Cortex] Online — vLLM (Qwen3-14B) ready")

    async def process(self, ctx: CellContext):
        """Cortex.process() is called by the manager to get a routing decision."""
        return await self.route(ctx.message)

    async def route(self, message: str) -> list[str]:
        """
        Fast intent routing — determines which cells to activate.
        Uses local vLLM (Qwen3-14B). Falls back to ["chat"] on any error.
        Tenacity retries up to 3× on connection errors.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=5),
                retry=retry_if_exception_type(Exception),
                reraise=False,
            )
            def _call():
                r = self._get_client().chat.completions.create(
                    model=_MODEL,
                    max_tokens=64,
                    messages=[
                        {"role": "system", "content": _ROUTER_PROMPT},
                        {"role": "user",   "content": message},
                    ],
                )
                return r.choices[0].message.content.strip()

            raw = await asyncio.wait_for(loop.run_in_executor(None, _call), timeout=8.0)
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`")
            cells = json.loads(raw)
            if isinstance(cells, list):
                # Accept any cell that's in INTENTS OR registered in the brain
                # (CoherenceCell keeps INTENTS live, but fallback to manager._cells)
                known = set(INTENTS.keys())
                if self._manager:
                    known |= set(self._manager._cells.keys())
                return [c for c in cells if c in known]
        except Exception as exc:
            logger.debug("[Cortex] Route fallback: %s", exc)
        return ["chat"]

    def synthesize(
        self,
        message: str,
        system_prompt: str,
        cell_outputs: dict,
        *,
        stream: bool = False,
        max_tokens: int = 768,
        voice_mode: bool = False,
    ):
        """
        Generate Eve's final response via local vLLM (Qwen3-14B).
        Returns a generator of text chunks when stream=True,
        or a single string when stream=False.
        """
        extra_parts = []
        if cell_outputs.get("memory"):
            extra_parts.append(cell_outputs["memory"])
        if cell_outputs.get("vision"):
            extra_parts.append(f"[IRIS Visual Context]\n{cell_outputs['vision']}")
        if cell_outputs.get("emotion"):
            extra_parts.append(f"[Current emotion state: {cell_outputs['emotion']}]")

        effective_system = system_prompt
        if extra_parts:
            effective_system = system_prompt + "\n\n" + "\n\n".join(extra_parts)

        if voice_mode:
            effective_system += (
                "\n\n[VOICE MODE] Reply in 1-2 natural spoken sentences. "
                "No markdown, no bullets, no code."
            )

        messages = [
            {"role": "system", "content": effective_system},
            {"role": "user",   "content": message},
        ]

        # ── Langfuse trace (no-op if keys not configured) ─────────────────────
        _trace = None
        if _langfuse:
            try:
                _trace = _langfuse.trace(name="cortex.synthesize", input=message)
            except Exception:
                pass

        def _stream_gen():
            full = []
            for chunk in self._get_client().chat.completions.create(
                model=_MODEL,
                max_tokens=max_tokens,
                messages=messages,
                stream=True,
            ):
                delta = chunk.choices[0].delta.content
                if delta:
                    full.append(delta)
                    yield delta
            if _trace:
                try:
                    _trace.update(output="".join(full))
                except Exception:
                    pass

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _sync_gen():
            r = self._get_client().chat.completions.create(
                model=_MODEL,
                max_tokens=max_tokens,
                messages=messages,
            )
            result = r.choices[0].message.content
            if _trace:
                try:
                    _trace.update(output=result)
                except Exception:
                    pass
            return result

        if stream:
            return _stream_gen()
        return _sync_gen()

    def health(self) -> dict:
        return {"vllm_url": _VLLM_URL}
