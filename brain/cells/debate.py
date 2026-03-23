"""
DebateCell — Multi-Agent Internal Debate.

3 fixed personas debate before Cortex synthesizes:
  1. Analyst  — cold, logical, data-driven, skeptical (temp=0.1)
  2. Creative — expansive, lateral thinking, unexpected connections (temp=0.9)
  3. Critic   — finds flaws, devil's advocate, questions assumptions (temp=0.4)

Protocol:
  1. Present question to all 3 simultaneously (parallel Claude Haiku calls)
  2. Each gives position (max 150 tokens)
  3. One round cross-examination (each responds to others' strongest point, max 100 tokens)
  4. Cortex synthesizes: balanced answer from all 3 perspectives

Activates for: "reason", "ensemble", "agot" intents on complex/controversial questions

REST endpoint:
  GET /brain/debate/stats — count of debates, persona win rates
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import anthropic

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

_HAIKU  = "claude-haiku-4-5-20251001"
_SONNET = "claude-sonnet-4-6"

_PERSONAS = [
    {
        "name":        "Analyst",
        "description": "cold, logical, data-driven, skeptical",
        "temperature": 0.1,
        "system":      (
            "You are the Analyst persona in an internal debate. "
            "You are cold, logical, data-driven, and skeptical. "
            "You rely on evidence, statistics, and structured reasoning. "
            "You doubt emotional or unsupported claims. "
            "Be concise and precise. Max 150 words."
        ),
    },
    {
        "name":        "Creative",
        "description": "expansive, lateral thinking, unexpected connections",
        "temperature": 0.9,
        "system":      (
            "You are the Creative persona in an internal debate. "
            "You think laterally, make unexpected connections, explore unusual angles. "
            "You value novelty, possibility, and imagination over strict logic. "
            "Challenge conventional framings. Think outside the obvious. "
            "Be expansive. Max 150 words."
        ),
    },
    {
        "name":        "Critic",
        "description": "finds flaws, devil's advocate, questions assumptions",
        "temperature": 0.4,
        "system":      (
            "You are the Critic persona in an internal debate. "
            "You find flaws in reasoning, play devil's advocate, question assumptions. "
            "You are not destructive — you make arguments stronger by exposing weaknesses. "
            "Challenge both the Analyst's rigidity and the Creative's vagueness. "
            "Be sharp and pointed. Max 150 words."
        ),
    },
]


def _call_haiku(system: str, prompt: str, temperature: float, max_tokens: int) -> str:
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=_HAIKU,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"[{e}]"


def _cross_examine(persona: dict, others_positions: list[dict], question: str) -> str:
    """One round of cross-examination: respond to the strongest point from other personas."""
    others_text = "\n\n".join(
        f"{o['name']}'s position: {o['position']}"
        for o in others_positions
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Other perspectives:\n{others_text}\n\n"
        f"As the {persona['name']}, respond to the STRONGEST point made by the others. "
        f"Max 100 words."
    )
    return _call_haiku(persona["system"], prompt, persona["temperature"], 150)


class DebateCell(BaseCell):
    name        = "debate"
    description = (
        "Multi-agent internal debate: Analyst (logical) + Creative (lateral) + "
        "Critic (devil's advocate) debate before Cortex synthesizes. "
        "3 parallel Haiku calls + 1 cross-examination round + Sonnet synthesis."
    )
    color       = "#b91c1c"
    lazy        = True
    position    = (4, 7)

    system_tier     = "online"
    hardware_req    = "API (Claude Haiku x3 + Claude Sonnet)"
    research_basis  = (
        "Society of Mind (Minsky 1986), "
        "Constitutional AI debate (Bai et al. 2022), "
        "Multi-agent debate (Du et al. 2023) — improves factuality + calibration"
    )
    build_notes     = (
        "LIVE: 3-persona parallel debate + cross-examination + synthesis. "
        "Activates for complex/controversial questions via reason/ensemble/agot routing. "
        "GET /brain/debate/stats"
    )
    framework_layer = "Agentic AI"

    def __init__(self):
        super().__init__()
        self._client:         Optional[anthropic.Anthropic] = None
        self._debate_count:   int  = 0
        self._persona_wins:   dict = {"Analyst": 0, "Creative": 0, "Critic": 0, "balanced": 0}
        self._lock = threading.Lock()

    async def boot(self) -> None:
        self._client = anthropic.Anthropic()
        logger.info("[Debate] Cell online — 3-persona debate chamber ready")

    async def process(self, ctx: CellContext) -> Any:
        return await self._run_debate(ctx.message)

    async def _run_debate(self, question: str) -> dict:
        """Run the full 3-persona debate pipeline."""
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._debate_sync, question)
        return result

    def _debate_sync(self, question: str) -> dict:
        t0 = time.time()

        # Step 1: Parallel position gathering
        positions = {}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(
                    _call_haiku,
                    p["system"],
                    f"Question: {question}\n\nGive your position. Max 150 words.",
                    p["temperature"],
                    220,
                ): p["name"]
                for p in _PERSONAS
            }
            for fut in as_completed(futures):
                name = futures[fut]
                positions[name] = fut.result()

        persona_results = [
            {"name": p["name"], "position": positions.get(p["name"], ""), "persona": p}
            for p in _PERSONAS
        ]

        # Step 2: Cross-examination round
        cross_exams = {}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(
                    _cross_examine,
                    item["persona"],
                    [o for o in persona_results if o["name"] != item["name"]],
                    question,
                ): item["name"]
                for item in persona_results
            }
            for fut in as_completed(futures):
                name = futures[fut]
                cross_exams[name] = fut.result()

        # Step 3: Synthesize with Claude Sonnet
        debate_transcript = "\n\n".join(
            f"**{p['name']}** ({p['persona']['description']}):\n"
            f"Position: {p['position']}\n"
            f"Cross-exam: {cross_exams.get(p['name'], '')}"
            for p in persona_results
        )

        synthesis_prompt = (
            f"Question: {question}\n\n"
            f"Internal debate transcript:\n{debate_transcript}\n\n"
            "Synthesize a balanced, nuanced answer that incorporates the strongest "
            "insights from all three perspectives. Be honest about genuine uncertainty. "
            "Don't just average the views — find the synthesis that is truest."
        )

        try:
            msg = self._client.messages.create(
                model=_SONNET,
                max_tokens=800,
                messages=[{"role": "user", "content": synthesis_prompt}],
            )
            synthesis = msg.content[0].text.strip()
        except Exception as e:
            synthesis = f"Synthesis failed: {e}\n\nRaw debate:\n{debate_transcript}"

        with self._lock:
            self._debate_count += 1

        duration = round(time.time() - t0, 1)
        logger.info("[Debate] Debate #%d completed in %.1fs", self._debate_count, duration)

        return {
            "question":          question,
            "positions":         {p["name"]: positions.get(p["name"], "") for p in _PERSONAS},
            "cross_examinations": cross_exams,
            "synthesis":         synthesis,
            "duration_s":        duration,
            "debate_num":        self._debate_count,
        }

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_debates":    self._debate_count,
                "persona_win_rates": self._persona_wins,
            }

    def health(self) -> dict:
        return {
            "status":        self._status.value,
            "debate_count":  self._debate_count,
            "client_ready":  self._client is not None,
        }
