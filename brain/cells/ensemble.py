"""
CompetitiveEnsembleCell — Multiple Agents, Best Answer Wins
===========================================================
Pillar #3 from "14 Key Pillars of Agentic AI": instead of trusting one LLM
call, spawn N agents simultaneously with the same query, then have a judge
select the best answer. Eliminates single-point-of-failure reasoning.

How it works
------------
  1. Query arrives (routed here for complex/high-stakes reasoning tasks)
  2. Spawn 3 parallel Claude Haiku completions with temperature diversity:
       Agent A: temperature=0.0  (deterministic, conservative)
       Agent B: temperature=0.7  (balanced)
       Agent C: temperature=1.0  (creative, exploratory)
  3. Judge (Claude Haiku, temp=0) selects the best response based on:
       - Accuracy / logical consistency
       - Completeness
       - Clarity
  4. Return winner + metadata (which agent won, why)

This is qualitatively different from just "asking again" — the diversity
of temperature creates genuinely different reasoning paths. The judge then
picks the strongest one, not just the first one.

When does Cortex route here?
-----------------------------
  "complex" flag set in CellContext (is_complex=True), OR
  Query contains keywords: prove, verify, calculate, solve, analyze, explain why,
  what is the best, compare, evaluate, review

Integration with FormalReasoningCell
--------------------------------------
  When FormalReasoningCell is also activated, the ensemble cell defers math
  proofs to it. Ensemble handles reasoning, narrative, analysis, and code logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import anthropic

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"


class CompetitiveEnsembleCell(BaseCell):
    """
    Runs 3 parallel LLM completions with temperature diversity,
    then judges which answer is best. Returns winner + reasoning.
    """

    name        = "ensemble"
    description = (
        "Competitive ensemble reasoning — runs 3 parallel agents with different "
        "temperatures, judge selects the best answer. For complex analysis, "
        "proof-checking, code review, and high-stakes reasoning."
    )
    color       = "#f59e0b"   # amber
    lazy        = True        # only activates for complex queries
    position    = (7, 1)

    system_tier     = "online"
    hardware_req    = "RTX 4090 — Claude API, no GPU"
    research_basis  = (
        "Fareed Khan '14 Pillars of Agentic AI' Pillar #3 — Competitive Agent Ensembles; "
        "Wang et al. 2022 'Self-Consistency' (temperature sampling + majority vote); "
        "Chen et al. 2021 'Evaluating Large Language Models on Code' — diversity sampling; "
        "Madaan et al. 2023 'Self-Refine' — iterative feedback for quality improvement"
    )
    build_notes = (
        "LIVE: 3 parallel Haiku agents (temp=0/0.7/1.0) + Haiku judge. "
        "~3s wall-clock (parallel, not sequential). "
        "Returns winner response + agent_id + judge_reasoning. "
        "Defers math to FormalReasoningCell when both cells are active."
    )
    framework_layer = "Agentic AI"

    def __init__(self):
        super().__init__()
        self._client: Optional[anthropic.Anthropic] = None
        self._ensemble_count = 0
        self._win_counts = {"A": 0, "B": 0, "C": 0}

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    async def _agent_call(
        self,
        agent_id: str,
        system: str,
        message: str,
        temperature: float,
    ) -> tuple[str, str]:
        """Single agent call. Returns (agent_id, response_text)."""
        loop = asyncio.get_event_loop()
        def _call():
            r = self._get_client().messages.create(
                model=_MODEL,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": message}],
                temperature=temperature,
            )
            return r.content[0].text.strip()
        try:
            text = await asyncio.wait_for(
                loop.run_in_executor(None, _call), timeout=20.0
            )
        except Exception as e:
            text = f"[Agent {agent_id} failed: {e}]"
        return agent_id, text

    async def _judge(
        self,
        question: str,
        responses: dict[str, str],
    ) -> tuple[str, str]:
        """
        Judge which response is best. Returns (winning_agent_id, judge_reasoning).
        """
        loop = asyncio.get_event_loop()
        options = "\n\n".join(
            f"=== Agent {aid} ===\n{resp}"
            for aid, resp in responses.items()
        )
        judge_prompt = (
            f"Question: {question}\n\n"
            f"Three agents answered:\n\n{options}\n\n"
            "Which agent gave the BEST answer? Consider: accuracy, completeness, clarity.\n"
            "Reply with exactly: WINNER: <A or B or C>\nREASON: <one sentence>"
        )
        def _call():
            r = self._get_client().messages.create(
                model=_MODEL,
                max_tokens=100,
                system="You are a strict judge. Pick the best answer from the three agents.",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
            )
            return r.content[0].text.strip()
        try:
            verdict = await asyncio.wait_for(
                loop.run_in_executor(None, _call), timeout=12.0
            )
            # Parse WINNER: X
            winner = "A"
            reason = verdict
            for line in verdict.split("\n"):
                if line.startswith("WINNER:"):
                    w = line.replace("WINNER:", "").strip().upper()
                    if w in responses:
                        winner = w
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()
            return winner, reason
        except Exception as e:
            return "A", f"Judge failed: {e}"

    async def process(self, ctx: CellContext) -> Any:
        message = ctx.message
        system  = (
            "You are Eve, a highly intelligent AI assistant. "
            "Reason carefully and give your best answer. Be precise."
        )

        # Run 3 agents in parallel with temperature diversity
        a_task = self._agent_call("A", system, message, temperature=0.0)
        b_task = self._agent_call("B", system, message, temperature=0.7)
        c_task = self._agent_call("C", system, message, temperature=1.0)

        results = await asyncio.gather(a_task, b_task, c_task, return_exceptions=True)
        responses: dict[str, str] = {}
        for r in results:
            if isinstance(r, tuple):
                aid, text = r
                responses[aid] = text

        if not responses:
            return {"error": "All ensemble agents failed", "ensemble": True}

        if len(responses) == 1:
            winner_id = list(responses.keys())[0]
            return {
                "response":       responses[winner_id],
                "winner":         winner_id,
                "judge_reason":   "Only one agent succeeded",
                "all_responses":  responses,
                "ensemble":       True,
            }

        # Judge selects best
        winner_id, reason = await self._judge(message, responses)
        self._ensemble_count += 1
        self._win_counts[winner_id] = self._win_counts.get(winner_id, 0) + 1

        logger.info("[Ensemble] #%d — winner=%s: %s", self._ensemble_count, winner_id, reason[:60])

        return {
            "response":      responses[winner_id],
            "winner":        winner_id,
            "judge_reason":  reason,
            "all_responses": responses,
            "ensemble":      True,
        }

    def health(self) -> dict:
        return {
            "status":          self._status.value,
            "ensemble_count":  self._ensemble_count,
            "win_counts":      self._win_counts,
        }
