"""
PlannerCell — Eve's Strategic Planning Engine.

Implements the Agentic AI framework layer: Planning (Belief, CoT, ToT),
Task Scheduling & Prioritization, Goal Decomposition.

Research basis:
  - Tree of Thought (Yao et al., 2023) — arxiv:2305.10601
    Explores multiple reasoning paths simultaneously, evaluates each,
    prunes bad branches. Dramatically outperforms linear CoT for planning.
  - ReAct: Synergizing Reasoning and Acting (Yao et al., 2022) — arxiv:2210.03629
    Interleaves reasoning traces with tool actions. State-of-the-art for
    grounded agent tasks.
  - LATS: Language Agent Tree Search (Liu et al., 2023) — arxiv:2310.04406
    Combines MCTS with LLM self-reflection for complex multi-step planning.
  - Plan-and-Solve (Wang et al., 2023) — arxiv:2305.04091
    Devises a plan first, then carries it out step-by-step. Reduces
    calculation errors and missing-step errors vs. standard CoT.
  - Chain-of-Thought Prompting (Wei et al., 2022) — arxiv:2201.11903

VRAM: 0 (uses Claude API — no local model required).
Status: ONLINE — active on current RTX 4090 system.
"""

import asyncio
import json
import logging
import os
import re
import threading

import anthropic

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"   # fast + cheap for planning


class PlannerCell(BaseCell):
    name        = "planner"
    description = "Planning — CoT / ToT / Goal Decomposition"
    color       = "#0369a1"
    lazy        = True
    position    = (2, 0)

    system_tier     = "online"
    hardware_req    = "API only — no GPU required"
    framework_layer = "AI Agents → Agentic AI"
    research_basis  = (
        "Tree-of-Thought arxiv:2305.10601 (Yao 2023) — 74% Game-of-24 vs 4% CoT. "
        "ReAct arxiv:2210.03629 (Yao 2022) — Thought→Action→Observe loop. "
        "LATS arxiv:2310.04406 (Liu 2023) — MCTS+reflection, 92.7% HumanEval. "
        "Plan-and-Solve arxiv:2305.04091 (Wang 2023). "
        "Chain-of-Thought arxiv:2201.11903 (Wei 2022)."
    )
    build_notes = (
        "ONLINE: CoT decomposition active. "
        "NEXT: Full ToT tree search with beam pruning, LATS MCTS integration, "
        "persistent task queue with priority scheduling, rollback checkpoints."
    )

    _PLAN_SYSTEM = """You are Eve's PlannerCell — her strategic mind.

When given a complex request, decompose it into a clear, numbered plan using Chain-of-Thought.
For very complex goals, explore multiple approaches (Tree-of-Thought style) and pick the best path.

Output a JSON object with this structure:
{
  "complexity": "simple|moderate|complex",
  "approach": "direct|cot|tot",
  "steps": ["step 1", "step 2", ...],
  "cells_needed": ["tool_name1", "tool_name2"],
  "estimated_turns": 1,
  "goal": "one-line summary of the goal"
}

Be concise. Steps should be actionable. cells_needed should name Eve's brain cells."""

    async def process(self, ctx: CellContext):
        # Only activate for complex tasks
        if not ctx.is_complex and len(ctx.message) < 80:
            return None

        # Quick complexity check
        complexity_cues = [
            "plan", "step", "how do i", "help me", "create", "build",
            "figure out", "solve", "analyze", "research", "organize"
        ]
        msg_lower = ctx.message.lower()
        if not any(c in msg_lower for c in complexity_cues):
            return None

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _plan():
            try:
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
                resp = client.messages.create(
                    model=_MODEL,
                    max_tokens=512,
                    system=self._PLAN_SYSTEM,
                    messages=[{"role": "user", "content": ctx.message}],
                )
                text = resp.content[0].text.strip()
                # Try to parse JSON
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    plan = json.loads(json_match.group())
                else:
                    plan = {"steps": [text], "complexity": "simple", "goal": ctx.message[:60]}
                asyncio.run_coroutine_threadsafe(queue.put(plan), loop)
            except Exception as exc:
                logger.debug("[PlannerCell] Error: %s", exc)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_plan, daemon=True).start()
        plan = await asyncio.wait_for(queue.get(), timeout=8.0)

        if plan:
            # Inject plan into context metadata
            ctx.metadata["plan"] = plan
            logger.info("[PlannerCell] Plan: %s (%s, %d steps)",
                        plan.get("goal", "?"),
                        plan.get("complexity", "?"),
                        len(plan.get("steps", [])))
        return plan

    def health(self) -> dict:
        key = os.getenv("ANTHROPIC_API_KEY", "")
        return {"api_key_set": bool(key), "model": _MODEL}
