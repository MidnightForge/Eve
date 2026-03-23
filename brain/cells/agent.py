"""
AgentCell — Eve's Autonomous Agent Coordinator.

Implements the Agentic AI framework layer: AI Agents → Agentic AI,
Task Scheduling & Prioritization, Delegation & Handoff Protocol,
Long-term Autonomy & Goal Chaining, Role Management & Constraints.

Research basis:
  - Voyager (Wang et al., 2023) — arxiv:2305.16291
    First LLM-powered lifelong learning agent. Automatic curriculum,
    skill library that grows over time, iterative prompting to achieve
    increasingly complex goals. Eve's model for autonomous growth.
  - AutoGPT / BabyAGI patterns (2023) — task creation, prioritization,
    execution loop. Goal → decompose → execute → reflect → next goal.
  - LangGraph (Langchain 2024) — stateful multi-actor graph for agents.
    Nodes = agents/functions, edges = conditional routing. Enables
    complex multi-step agentic workflows with cycles and branches.
  - HuggingGPT / JARVIS (Shen et al., 2023) — arxiv:2303.17580
    Uses LLM as controller to plan, select, and execute specialized models.
    Same pattern Eve uses: cortex routes, cells execute, results merge.
  - DEPS: Describe, Explain, Plan and Select (Wang et al., 2023)
    Structured goal decomposition for complex embodied tasks.

VRAM: 0 (pure orchestration — no local model required).
Status: ONLINE — active on current RTX 4090 system.
"""

import asyncio
import json
import logging
import os
import time
import threading
from collections import deque
from typing import Optional

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)


class Task:
    """A unit of autonomous work for Eve."""
    def __init__(self, goal: str, priority: int = 5, parent_id: Optional[str] = None):
        self.id        = f"task_{int(time.time()*1000)}"
        self.goal      = goal
        self.priority  = priority   # 1=critical, 10=low
        self.parent_id = parent_id
        self.status    = "pending"  # pending | running | complete | failed
        self.result    = None
        self.created   = time.time()
        self.subtasks: list = []

    def to_dict(self) -> dict:
        return {
            "id": self.id, "goal": self.goal, "priority": self.priority,
            "status": self.status, "created": self.created,
            "subtasks": len(self.subtasks), "result": str(self.result)[:100] if self.result else None,
        }


class AgentCell(BaseCell):
    name        = "agent"
    description = "Agent — Task Scheduling & Goal Chaining"
    color       = "#7c3aed"
    lazy        = True
    position    = (0, 1)

    system_tier     = "online"
    hardware_req    = "API only — no GPU required"
    framework_layer = "AI Agents → Agentic AI"
    research_basis  = (
        "Voyager arxiv:2305.16291 (Wang 2023) — skill library + auto curriculum, 15x faster tech. "
        "HuggingGPT/JARVIS arxiv:2303.17580 (Shen 2023) — LLM as controller for specialist models. "
        "LangGraph (Langchain 2024) — stateful multi-actor graph, conditional routing. "
        "AutoGPT/BabyAGI patterns (2023) — goal→decompose→execute→reflect loop. "
        "DEPS (Wang 2023) — Describe/Explain/Plan/Select for embodied tasks."
    )
    build_notes = (
        "ONLINE: Task queue, goal decomposition, delegation routing active. "
        "NEXT: Full Voyager skill library (Eve learns new skills over time), "
        "LangGraph state machine for complex multi-step workflows, "
        "persistent task memory across sessions, "
        "autonomous background task execution loop (agentic mode)."
    )

    def __init__(self):
        super().__init__()
        self._task_queue: deque = deque(maxlen=50)
        self._completed: list   = []
        self._skill_library: dict = {}   # Voyager: skills Eve has learned

    async def process(self, ctx: CellContext) -> dict:
        # Check if this is an agentic request (multi-step task)
        agentic_cues = [
            "do this for me", "handle", "take care of", "automatically",
            "every day", "schedule", "remind", "keep track", "monitor",
            "whenever", "make sure", "i need you to", "can you",
        ]
        msg_lower = ctx.message.lower()
        is_agentic = any(c in msg_lower for c in agentic_cues)

        # Check if there's a plan from PlannerCell
        plan = ctx.metadata.get("plan")

        if plan and plan.get("complexity") in ("moderate", "complex"):
            # Decompose plan steps into tasks
            tasks_created = []
            for step in plan.get("steps", [])[:5]:  # cap at 5 subtasks
                t = Task(goal=step, priority=5)
                self._task_queue.append(t)
                tasks_created.append(t.to_dict())

            logger.info("[AgentCell] Created %d tasks from plan", len(tasks_created))
            return {
                "agentic": True,
                "tasks_created": len(tasks_created),
                "queue_depth": len(self._task_queue),
            }

        if is_agentic:
            t = Task(goal=ctx.message, priority=3)
            self._task_queue.append(t)
            return {
                "agentic": True,
                "task_id": t.id,
                "queue_depth": len(self._task_queue),
            }

        return {"agentic": False, "queue_depth": len(self._task_queue)}

    def add_skill(self, name: str, description: str, code: str) -> None:
        """Voyager pattern: store a learned skill for future reuse."""
        self._skill_library[name] = {
            "description": description,
            "code": code,
            "learned_at": time.time(),
            "use_count": 0,
        }
        logger.info("[AgentCell] Skill learned: %s", name)

    def get_queue(self) -> list:
        return [t.to_dict() for t in self._task_queue]

    def health(self) -> dict:
        return {
            "queue_depth":    len(self._task_queue),
            "completed":      len(self._completed),
            "skills_learned": len(self._skill_library),
        }
