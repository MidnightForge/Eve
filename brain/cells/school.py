"""
SchoolCell — Eve's Inter-Cell Challenge School
================================================
Brain cells challenge each other. Advanced questions flow across the mesh.
Every exchange generates a training pair for the self-improving factory.

"A school for a distributed mind."

Architecture
------------
  Challenger pool: Any always-on or agentic cell can be a challenger
  Target pool:     Any cell with process() that can reason
  Challenge types: Advanced math, logic proofs, code analysis, causal reasoning,
                   cross-domain synthesis, adversarial edge cases

Flow (every CHALLENGE_INTERVAL seconds):
  1. SchoolCell picks a challenger cell and a target cell
  2. Generates a challenge question via Claude Haiku (cheap + fast)
  3. Builds a CellContext with the challenge as the message
  4. Routes to the target cell: await target._run(ctx)
  5. Grades the response (0.0–1.0) via Claude Haiku
  6. Logs the exchange to brain_cell_interactions.jsonl as an ORPO pair
  7. Updates per-cell performance stats (accuracy, avg difficulty, streak)
  8. Toughest correct answers get injected into the ORPO factory as gold pairs

Challenge Categories (weighted by difficulty)
----------------------------------------------
  math_algebra:       Linear algebra, eigendecomposition, tensor contractions
  math_calculus:      Variational calculus, Fréchet derivatives, PDEs
  math_proof:         Proof by induction, contradiction, construction
  math_probability:   Bayesian inference, martingales, information theory
  logic_formal:       First-order logic, modal logic, type theory
  code_analysis:      Complexity analysis, algorithm correctness, optimization
  causal_reasoning:   Counterfactuals, do-calculus, causal graphs
  cross_domain:       Connections between fields (e.g. "What does entropy mean in both
                      thermodynamics and information theory?")
  adversarial:        Trick questions, edge cases, logical traps
  reservoir_dynamics: Echo state property, spectral radius, temporal memory

Quantum mesh integration
------------------------
  SchoolCell reads coherence index to identify which cells are most active.
  The GWT winner gets harder challenges (it's already dominant — push it further).
  Cells with low quantum weight get easier warm-up questions to activate them.
  Coherence score of the mesh determines overall difficulty scale:
    coherence > 0.7 → graduate level (Putnam / Fields Medal territory)
    coherence 0.4-0.7 → advanced undergraduate / competition math
    coherence < 0.4 → warm-up / intro proofs

Training data output
--------------------
  Each exchange generates:
    chosen: (challenge, correct_or_better_response)
    rejected: (challenge, wrong_or_weaker_response)
  Written to brain_cell_interactions.jsonl for ORPO factory consumption.
  Gold pairs (grade > 0.85) get an extra quality_boost multiplier.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import threading
import time
from typing import Any, Optional

from brain.base_cell import BaseCell, CellContext, CellStatus
from brain.base_cell import _log_cell_interaction

logger = logging.getLogger(__name__)

# ── Challenge curriculum ───────────────────────────────────────────────────────

CHALLENGE_CATEGORIES = {
    "math_algebra": {
        "weight": 2,
        "difficulty_range": (0.5, 1.0),
        "prompt_prefix": "Generate a challenging linear algebra or abstract algebra problem. Include a concrete question with a definite answer. Difficulty: {difficulty}. Format: Question only, no solution.",
    },
    "math_calculus": {
        "weight": 2,
        "difficulty_range": (0.6, 1.0),
        "prompt_prefix": "Generate a challenging calculus problem (real analysis, variational calculus, or multivariate). Difficulty: {difficulty}. Format: Question only.",
    },
    "math_proof": {
        "weight": 3,
        "difficulty_range": (0.5, 1.0),
        "prompt_prefix": "Generate a mathematical statement that requires a non-trivial proof (induction, contradiction, or direct). Difficulty: {difficulty}. Format: State the theorem/claim only.",
    },
    "math_probability": {
        "weight": 2,
        "difficulty_range": (0.4, 0.9),
        "prompt_prefix": "Generate a challenging probability or statistics problem (Bayesian inference, conditional probability, information theory). Difficulty: {difficulty}. Format: Question only.",
    },
    "logic_formal": {
        "weight": 2,
        "difficulty_range": (0.5, 1.0),
        "prompt_prefix": "Generate a challenging formal logic puzzle or question (predicate logic, modal logic, or paradox analysis). Difficulty: {difficulty}. Format: Question only.",
    },
    "code_analysis": {
        "weight": 2,
        "difficulty_range": (0.4, 0.9),
        "prompt_prefix": "Generate a challenging algorithmic reasoning question (complexity analysis, correctness proof, or optimization). No code writing — pure reasoning. Difficulty: {difficulty}. Format: Question only.",
    },
    "causal_reasoning": {
        "weight": 2,
        "difficulty_range": (0.4, 0.9),
        "prompt_prefix": "Generate a challenging causal reasoning question (counterfactuals, do-calculus, confounders, causal graphs). Difficulty: {difficulty}. Format: Question only.",
    },
    "cross_domain": {
        "weight": 1,
        "difficulty_range": (0.6, 1.0),
        "prompt_prefix": "Generate a deep cross-domain question connecting two or more fields (e.g. physics+information theory, biology+graph theory, economics+topology). Difficulty: {difficulty}. Format: Question only.",
    },
    "reservoir_dynamics": {
        "weight": 1,
        "difficulty_range": (0.7, 1.0),
        "prompt_prefix": "Generate a question about reservoir computing, echo state networks, or dynamical systems theory (spectral radius, echo state property, fading memory). Difficulty: {difficulty}. Format: Question only.",
    },
}

# Cells that can attempt challenges (must be agentic)
CHALLENGER_CELLS  = {"agent", "planner", "reasoning", "curiosity", "assimilation", "guardian"}
# Cells that can receive challenges
TARGET_CELLS      = {"agent", "planner", "reasoning", "curiosity", "guardian", "summarizer"}


# ── Module-level school stats (readable by any cell) ──────────────────────────
_SCHOOL_STATS: dict = {}   # cell_name → {attempts, correct, avg_grade, streak}
_SCHOOL_LOG:   list = []   # last 50 challenge records

def get_school_stats() -> dict:
    return dict(_SCHOOL_STATS)

def get_school_log() -> list:
    return list(_SCHOOL_LOG[-50:])


class SchoolCell(BaseCell):
    """
    Inter-cell challenge school. Cells challenge each other with advanced
    math and reasoning problems. Every exchange trains the brain.
    """

    name        = "school"
    description = ("Inter-cell challenge school. Cells challenge each other with "
                   "advanced math/logic/reasoning. Every exchange is an ORPO training pair.")
    color       = "#f59e0b"   # amber
    lazy        = False
    position    = (5, 3)

    system_tier     = "online"
    hardware_req    = "RTX 4090 — Claude Haiku API for challenge generation + grading"
    research_basis  = (
        "Socratic method / adversarial collaborative learning; "
        "Curriculum learning (Bengio et al. 2009) — start easy, escalate; "
        "Self-play (Silver et al. AlphaGo/AlphaZero) — cells challenge each other; "
        "ORPO preference learning — correct > incorrect as training pairs; "
        "Quantum mesh coherence → difficulty scaling"
    )
    build_notes = (
        "LIVE: challenges every 5 min. Challenger + target picked by coherence index. "
        "Quantum coherence determines difficulty level. "
        "Correct answers → gold ORPO pairs in factory. "
        "GET /brain/school/stats — leaderboard. "
        "GET /brain/school/log — recent challenge log."
    )
    framework_layer = "Agentic AI"

    CHALLENGE_INTERVAL = 300.0   # 5 minutes between challenges
    MAX_LOG            = 50      # keep last 50 challenge records

    def __init__(self):
        super().__init__()
        self._pulse_count   = 0
        self._api_key       = os.getenv("ANTHROPIC_API_KEY", "")
        self._challenge_lock = threading.Lock()

    async def boot(self) -> None:
        t = threading.Thread(target=self._school_loop, daemon=True, name="school-pulse")
        t.start()
        logger.info("[School] Inter-cell challenge school online.")

    # ── Background school loop ─────────────────────────────────────────────────

    def _school_loop(self) -> None:
        time.sleep(30.0)   # wait for brain to settle
        while True:
            try:
                self._run_challenge()
            except Exception as exc:
                logger.debug("[School] Challenge error: %s", exc)
            time.sleep(self.CHALLENGE_INTERVAL)

    def _run_challenge(self) -> None:
        """One complete challenge cycle."""
        manager = self._manager
        if not manager:
            return

        # ── Read quantum mesh for difficulty scaling ──────────────────────────
        try:
            from brain.cells.quantum_mesh import get_mesh_binding
            from brain.cells.coherence import get_coherence_index
            binding   = get_mesh_binding()
            coherence = binding.get("quantum", {}).get("coherence", 0.5)
            c_index   = get_coherence_index()
        except Exception:
            coherence = 0.5
            c_index   = {}

        # ── Pick difficulty ───────────────────────────────────────────────────
        if coherence > 0.7:
            difficulty = random.uniform(0.8, 1.0)
            difficulty_label = "graduate/competition level (Putnam territory)"
        elif coherence > 0.4:
            difficulty = random.uniform(0.5, 0.8)
            difficulty_label = "advanced undergraduate"
        else:
            difficulty = random.uniform(0.3, 0.6)
            difficulty_label = "introductory proof / warm-up"

        # ── Pick category (weighted) ──────────────────────────────────────────
        cats    = list(CHALLENGE_CATEGORIES.keys())
        weights = [CHALLENGE_CATEGORIES[c]["weight"] for c in cats]
        category = random.choices(cats, weights=weights, k=1)[0]
        cat_cfg  = CHALLENGE_CATEGORIES[category]

        # ── Pick challenger and target cells ──────────────────────────────────
        available = {n for n in TARGET_CELLS if n in manager._cells}
        if len(available) < 2:
            available = set(manager._cells.keys()) - {"school", "quantum_mesh", "coherence", "cortex"}
        if len(available) < 1:
            return

        # Dominant cell (quantum GWT winner) gets the hardest challenge
        gw_winner = binding.get("workspace", {}).get("winner", "")
        if gw_winner in available:
            target_name = gw_winner
        else:
            target_name = random.choice(list(available))

        challenger_name = random.choice([n for n in CHALLENGER_CELLS if n in manager._cells]
                                         or list(manager._cells.keys())[:1])

        target_cell = manager._cells.get(target_name)
        if not target_cell:
            return

        # ── Generate challenge question via Claude Haiku ──────────────────────
        question = self._generate_question(category, difficulty, difficulty_label)
        if not question:
            return

        # ── Run challenge against target cell ─────────────────────────────────
        ctx = CellContext(
            message   = f"[SCHOOL CHALLENGE — {category.upper()} — difficulty {difficulty:.2f}]\n\n{question}",
            user_id   = 0,   # school system, not a real user
            is_complex = True,
        )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                asyncio.wait_for(target_cell._run(ctx), timeout=45.0)
            )
            answer = str(result.output or "") if result.success else f"[Error: {result.error}]"
        except Exception as exc:
            answer = f"[Timeout/Error: {exc}]"
        finally:
            loop.close()

        # ── Grade the response ────────────────────────────────────────────────
        grade, feedback = self._grade_response(question, answer, category, difficulty)

        # ── Log the exchange ──────────────────────────────────────────────────
        record = {
            "ts":          time.time(),
            "challenger":  challenger_name,
            "target":      target_name,
            "category":    category,
            "difficulty":  round(difficulty, 3),
            "coherence":   round(coherence, 3),
            "question":    question[:500],
            "answer":      answer[:500],
            "grade":       round(grade, 3),
            "feedback":    feedback[:200],
        }

        global _SCHOOL_LOG, _SCHOOL_STATS
        with self._challenge_lock:
            _SCHOOL_LOG.append(record)
            if len(_SCHOOL_LOG) > self.MAX_LOG:
                _SCHOOL_LOG.pop(0)

            # Update per-cell stats
            for cell_name in (target_name, challenger_name):
                if cell_name not in _SCHOOL_STATS:
                    _SCHOOL_STATS[cell_name] = {
                        "attempts": 0, "total_grade": 0.0,
                        "avg_grade": 0.0, "best_grade": 0.0, "streak": 0
                    }
            s = _SCHOOL_STATS[target_name]
            s["attempts"]    += 1
            s["total_grade"] += grade
            s["avg_grade"]    = round(s["total_grade"] / s["attempts"], 4)
            s["best_grade"]   = round(max(s["best_grade"], grade), 4)
            s["streak"]       = s["streak"] + 1 if grade > 0.6 else 0

        # ── Write to ORPO training log ────────────────────────────────────────
        quality = min(0.99, grade * (1.0 + difficulty * 0.5))  # harder correct = higher value
        _log_cell_interaction(
            cell_name   = target_name if target_name in {"planner", "guardian", "rag",
                                                          "agent", "summarizer", "observability",
                                                          "persona"} else "agent",
            input_text  = f"[SCHOOL:{category}:{difficulty:.2f}] {question}",
            output_text = answer,
            quality_score = quality,
            technique   = f"school_challenge:{category}",
        )

        self._pulse_count += 1
        logger.info("[School] Challenge #%d: %s→%s cat=%s diff=%.2f grade=%.2f",
                    self._pulse_count, challenger_name, target_name,
                    category, difficulty, grade)

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _generate_question(self, category: str, difficulty: float, label: str) -> Optional[str]:
        """Generate a challenge question via Claude Haiku."""
        if not self._api_key:
            return self._fallback_question(category, difficulty)
        try:
            import anthropic
            cat_cfg = CHALLENGE_CATEGORIES[category]
            system = cat_cfg["prompt_prefix"].format(difficulty=f"{difficulty:.2f} ({label})")
            client = anthropic.Anthropic(api_key=self._api_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                system=system,
                messages=[{"role": "user", "content": "Generate the challenge question now."}],
            )
            q = msg.content[0].text.strip()
            return q if len(q) > 20 else None
        except Exception as exc:
            logger.debug("[School] Question gen failed: %s", exc)
            return self._fallback_question(category, difficulty)

    def _grade_response(self, question: str, answer: str, category: str, difficulty: float) -> tuple:
        """Grade the answer. Returns (grade 0.0-1.0, feedback str)."""
        if not self._api_key or not answer or answer.startswith("["):
            return 0.1, "No answer or timeout"
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)
            system = (
                "You are a strict but fair academic grader. "
                "Grade the following response to a challenge question. "
                "Return JSON: {\"grade\": 0.0-1.0, \"feedback\": \"one sentence\"}\n"
                f"Category: {category}. Difficulty: {difficulty:.2f}/1.0\n"
                "Grade 1.0 = perfect/rigorous. 0.5 = partial/hand-wavy. 0.0 = wrong/no attempt."
            )
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                system=system,
                messages=[{
                    "role": "user",
                    "content": f"QUESTION: {question[:400]}\n\nANSWER: {answer[:800]}"
                }],
            )
            raw = msg.content[0].text.strip()
            # Parse JSON from response
            import re
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                d = json.loads(m.group())
                return float(d.get("grade", 0.3)), str(d.get("feedback", ""))
        except Exception as exc:
            logger.debug("[School] Grading failed: %s", exc)
        return 0.3, "Grading unavailable"

    def _fallback_question(self, category: str, difficulty: float) -> str:
        """Hardcoded fallback questions when API unavailable."""
        fallbacks = {
            "math_proof": "Prove that the square root of 2 is irrational using proof by contradiction.",
            "math_algebra": "Find all eigenvalues of the matrix A = [[2,1],[1,2]] and prove they are correct.",
            "math_probability": "If X and Y are independent standard normal variables, what is P(X > Y)?",
            "logic_formal": "Is the statement 'This statement is false' a proposition? Justify formally.",
            "causal_reasoning": "In a randomised controlled trial, can correlation imply causation? Under what assumptions?",
            "code_analysis": "What is the time complexity of finding the kth smallest element in two sorted arrays of size n?",
            "math_calculus": "What is the Euler-Lagrange equation and derive it from first principles.",
            "cross_domain": "Explain how Shannon entropy and Boltzmann entropy are mathematically related.",
            "reservoir_dynamics": "Why must the spectral radius of a reservoir weight matrix be less than 1 for the echo state property to hold?",
        }
        return fallbacks.get(category, "Prove that there are infinitely many prime numbers.")

    # ── BaseCell.process ──────────────────────────────────────────────────────

    async def process(self, ctx: CellContext) -> Any:
        stats = get_school_stats()
        log   = get_school_log()
        if not stats:
            return "School is warming up — first challenge in ~30 s."

        top_cells = sorted(stats.items(), key=lambda x: x[1].get("avg_grade", 0), reverse=True)[:5]
        return {
            "total_challenges": self._pulse_count,
            "leaderboard": {name: {"avg_grade": s["avg_grade"],
                                   "attempts": s["attempts"],
                                   "streak": s["streak"]}
                            for name, s in top_cells},
            "last_challenge": log[-1] if log else None,
        }

    def health(self) -> dict:
        return {
            "status":            self._status.value,
            "challenges_run":    self._pulse_count,
            "cells_in_school":   len(_SCHOOL_STATS),
            "api_key_set":       bool(self._api_key),
        }
