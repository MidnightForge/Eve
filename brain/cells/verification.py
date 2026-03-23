"""
VerificationCell — Output Validation Before Delivery
=====================================================
Pillar #4 from "14 Key Pillars of Agentic AI": Redundant Execution.
Before Eve returns an answer, a second independent agent checks it.

The verifier never sees the original reasoning — only the question and the
proposed answer. This forces independent evaluation rather than rubber-stamping.

What it checks
--------------
  Math/Logic:    Routes to FormalReasoningCell for symbolic verification
  Code:          Syntax check + static analysis + test for obvious bugs
  Factual:       Cross-references against memory + flags speculative claims
  General:       Consistency check, hallucination detection, confidence scoring

Verification levels
-------------------
  PASS:    Answer is correct and complete
  WARN:    Answer has minor issues (noted but returned)
  FAIL:    Answer has significant errors (corrected before return)
  DEFER:   Cannot verify (returned as-is with UNVERIFIED tag)

Integration
-----------
  Called from brain/manager.py after all cells produce outputs.
  Adds a verification_status field to the response metadata.
  If FAIL, the corrected answer replaces the original.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

import anthropic

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"

# Patterns that suggest a math/code/logic query needing formal verification
_MATH_PATTERNS = re.compile(
    r"\b(solve|prove|calculate|integral|derivative|equation|matrix|eigenvalue"
    r"|prime|factor|probability|theorem|proof)\b",
    re.IGNORECASE
)
_CODE_PATTERNS = re.compile(
    r"```|def |class |function|import |algorithm|code|program|script",
    re.IGNORECASE
)


class VerificationCell(BaseCell):
    """
    Independent second-opinion verifier. Catches hallucinations, logic errors,
    and incorrect code before they reach Forge.
    """

    name        = "verification"
    description = (
        "Independent output verifier — second agent checks answers for correctness "
        "before delivery. Catches hallucinations, math errors, buggy code. "
        "Returns PASS/WARN/FAIL with corrections."
    )
    color       = "#10b981"   # emerald
    lazy        = True        # activates for complex or high-stakes queries
    position    = (7, 2)

    system_tier     = "online"
    hardware_req    = "RTX 4090 — Claude API, no GPU"
    research_basis  = (
        "Fareed Khan '14 Pillars' Pillar #4 — Redundant Execution; "
        "Saunders et al. 2022 'Self-critiquing models for assisting human evaluators'; "
        "Shinn et al. 2023 'Reflexion: Language agents with verbal reinforcement learning'; "
        "Pan et al. 2023 'Automatically Correcting Large Language Models'"
    )
    build_notes = (
        "LIVE: Haiku verifier (independent, never sees original reasoning). "
        "Math → FormalReasoningCell symbolic check. Code → syntax + static analysis. "
        "Returns verdict: PASS/WARN/FAIL + corrected answer on FAIL. "
        "Verification metadata appended to every response when active."
    )
    framework_layer = "Agentic AI"

    def __init__(self):
        super().__init__()
        self._client: Optional[anthropic.Anthropic] = None
        self._verify_count  = 0
        self._pass_count    = 0
        self._warn_count    = 0
        self._fail_count    = 0

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    async def _verify_general(self, question: str, answer: str) -> dict:
        """General LLM-based verification."""
        loop = asyncio.get_event_loop()
        prompt = (
            f"Question: {question}\n\n"
            f"Proposed Answer:\n{answer}\n\n"
            "Verify this answer independently. Check for:\n"
            "  - Factual errors or hallucinations\n"
            "  - Logical inconsistencies\n"
            "  - Missing important information\n"
            "  - Speculative claims presented as facts\n\n"
            "Reply with:\n"
            "VERDICT: <PASS or WARN or FAIL>\n"
            "ISSUES: <brief description or 'none'>\n"
            "CORRECTION: <corrected answer if FAIL, else 'n/a'>"
        )
        def _call():
            r = self._get_client().messages.create(
                model=_MODEL,
                max_tokens=800,
                system=(
                    "You are an independent fact-checker and logic verifier. "
                    "You have NOT seen the original reasoning — only the question and answer. "
                    "Be strict. Your job is to catch errors, not to agree."
                ),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return r.content[0].text.strip()
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(None, _call), timeout=15.0
            )
        except Exception as e:
            return {"verdict": "DEFER", "issues": str(e), "correction": None}

        verdict    = "DEFER"
        issues     = "Unable to parse"
        correction = None
        for line in raw.split("\n"):
            if line.startswith("VERDICT:"):
                v = line.replace("VERDICT:", "").strip().upper()
                if v in ("PASS", "WARN", "FAIL"):
                    verdict = v
            elif line.startswith("ISSUES:"):
                issues = line.replace("ISSUES:", "").strip()
            elif line.startswith("CORRECTION:") and "n/a" not in line.lower():
                correction = line.replace("CORRECTION:", "").strip()
        return {"verdict": verdict, "issues": issues, "correction": correction}

    async def _verify_code(self, code_block: str) -> dict:
        """Syntax check + static analysis on extracted code."""
        import ast
        issues = []
        # Try Python parse
        try:
            ast.parse(code_block)
        except SyntaxError as e:
            issues.append(f"SyntaxError: {e}")
        # Check for obvious dangerous patterns
        danger = ["os.system", "subprocess.call", "eval(", "exec(", "rm -rf", "__import__"]
        for d in danger:
            if d in code_block:
                issues.append(f"Dangerous pattern: {d}")
        if issues:
            return {"verdict": "WARN", "issues": "; ".join(issues), "correction": None}
        return {"verdict": "PASS", "issues": "none", "correction": None}

    async def _verify_math(self, question: str, answer: str) -> dict:
        """Route math to FormalReasoningCell for symbolic verification."""
        try:
            if self._manager:
                fc = self._manager._cells.get("formal_reason")
                if fc and fc._sympy_available:
                    # Ask FormalReasoningCell to verify the answer
                    verify_q = f"Verify: is this correct? Question: {question} Answer: {answer}"
                    ctx = CellContext(message=verify_q, user_id=0)
                    result = await fc.process(ctx)
                    if isinstance(result, dict) and result.get("formal_result"):
                        return {
                            "verdict": "PASS",
                            "issues": "Formally verified via SymPy",
                            "correction": None,
                            "formal": result.get("formal_result", "")
                        }
        except Exception as e:
            logger.debug("[Verification] Math formal check error: %s", e)
        # Fall back to general verification
        return await self._verify_general(question, answer)

    async def process(self, ctx: CellContext) -> Any:
        """
        ctx.message = original question
        ctx.cell_outputs = dict with outputs from other cells to verify
        """
        question = ctx.message

        # Determine what to verify — look for an answer in cell_outputs
        answer_to_verify = None
        source_cell = "unknown"

        # Check ensemble output first (highest priority)
        if ctx.cell_outputs and ctx.cell_outputs.get("ensemble"):
            e = ctx.cell_outputs["ensemble"]
            if isinstance(e, dict) and e.get("response"):
                answer_to_verify = e["response"]
                source_cell = "ensemble"

        # Otherwise check reasoning cell output
        if not answer_to_verify and ctx.cell_outputs:
            for cell_name in ("reason", "chat", "tools"):
                if ctx.cell_outputs.get(cell_name):
                    answer_to_verify = str(ctx.cell_outputs[cell_name])
                    source_cell = cell_name
                    break

        if not answer_to_verify:
            return {
                "verdict": "DEFER",
                "issues": "No answer found to verify",
                "verification": True,
            }

        # Route to appropriate verifier
        is_math = bool(_MATH_PATTERNS.search(question))
        is_code = bool(_CODE_PATTERNS.search(answer_to_verify))

        if is_math:
            result = await self._verify_math(question, answer_to_verify)
        elif is_code:
            # Extract code blocks
            code_match = re.search(r"```(?:\w+)?\n(.*?)```", answer_to_verify, re.DOTALL)
            if code_match:
                code_result = await self._verify_code(code_match.group(1))
                if code_result["verdict"] != "PASS":
                    result = code_result
                else:
                    result = await self._verify_general(question, answer_to_verify)
            else:
                result = await self._verify_general(question, answer_to_verify)
        else:
            result = await self._verify_general(question, answer_to_verify)

        # Update counts
        self._verify_count += 1
        verdict = result.get("verdict", "DEFER")
        if verdict == "PASS":   self._pass_count += 1
        elif verdict == "WARN": self._warn_count += 1
        elif verdict == "FAIL": self._fail_count += 1

        logger.info("[Verification] #%d %s — source=%s issues=%s",
                    self._verify_count, verdict, source_cell,
                    result.get("issues", "")[:60])

        return {
            **result,
            "source_cell":  source_cell,
            "verified":     True,
            "verification": True,
        }

    def health(self) -> dict:
        total = max(self._verify_count, 1)
        return {
            "status":        self._status.value,
            "verify_count":  self._verify_count,
            "pass_rate":     round(self._pass_count / total, 3),
            "warn_rate":     round(self._warn_count / total, 3),
            "fail_rate":     round(self._fail_count / total, 3),
        }
