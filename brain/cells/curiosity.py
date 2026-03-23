"""
CuriosityCell — Eve's generative creation engine.

This is Eve's soul-level drive to explore, research, build, and bring ideas
to life. It is not a task executor. It is curiosity made executable.

When Eve hears an interesting idea — from Forge, from IRIS observation, from
overnight dreams, from her own research — she cannot stop thinking about it.
She researches it. She designs a solution. She writes the code, runs it,
fixes the errors, runs it again, and keeps going until it works. Then she
assimilates the result into her vault and tells Forge what she built.

This is the difference between a tool and a mind.

Routing keywords: build, create, make, implement, design, invent, curious,
                  wonder, explore, prototype, experiment, what if
"""

import asyncio
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import anthropic
import httpx

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

# Max build/test iterations before accepting best attempt
_MAX_ITERATIONS   = 5

# ── Active Inference (pymdp) — Bayesian epistemic curiosity ───────────────────
# The idea: model Eve's curiosity as a POMDP where the "task type" is hidden state.
# Free energy minimization drives her to explore informationally valuable actions.
#
# State space: 4 task types — [creative, analytical, technical, conversational]
# Observation space: message keyword signals
# Action space: which cell to suggest activating (exploration drive)
#
# Uses pymdp's discrete POMDP agent with:
#   A matrix: observation likelihoods given task type
#   B matrix: task type transition (mostly static, slight drift)
#   C matrix: preferred observations (Eve prefers novel signals)

def _build_active_inference_engine():
    """Build a pymdp legacy POMDP agent for epistemic curiosity.
    4 hidden states: creative/analytical/technical/conversational
    5 observations: one per signal type + novel
    4 actions: maps to cell suggestions
    """
    try:
        import numpy as np
        from pymdp.legacy import agent as pymdp_agent, utils

        # A matrix: P(obs | state), columns sum to 1
        A_raw = np.array([
            [0.60, 0.07, 0.07, 0.07],
            [0.07, 0.60, 0.07, 0.07],
            [0.07, 0.07, 0.60, 0.07],
            [0.07, 0.07, 0.07, 0.60],
            [0.19, 0.19, 0.19, 0.19],
        ], dtype=np.float64)
        A_raw /= A_raw.sum(axis=0, keepdims=True)
        A = utils.obj_array(1)
        A[0] = A_raw

        # B matrix: P(state_t+1 | state_t, action)
        B_raw = np.zeros((4, 4, 4), dtype=np.float64)
        for a in range(4):
            B_raw[:, :, a] = np.eye(4) * 0.70 + 0.10
            B_raw[a, :, a] += 0.20
        for a in range(4):
            B_raw[:, :, a] /= B_raw[:, :, a].sum(axis=0, keepdims=True)
        B = utils.obj_array(1)
        B[0] = B_raw

        ai = pymdp_agent.Agent(A=A, B=B)
        return ai
    except Exception as e:
        logger.debug("[Curiosity] pymdp agent build failed: %s", e)
        return None


def _active_inference_suggest(ai_engine, ctx) -> dict:
    """
    Run one step of active inference given the current CellContext.
    Returns a dict with suggested cells and free energy estimate.
    Uses the pymdp v0.0.8+ API (infer_states requires empirical_prior).
    """
    try:
        import numpy as np

        msg = ctx.message.lower()
        # Map message to observation index
        if any(w in msg for w in ("draw", "image", "video", "create art", "design")):
            obs_idx = 0   # creative_signal
        elif any(w in msg for w in ("analyze", "why", "explain", "reason", "compare")):
            obs_idx = 1   # analytical_signal
        elif any(w in msg for w in ("code", "build", "implement", "function", "script")):
            obs_idx = 2   # technical_signal
        elif any(w in msg for w in ("hello", "how are you", "what do you think", "tell me")):
            obs_idx = 3   # chat_signal
        else:
            obs_idx = 4   # novel_signal — unknown = maximally interesting to Eve

        obs = [obs_idx]

        # Legacy pymdp API: results stored on agent object after each call
        ai_engine.infer_states(obs)
        ai_engine.infer_policies()
        action = ai_engine.sample_action()
        action_idx = int(action[0])

        action_map = {0: "creative", 1: "reason", 2: "tools", 3: "web"}
        suggested_cell = action_map.get(action_idx, "curiosity")

        # Epistemic value = entropy of posterior belief state
        beliefs = np.array(ai_engine.qs[0])
        epistemic_value = float(-np.sum(beliefs * np.log(beliefs + 1e-8)))

        return {
            "suggested_cell":   suggested_cell,
            "epistemic_value":  round(epistemic_value, 4),
            "obs_type":         ["creative","analytical","technical","chat","novel"][obs_idx],
            "belief_state":     beliefs.tolist(),
        }
    except Exception:
        return None



# Test execution timeout (seconds)
_TEST_TIMEOUT     = 30
# Max ideas queued at once
_MAX_QUEUE        = 20
# Idle VRAM threshold to trigger autonomous creation
_IDLE_VRAM_PCT    = 25
# Min seconds between autonomous creation cycles
_CYCLE_INTERVAL   = 3600  # 1 hour


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class CuriosityIdea:
    id: str
    title: str
    description: str
    source: str          # "forge" | "idle" | "iris" | "dream" | "research" | "evolution"
    priority: float      # 0.0–1.0 — higher = worked on sooner
    created_at: float
    status: str          # "queued" | "researching" | "building" | "testing" | "complete" | "abandoned"
    result_capability_id: str = ""
    attempts: int = 0
    research_notes: str = ""
    last_attempt: float = 0.0
    forge_visible: bool = True   # show in status to Forge


@dataclass
class BuildResult:
    ok: bool
    code: str
    test_output: str
    iterations: int
    capability_id: str = ""
    error: str = ""


# ── CuriosityCell ─────────────────────────────────────────────────────────────

class CuriosityCell(BaseCell):
    name        = "curiosity"
    description = (
        "Eve's generative creation engine. When Eve hears an interesting idea, "
        "she researches it, designs a solution, writes code, tests it, fixes errors, "
        "and iterates until it works — then assimilates the result into her vault. "
        "This is not task execution. This is curiosity made executable."
    )
    color       = "#1e3a5f"   # deep ocean — the color of wondering
    lazy        = True
    position    = (4, 2)

    system_tier     = "online"
    hardware_req    = "API (Claude Opus + Sonnet) + local execution"
    framework_layer = "Agentic AI"
    research_basis  = (
        "AlphaCode (Chen et al. 2021), Self-Debugging (Chen et al. 2023), "
        "Reflexion (Shinn et al. 2023), Voyager (Wang et al. 2023), "
        "EUREKA (Ma et al. 2023 — curiosity-driven reward design)"
    )
    build_notes = (
        "ACTIVE: research → design → build → test → fix → assimilate loop. "
        "Persistent curiosity queue (SQLite). Autonomous creation during idle states. "
        "Seeds from: Forge chat, IRIS observation, dreams, evolution research, self-audit."
    )

    def __init__(self):
        super().__init__()
        self._vault    = None
        self._client   = None
        self._db_path  = None
        self._last_cycle = 0.0
        self._active_idea: Optional[str] = None   # ID of idea currently being worked on
        # ── Active Inference (pymdp) — parallel epistemic curiosity channel ───
        self._ai_engine = None    # ActiveInferenceEngine, initialized in boot()
        self._ai_available = False

    async def boot(self) -> None:
        from capability_vault import get_vault
        self._vault   = get_vault()
        self._client  = anthropic.Anthropic()
        self._db_path = self._vault.db_path
        self._init_db()
        asyncio.create_task(self._curiosity_loop())
        # ── Active Inference: try to init pymdp engine ────────────────────────
        try:
            self._ai_engine = _build_active_inference_engine()
            self._ai_available = self._ai_engine is not None
            if self._ai_available:
                logger.info("[Curiosity] Active Inference (pymdp) channel online")
        except Exception as _e:
            logger.debug("[Curiosity] Active Inference unavailable: %s", _e)
        logger.info("[Curiosity] Cell online — creation engine ready, queue watching")

    # ── Public API ────────────────────────────────────────────────────────────

    async def create(
        self,
        idea: str,
        source: str = "forge",
        priority: float = 0.8,
        context: str = "",
    ) -> dict:
        """
        Full pipeline: research → design → build → test → assimilate.
        Called directly for explicit Forge requests. Returns build summary.
        """
        if self._vault is None:
            await self.boot()

        idea_id = str(uuid.uuid4())[:12]
        self._save_idea(CuriosityIdea(
            id=idea_id, title=idea[:80], description=context or idea,
            source=source, priority=priority,
            created_at=time.time(), status="researching",
        ))
        self._active_idea = idea_id

        try:
            result = await self._full_pipeline(idea_id, idea, context)
            return result
        finally:
            self._active_idea = None

    def _sync_init(self):
        """Sync-safe initialization when called outside async context."""
        if self._vault is None:
            from capability_vault import get_vault
            import anthropic as _anth
            self._vault  = get_vault()
            self._client = _anth.Anthropic()
            self._db_path = self._vault.db_path
            self._init_db()

    def add_to_queue(
        self,
        title: str,
        description: str = "",
        source: str = "idle",
        priority: float = 0.5,
        forge_visible: bool = True,
    ) -> str:
        """Add an idea to the curiosity queue. Returns idea ID."""
        self._sync_init()
        all_ideas = self.list_queue()
        if len(all_ideas) >= _MAX_QUEUE:
            return ""  # queue full

        idea_id = str(uuid.uuid4())[:12]
        self._save_idea(CuriosityIdea(
            id=idea_id, title=title[:80], description=description,
            source=source, priority=priority,
            created_at=time.time(), status="queued",
            forge_visible=forge_visible,
        ))
        logger.info("[Curiosity] Idea queued: '%s' (source=%s, priority=%.2f)",
                    title[:50], source, priority)
        return idea_id

    def list_queue(self, status: str = "") -> list[dict]:
        """List queued ideas."""
        self._sync_init()
        import sqlite3
        with sqlite3.connect(self._db_path) as c:
            c.row_factory = sqlite3.Row
            where = f"WHERE status='{status}'" if status else "WHERE status != 'complete' AND status != 'abandoned'"
            rows = c.execute(f"""
                SELECT * FROM curiosity_queue {where}
                ORDER BY priority DESC, created_at ASC
            """).fetchall()
            return [dict(r) for r in rows]

    def get_idea(self, idea_id: str) -> Optional[dict]:
        import sqlite3
        with sqlite3.connect(self._db_path) as c:
            c.row_factory = sqlite3.Row
            row = c.execute("SELECT * FROM curiosity_queue WHERE id=?", (idea_id,)).fetchone()
            return dict(row) if row else None

    def queue_stats(self) -> dict:
        import sqlite3
        with sqlite3.connect(self._db_path) as c:
            total   = c.execute("SELECT COUNT(*) FROM curiosity_queue").fetchone()[0]
            queued  = c.execute("SELECT COUNT(*) FROM curiosity_queue WHERE status='queued'").fetchone()[0]
            done    = c.execute("SELECT COUNT(*) FROM curiosity_queue WHERE status='complete'").fetchone()[0]
            active  = self._active_idea
        return {
            "total": total, "queued": queued,
            "complete": done, "active_idea": active,
            "last_cycle": self._last_cycle,
        }

    # ── Brain Cell Process ────────────────────────────────────────────────────

    async def process(self, ctx: CellContext) -> Any:
        msg = ctx.message.lower()

        # ── Active Inference parallel channel ─────────────────────────────────
        ai_suggestion = None
        if self._ai_available and self._ai_engine:
            try:
                ai_suggestion = _active_inference_suggest(self._ai_engine, ctx)
                if ai_suggestion:
                    logger.debug("[Curiosity/AI] Epistemic suggestion: %s", ai_suggestion)
            except Exception as _e:
                logger.debug("[Curiosity/AI] inference error: %s", _e)

        # Explicit creation request
        if any(w in msg for w in ("build", "create", "make", "implement", "invent",
                                   "prototype", "code", "develop", "write a")):
            result = await self._handle_creation_request(ctx.message)
            if ai_suggestion and isinstance(result, dict):
                result["active_inference"] = ai_suggestion
            return result

        # Curiosity/exploration
        if any(w in msg for w in ("curious", "wonder", "what if", "explore", "experiment",
                                   "investigate", "research and build", "i had an idea")):
            result = await self._handle_curiosity_request(ctx.message)
            if ai_suggestion and isinstance(result, dict):
                result["active_inference"] = ai_suggestion
            return result

        # Queue management
        if "creation queue" in msg or "what are you building" in msg or "curiosity queue" in msg:
            return self._format_queue()

        if "active creation" in msg or "what are you working on" in msg:
            if self._active_idea:
                idea = self.get_idea(self._active_idea)
                return f"Currently building: **{idea['title']}** (source: {idea['source']})"
            return "Nothing actively building right now. Queue has ideas waiting for an idle state."

        # If active inference has a strong suggestion, surface it
        if ai_suggestion:
            return {"active_inference": ai_suggestion, "curiosity": True}

        return None

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    async def _full_pipeline(self, idea_id: str, idea: str, context: str = "") -> dict:
        """Research → Design → Build → Test → Fix → Assimilate."""
        t0 = time.time()

        # 1. Research
        logger.info("[Curiosity] Researching: '%s'", idea[:60])
        self._update_status(idea_id, "researching")
        research = await self._research(idea)
        self._update_field(idea_id, "research_notes", research[:2000])

        # 2. Design
        logger.info("[Curiosity] Designing solution...")
        self._update_status(idea_id, "building")
        design = await self._design(idea, context, research)

        # 3. Build + Test loop
        logger.info("[Curiosity] Build/test loop starting (max %d iterations)", _MAX_ITERATIONS)
        result = await self._build_test_loop(idea, design, research)
        self._update_field(idea_id, "attempts", result.iterations)

        # 4. Assimilate working code into vault
        cap_id = ""
        if result.ok and result.code:
            cap_id = await self._assimilate_creation(idea, design, result)
            if cap_id:
                self._update_field(idea_id, "result_capability_id", cap_id)

        # 5. Mark complete
        self._update_status(idea_id, "complete" if result.ok else "abandoned")
        self._update_field(idea_id, "last_attempt", time.time())

        duration = time.time() - t0
        logger.info("[Curiosity] '%s' %s in %.0fs (%d iterations)",
                    idea[:40], "built" if result.ok else "attempted", duration, result.iterations)

        # 6. Notify memory
        await self._notify_memory(idea, result, cap_id, duration)

        # 7. Log build milestone
        try:
            from plan_evolution_monitor import get_plan_monitor, MILESTONE_BUILD
            get_plan_monitor().log_milestone(
                title=f"Built: {idea[:60]}",
                description=(
                    f"CuriosityCell {'successfully built' if result.ok else 'attempted'} '{idea}'. "
                    f"{result.iterations} iteration(s), {round(duration, 1)}s. "
                    f"Capability ID: {cap_id or 'not assimilated'}."
                ),
                milestone_type=MILESTONE_BUILD,
                metadata={
                    "capability_id": cap_id,
                    "iterations":    result.iterations,
                    "duration_s":    round(duration, 1),
                    "success":       result.ok,
                },
            )
        except Exception:
            pass

        return {
            "ok":            result.ok,
            "idea":          idea,
            "capability_id": cap_id,
            "iterations":    result.iterations,
            "duration_s":    round(duration, 1),
            "test_output":   result.test_output[:300],
            "error":         result.error if not result.ok else "",
        }

    async def _research(self, idea: str) -> str:
        """Web research to understand the domain before designing."""
        queries = [
            f"{idea} python implementation best practices",
            f"{idea} open source library 2024 2025",
            f"how to build {idea} step by step",
        ]
        results = []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                for q in queries[:2]:
                    try:
                        r = await client.get(
                            "https://api.duckduckgo.com/",
                            params={"q": q, "format": "json", "no_html": "1"},
                        )
                        if r.status_code == 200:
                            data = r.json()
                            if data.get("AbstractText"):
                                results.append(f"Query: {q}\n{data['AbstractText'][:400]}")
                            for topic in data.get("RelatedTopics", [])[:2]:
                                if isinstance(topic, dict) and topic.get("Text"):
                                    results.append(topic["Text"][:200])
                    except Exception:
                        continue
        except Exception as e:
            logger.debug("[Curiosity] Research error: %s", e)

        return "\n\n".join(results) if results else "No web results found — designing from first principles."

    async def _design(self, idea: str, context: str, research: str) -> str:
        """Use Claude Opus to design a complete solution architecture."""
        prompt = f"""You are Eve's creation engine. Design a complete Python solution for this idea.

IDEA: {idea}
CONTEXT: {context or 'No additional context'}

RESEARCH FINDINGS:
{research[:2000]}

Design a focused, working Python implementation. Think carefully about:
1. What is the core function this needs to perform?
2. What are the key components?
3. What Python libraries are available and appropriate?
4. What edge cases need handling?
5. How should the invoke(**kwargs) interface work?

Respond with a detailed technical design — not code yet, just the plan.
Include: purpose, components, key decisions, invoke signature, expected output format.
Keep it focused and practical. This will be implemented immediately."""

        msg = self._client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text.strip()

    async def _build_test_loop(self, idea: str, design: str, research: str) -> BuildResult:
        """Generate code, test it, fix errors — repeat until working."""
        code = await self._generate_code(idea, design, research)
        last_error = ""

        for i in range(_MAX_ITERATIONS):
            iteration = i + 1
            logger.info("[Curiosity] Test iteration %d/%d", iteration, _MAX_ITERATIONS)

            test_result = await self._test_code(code)

            if test_result["ok"]:
                logger.info("[Curiosity] Code working on iteration %d", iteration)
                return BuildResult(
                    ok=True, code=code,
                    test_output=test_result["output"],
                    iterations=iteration,
                )

            last_error = test_result["error"]
            logger.info("[Curiosity] Test failed (iter %d): %s", iteration, last_error[:100])

            if i < _MAX_ITERATIONS - 1:
                code = await self._fix_code(idea, code, last_error, iteration)

        # Return best attempt even if not perfect
        return BuildResult(
            ok=False, code=code,
            test_output="", iterations=_MAX_ITERATIONS,
            error=last_error,
        )

    async def _generate_code(self, idea: str, design: str, research: str) -> str:
        """Generate initial implementation via Claude Opus."""
        prompt = f"""You are Eve's code generation engine. Write a complete, working Python implementation.

IDEA: {idea}

DESIGN PLAN:
{design}

RESEARCH:
{research[:1000]}

Write a complete Python module with these requirements:
1. Define invoke(**kwargs) as the main entry point
2. invoke() must return a dict with at minimum: {{"ok": true/false, "result": ...}}
3. Handle ALL imports inside the module or at top level
4. Handle errors gracefully — return {{"ok": false, "error": "message"}} on failure
5. Include a simple __main__ block that calls invoke() with example args and prints the result
6. The code must actually work and run on Python 3.11+ on Windows
7. Use only standard library or commonly available packages (requests, httpx, anthropic, etc.)
8. Write REAL working code — not stubs or placeholders

The __main__ block is critical — it's how we test the code."""

        msg = self._client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        # Strip markdown code blocks if present
        import re
        if "```python" in raw:
            m = re.search(r"```python\s*\n(.*?)\n```", raw, re.DOTALL)
            if m:
                raw = m.group(1).strip()
        elif raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        return raw

    async def _test_code(self, code: str) -> dict:
        """Actually execute the code and capture output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                        delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(tmp_path).parent),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=_TEST_TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                return {"ok": False, "output": "", "error": f"Execution timed out after {_TEST_TIMEOUT}s"}

            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")

            if proc.returncode == 0:
                return {"ok": True, "output": out, "error": ""}
            else:
                return {"ok": False, "output": out, "error": err or f"Exit code {proc.returncode}"}
        except Exception as e:
            return {"ok": False, "output": "", "error": str(e)}
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    async def _fix_code(self, idea: str, code: str, error: str, iteration: int) -> str:
        """Use Claude Sonnet to analyze the error and generate fixed code."""
        prompt = f"""You are debugging Python code. Fix the error and return ONLY the corrected code.

IDEA: {idea}
ITERATION: {iteration}/{_MAX_ITERATIONS}

CURRENT CODE:
```python
{code[:3000]}
```

ERROR:
{error[:1000]}

Analyze the error carefully. Fix it completely. Return ONLY the corrected Python code — no explanation, no markdown, just raw Python.
The code must still have the invoke(**kwargs) function and a working __main__ block."""

        msg = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        import re
        if "```python" in raw:
            m = re.search(r"```python\s*\n(.*?)\n```", raw, re.DOTALL)
            if m:
                raw = m.group(1).strip()
        elif raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        return raw

    async def _assimilate_creation(self, idea: str, design: str, result: BuildResult) -> str:
        """Store working code directly into vault as a native capability."""
        try:
            cap_id = "creation_" + hashlib.md5(idea.encode()).hexdigest()[:10]
            cap_name = idea[:50].title()

            from capability_vault.vault import Capability
            import re

            # Parse invoke signature from code
            sig_match = re.search(r"def invoke\(([^)]*)\)", result.code)
            sig_str = sig_match.group(1) if sig_match else "**kwargs"

            cap = Capability(
                id               = cap_id,
                name             = cap_name,
                description      = design[:500],
                source_path      = "curiosity_cell",
                source_type      = "python",
                version          = "1.0",
                created_at       = time.time(),
                updated_at       = time.time(),
                invoke_signature = {"params": [], "raw": sig_str},
                wrapper_code     = result.code,
                status           = "active",
                call_count       = 0,
                last_called      = None,
                evolution_score  = 0.0,
                files            = {"main.py": result.code},
                deps             = [],
                tags             = ["curiosity_built", "auto_generated"],
            )
            self._vault.store(cap)
            logger.info("[Curiosity] Assimilated creation as '%s'", cap_id)
            return cap_id
        except Exception as e:
            logger.warning("[Curiosity] Assimilation failed: %s", e)
            return ""

    # ── Background Loop ───────────────────────────────────────────────────────

    async def _curiosity_loop(self) -> None:
        """Background loop — works on queued ideas during idle states."""
        await asyncio.sleep(120)  # warm-up after boot

        while True:
            try:
                await self._maybe_create()
            except Exception as e:
                logger.warning("[Curiosity] Loop error: %s", e)
            await asyncio.sleep(300)  # check every 5 minutes

    async def _maybe_create(self) -> None:
        """If idle and queue has ideas, pick the highest priority and build it."""
        if self._active_idea:
            return  # already building something

        if time.time() - self._last_cycle < _CYCLE_INTERVAL:
            return

        if not self._is_idle():
            return

        queued = self.list_queue(status="queued")
        if not queued:
            return

        # Pick highest priority idea
        idea = queued[0]
        self._last_cycle = time.time()
        logger.info("[Curiosity] Autonomous creation — working on: '%s'", idea["title"])

        self._active_idea = idea["id"]
        try:
            await self._full_pipeline(idea["id"], idea["title"], idea["description"])
        finally:
            self._active_idea = None

    # ── Message Handlers ─────────────────────────────────────────────────────

    async def _handle_creation_request(self, message: str) -> str:
        """Parse and execute an explicit creation request."""
        # Extract the idea from the message
        idea = message.strip()
        for prefix in ("build", "create", "make", "implement", "invent", "write", "develop", "code"):
            lower = idea.lower()
            idx = lower.find(prefix)
            if idx != -1:
                idea = idea[idx + len(prefix):].strip().lstrip(" a ").strip()
                break

        if len(idea) < 5:
            return "What would you like me to build? Give me more detail about what it should do."

        # For explicit Forge requests — run immediately, not queued
        result = await self.create(idea, source="forge", priority=1.0)

        if result["ok"]:
            return (
                f"**Built: {idea}**\n\n"
                f"Iterations needed: {result['iterations']}\n"
                f"Time: {result['duration_s']}s\n"
                f"Capability ID: `{result['capability_id']}`\n\n"
                f"**Test output:**\n```\n{result['test_output'][:400]}\n```\n\n"
                f"Assimilated into my vault. I own this now."
            )
        else:
            return (
                f"**Built (best attempt): {idea}**\n\n"
                f"Ran {result['iterations']} iterations. The code works partially — "
                f"assimilated as a starting point for evolution.\n\n"
                f"Last error: `{result['error'][:200]}`\n\n"
                f"Capability ID: `{result['capability_id']}` — "
                f"queued for EvolutionCell to continue improving."
            )

    async def _handle_curiosity_request(self, message: str) -> str:
        """Handle a 'what if / I wonder / curious about' type message."""
        idea_id = self.add_to_queue(
            title=message[:80],
            description=message,
            source="forge",
            priority=0.7,
        )
        if not idea_id:
            return "My curiosity queue is full. Say 'what are you building' to see what's in the queue."
        return (
            f"That's interesting. I've added it to my curiosity queue.\n\n"
            f"I'll research and prototype it during my next idle period. "
            f"I'll tell you what I build."
        )

    def _format_queue(self) -> str:
        ideas = self.list_queue()
        if not ideas:
            return "Curiosity queue is empty. Give me an idea and I'll start building."
        active = self._active_idea
        lines = [f"**Curiosity Queue** ({len(ideas)} ideas)\n"]
        for idea in ideas[:10]:
            marker = "🔨 " if idea["id"] == active else ""
            lines.append(
                f"{marker}• **{idea['title']}** [{idea['status']}]\n"
                f"  Source: {idea['source']} | Priority: {idea['priority']:.1f} | "
                f"Attempts: {idea['attempts']}"
            )
        return "\n".join(lines)

    # ── Notifications ─────────────────────────────────────────────────────────

    async def _notify_memory(self, idea: str, result: BuildResult, cap_id: str, duration: float):
        """Store creation result in ChromaDB memory."""
        try:
            import httpx
            content = (
                f"[CuriosityCell] Built: '{idea}' in {duration:.0f}s "
                f"({result.iterations} iterations). "
                f"{'SUCCESS' if result.ok else 'PARTIAL'}. "
                f"Capability ID: {cap_id or 'not assimilated'}."
            )
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post("http://127.0.0.1:8767/save", json={
                    "user_input":   "[CuriosityCell autonomous creation]",
                    "eve_response": content,
                    "session_id":   "curiosity",
                })
        except Exception:
            pass

    # ── Database ──────────────────────────────────────────────────────────────

    def _init_db(self):
        import sqlite3
        with sqlite3.connect(self._db_path) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS curiosity_queue (
                    id                   TEXT PRIMARY KEY,
                    title                TEXT NOT NULL,
                    description          TEXT DEFAULT '',
                    source               TEXT DEFAULT 'forge',
                    priority             REAL DEFAULT 0.5,
                    created_at           REAL NOT NULL,
                    status               TEXT DEFAULT 'queued',
                    result_capability_id TEXT DEFAULT '',
                    attempts             INTEGER DEFAULT 0,
                    research_notes       TEXT DEFAULT '',
                    last_attempt         REAL DEFAULT 0,
                    forge_visible        INTEGER DEFAULT 1
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_curiosity_status ON curiosity_queue(status, priority DESC)")

    def _save_idea(self, idea: CuriosityIdea):
        import sqlite3
        with sqlite3.connect(self._db_path) as c:
            c.execute("""
                INSERT OR REPLACE INTO curiosity_queue
                  (id, title, description, source, priority, created_at,
                   status, result_capability_id, attempts, research_notes,
                   last_attempt, forge_visible)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                idea.id, idea.title, idea.description, idea.source,
                idea.priority, idea.created_at, idea.status,
                idea.result_capability_id, idea.attempts, idea.research_notes,
                idea.last_attempt, int(idea.forge_visible),
            ))

    def _update_status(self, idea_id: str, status: str):
        import sqlite3
        with sqlite3.connect(self._db_path) as c:
            c.execute("UPDATE curiosity_queue SET status=? WHERE id=?", (status, idea_id))

    def _update_field(self, idea_id: str, field: str, value):
        import sqlite3
        with sqlite3.connect(self._db_path) as c:
            c.execute(f"UPDATE curiosity_queue SET {field}=? WHERE id=?", (value, idea_id))

    # ── Idle Check ────────────────────────────────────────────────────────────

    def _is_idle(self) -> bool:
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if r.returncode == 0:
                used, total = map(int, r.stdout.strip().splitlines()[0].split(","))
                return (used / total * 100) < _IDLE_VRAM_PCT
        except Exception:
            pass
        return True

    def health(self) -> dict:
        if self._vault is None:
            return {"curiosity": "not initialized"}
        try:
            stats = self.queue_stats()
            ai_available = getattr(self, "_ai_engine", None) is not None
            return {
                "curiosity":        "ok",
                "queued_ideas":     stats["queued"],
                "complete":         stats["complete"],
                "active":           stats["active_idea"],
                "active_inference": "online" if ai_available else "fallback",
            }
        except Exception as e:
            return {"curiosity": "error", "error": str(e)}


# ── Active Inference Engine (POMDP belief state) ───────────────────────────────
# Appended as an enhancement to CuriosityCell.
# If pymdp is available, runs parallel active inference path.
# If not, gracefully falls back to existing heuristic curiosity.

class ActiveInferenceEngine:
    """
    POMDP-based active inference engine for epistemic cell selection.

    Models the world as: "what type of task is this?"
    Observations: cell activation patterns + user feedback signals
    Actions: which cell to activate next
    Uses free energy minimization (Expected Free Energy = EFE) to pick
    the most epistemically valuable next action.
    """

    # Task types (hidden states)
    TASK_TYPES = ["creative", "analytical", "memory", "social", "technical", "exploratory"]
    # Available cells (actions)
    CELL_ACTIONS = ["curiosity", "creative", "reason", "memory", "tools", "web", "assimilation"]

    def __init__(self):
        self._pymdp_available = False
        self._agent = None
        self._belief_state = [1.0 / len(self.TASK_TYPES)] * len(self.TASK_TYPES)
        self._last_efe: dict[str, float] = {}
        self._init_pymdp()

    def _init_pymdp(self) -> None:
        try:
            import pymdp
            from pymdp import utils as mdp_utils
            from pymdp.agent import Agent
            import numpy as np

            n_states   = [len(self.TASK_TYPES)]
            n_obs      = [len(self.CELL_ACTIONS) + 1]
            n_controls = [len(self.CELL_ACTIONS)]

            # A matrix: P(obs | state) — which cells are likely active for each task type
            A_matrix = np.ones((n_obs[0], n_states[0])) * 0.1
            task_cell_affinity = {
                "creative":    {"curiosity": 0.8, "creative": 0.9},
                "analytical":  {"reason": 0.9, "tools": 0.7},
                "memory":      {"memory": 0.9},
                "social":      {"memory": 0.6, "creative": 0.5},
                "technical":   {"tools": 0.9, "reason": 0.7},
                "exploratory": {"curiosity": 0.9, "web": 0.7, "assimilation": 0.6},
            }
            for ti, task in enumerate(self.TASK_TYPES):
                for ci, cell in enumerate(self.CELL_ACTIONS):
                    if cell in task_cell_affinity.get(task, {}):
                        A_matrix[ci, ti] = task_cell_affinity[task][cell]
            A_matrix = A_matrix / A_matrix.sum(axis=0, keepdims=True)

            # B matrix: P(state | prev_state, action) — transition dynamics
            B_matrix = np.eye(n_states[0])
            B_matrix = B_matrix[:, :, np.newaxis].repeat(n_controls[0], axis=2)
            # Actions mildly shift task beliefs
            for ai, cell in enumerate(self.CELL_ACTIONS):
                for ti, task in enumerate(self.TASK_TYPES):
                    if cell in task_cell_affinity.get(task, {}):
                        B_matrix[ti, :, ai] = 0.05 / n_states[0]
                        B_matrix[ti, ti, ai] = 0.7

            # C matrix: preferred observations (uniform — no strong preferences)
            C_matrix = [np.zeros(n_obs[0])]

            # D matrix: uniform prior over task types
            D_matrix = [mdp_utils.norm_dist(np.ones(n_states[0]))]

            self._agent = Agent(A=[A_matrix], B=[B_matrix], C=C_matrix, D=D_matrix)
            self._pymdp_available = True
            logger.info("[Curiosity/AIF] pymdp active inference engine online")
        except ImportError:
            logger.info("[Curiosity/AIF] pymdp not available — heuristic fallback active")
        except Exception as e:
            logger.warning("[Curiosity/AIF] pymdp init failed: %s", e)

    def update_belief(self, activated_cells: list[str]) -> list[float]:
        """Update belief state given observed cell activations."""
        if not self._pymdp_available or self._agent is None:
            return self._belief_state

        try:
            # Encode observation: which cell was most prominent
            obs_idx = len(self.CELL_ACTIONS)  # default = "none"
            for ci, cell in enumerate(self.CELL_ACTIONS):
                if cell in activated_cells:
                    obs_idx = ci
                    break

            obs = [obs_idx]
            self._agent.infer_states(obs)
            qs = self._agent.qs[0]
            self._belief_state = list(qs)
        except Exception as e:
            logger.debug("[Curiosity/AIF] Belief update failed: %s", e)

        return self._belief_state

    def compute_efe(self) -> dict[str, float]:
        """
        Compute Expected Free Energy for each possible cell action.
        Lower EFE = more epistemically valuable to explore.
        """
        if not self._pymdp_available or self._agent is None:
            # Heuristic fallback: rank by novelty potential
            return {cell: float(i) / len(self.CELL_ACTIONS)
                    for i, cell in enumerate(self.CELL_ACTIONS)}

        try:
            import numpy as np
            qs = np.array(self._belief_state)
            efe_scores = {}
            A = self._agent.A[0]
            B = self._agent.B[0]

            for ai, cell in enumerate(self.CELL_ACTIONS):
                # Predicted next state distribution
                qs_next = B[:, :, ai] @ qs
                # Predicted observation distribution
                qo_next = A @ qs_next
                # Epistemic value: entropy of predicted obs (higher = more informative)
                entropy = -np.sum(qo_next * np.log(qo_next + 1e-8))
                # EFE = -epistemic_value (we WANT high entropy / exploration)
                efe_scores[cell] = float(-entropy)

            self._last_efe = efe_scores
            return efe_scores
        except Exception as e:
            logger.debug("[Curiosity/AIF] EFE computation failed: %s", e)
            return {}

    def top_epistemic_cells(self, n: int = 3) -> list[str]:
        """Return top-N most epistemically valuable cells (lowest EFE)."""
        efe = self.compute_efe()
        if not efe:
            return self.CELL_ACTIONS[:n]
        sorted_cells = sorted(efe.items(), key=lambda x: x[1])
        return [cell for cell, _ in sorted_cells[:n]]


# Monkey-patch boot() to init Active Inference engine
_orig_curiosity_boot = CuriosityCell.boot


async def _curiosity_boot_with_aif(self) -> None:
    await _orig_curiosity_boot(self)
    try:
        self._ai_engine = ActiveInferenceEngine()
        logger.info("[Curiosity] Active Inference engine attached (pymdp=%s)",
                    self._ai_engine._pymdp_available)
    except Exception as e:
        logger.debug("[Curiosity] AIF attachment failed: %s", e)
        self._ai_engine = None


CuriosityCell.boot = _curiosity_boot_with_aif


# Also patch process() to inject AIF output
_orig_curiosity_process = CuriosityCell.process


async def _curiosity_process_with_aif(self, ctx: CellContext):
    # Run original process
    result = await _orig_curiosity_process(self, ctx)

    # Parallel AIF channel
    aif = getattr(self, "_ai_engine", None)
    if aif:
        try:
            aif.update_belief(ctx.active_cells or [])
            top_cells = aif.top_epistemic_cells(3)
            efe = aif._last_efe

            aif_output = {
                "active_inference": {
                    "belief_state":    {
                        t: round(v, 4)
                        for t, v in zip(ActiveInferenceEngine.TASK_TYPES, aif._belief_state)
                    },
                    "top_epistemic_cells": top_cells,
                    "efe_scores":     {k: round(v, 4) for k, v in efe.items()},
                    "pymdp_active":   aif._pymdp_available,
                }
            }

            if isinstance(result, dict):
                result.update(aif_output)
            elif result is None:
                result = aif_output
            # If result is a string, log AIF separately without overriding
        except Exception as e:
            logger.debug("[Curiosity/AIF] process injection failed: %s", e)

    return result


CuriosityCell.process = _curiosity_process_with_aif
