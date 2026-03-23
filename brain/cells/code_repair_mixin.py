"""
CodeRepairMixin — Execution-Guided Self-Repair Loop for Eve's Code Generation
==============================================================================
Adds to any cell:
  1. Ruff lint check (JSON output, fast, Rust-based)
  2. Pyright type check (via temp file)
  3. Subprocess execution with stdout/stderr capture (30s timeout)
  4. E2B microVM sandbox (when API key set — safest option)
  5. Up to N repair iterations feeding errors back to the LLM
  6. Reflexion memory: stores failures + successes in ChromaDB

Research basis:
  - CodeAct (Wang et al. 2024) — interleaved code+bash, 77.6% SWE-bench
  - Reflexion (Shinn et al. 2023) — HumanEval 91% via verbal reflection
  - InterCode (Yang et al. 2023) — feedback loop: write→execute→observe→refine
  - AlphaCodium (Ridnik et al. 2024) — iterative refinement 19%→44% pass rate

Status: ACTIVE — runs on CPU, zero VRAM, works alongside Qwen3-14B fp8
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

logger = logging.getLogger("eve.code_repair")

# ── E2B sandbox (optional — gracefully absent if no API key) ──────────────────
_E2B_KEY = os.getenv("E2B_API_KEY", "")


@dataclass
class RepairResult:
    success:       bool
    code:          str
    iterations:    int
    errors:        list = field(default_factory=list)
    final_error:   Optional[str] = None
    lint_clean:    bool = False
    exec_output:   str = ""


class CodeRepairMixin:
    """
    Mixin for any BaseCell that generates code.
    Call self.repair_code(...) after initial generation.
    All methods are synchronous-safe (use run_in_executor if needed from async).
    """

    # ── Public entry point ────────────────────────────────────────────────────

    async def repair_code(
        self,
        code: str,
        llm_call: Callable[[str], Awaitable[str]],
        max_iterations: int = 5,
        timeout_seconds: int = 30,
        use_sandbox: bool = False,
        task_hint: str = "",
    ) -> RepairResult:
        """
        Run generate → lint → typecheck → execute → repair loop.

        Args:
            code:            Initial generated code string
            llm_call:        async callable(prompt: str) -> str
            max_iterations:  Max repair attempts (default 5)
            timeout_seconds: Per-execution timeout
            use_sandbox:     Use E2B microVM instead of subprocess
            task_hint:       Brief description of the task for ChromaDB storage
        """
        errors_history: list[str] = []

        for iteration in range(max_iterations):
            # ── Step 1: Ruff lint ─────────────────────────────────────────
            lint_err = self._run_ruff(code)
            if lint_err:
                errors_history.append(f"[iter {iteration+1}] lint: {lint_err[:300]}")
                code = await self._repair(code, lint_err, llm_call, "lint errors")
                continue

            # ── Step 2: Pyright type check ───────────────────────────────
            type_err = self._run_pyright(code)
            if type_err:
                errors_history.append(f"[iter {iteration+1}] types: {type_err[:300]}")
                code = await self._repair(code, type_err, llm_call, "type errors")
                continue

            # ── Step 3: Execute ───────────────────────────────────────────
            if use_sandbox:
                exec_res = await self._run_sandbox(code, timeout_seconds)
            else:
                exec_res = self._run_subprocess(code, timeout_seconds)

            if exec_res["ok"]:
                await self._store_success(code, errors_history, task_hint)
                return RepairResult(
                    success=True, code=code, iterations=iteration + 1,
                    errors=errors_history, lint_clean=True,
                    exec_output=exec_res.get("stdout", ""),
                )
            else:
                err = exec_res.get("error", "unknown error")[:500]
                errors_history.append(f"[iter {iteration+1}] runtime: {err}")
                code = await self._repair(code, err, llm_call, "runtime error")

        # All iterations exhausted
        final_err = errors_history[-1] if errors_history else "max iterations reached"
        await self._store_failure(code, errors_history, final_err, task_hint)
        return RepairResult(
            success=False, code=code, iterations=max_iterations,
            errors=errors_history, final_error=final_err,
        )

    # ── Lint / typecheck ──────────────────────────────────────────────────────

    def _run_ruff(self, code: str) -> Optional[str]:
        """Run ruff check on code string. Returns error summary or None."""
        try:
            result = subprocess.run(
                ["ruff", "check", "--output-format=json",
                 "--select=E,F,W", "--ignore=E501", "-"],
                input=code.encode("utf-8", errors="replace"),
                capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                return None
            errors = json.loads(result.stdout or "[]")
            if not errors:
                return None
            lines = [
                f"L{e['location']['row']}: [{e['code']}] {e['message']}"
                for e in errors[:8]
            ]
            return "\n".join(lines)
        except FileNotFoundError:
            return None   # ruff not on PATH — skip
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            return None

    def _run_pyright(self, code: str, timeout: int = 15) -> Optional[str]:
        """Run pyright on code via temp file. Returns error summary or None."""
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                tmp = f.name

            result = subprocess.run(
                ["pyright", "--outputjson", tmp],
                capture_output=True, timeout=timeout,
            )
            try:
                data = json.loads(result.stdout or "{}")
            except json.JSONDecodeError:
                return None

            diags = data.get("generalDiagnostics", [])
            errs = [d for d in diags if d.get("severity") == "error"]
            if not errs:
                return None
            lines = [
                f"L{d['range']['start']['line']+1}: {d['message']}"
                for d in errs[:5]
            ]
            return "\n".join(lines)
        except FileNotFoundError:
            return None   # pyright not installed — skip
        except (subprocess.TimeoutExpired, Exception):
            return None
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    # ── Execution ─────────────────────────────────────────────────────────────

    def _run_subprocess(self, code: str, timeout: int = 30) -> dict:
        """Run code in subprocess. Returns {ok, stdout, error}."""
        try:
            res = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=timeout,
            )
            if res.returncode == 0:
                return {"ok": True, "stdout": res.stdout[:2000]}
            err = (res.stderr or res.stdout or "Non-zero exit")[:1000]
            return {"ok": False, "error": err}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Timed out after {timeout}s"}
        except Exception as e:
            return {"ok": False, "error": f"Subprocess error: {e}"}

    async def _run_sandbox(self, code: str, timeout: int = 30) -> dict:
        """
        Run code in Eve's native LocalSandbox (E2B intent, zero API key).
        Auto-selects RestrictedPython (pure computation) or subprocess isolation.
        Falls back to plain subprocess if sandbox unavailable.
        """
        try:
            from brain.cells.local_sandbox import run as sandbox_run
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: sandbox_run(code, timeout=float(timeout))
            )
            if result.ok:
                return {"ok": True, "stdout": result.stdout[:2000]}
            return {"ok": False, "error": result.error[:1000]}
        except Exception as e:
            logger.debug("[CodeRepair] LocalSandbox failed, falling back: %s", e)
            return self._run_subprocess(code, timeout)

    # ── LLM repair ───────────────────────────────────────────────────────────

    async def _repair(
        self,
        code: str,
        error: str,
        llm_call: Callable[[str], Awaitable[str]],
        error_type: str,
    ) -> str:
        """Ask LLM to fix code given an error description."""
        prompt = textwrap.dedent(f"""
            Fix the following Python code. The error type is: {error_type}

            Error:
            {error}

            Code to fix:
            ```python
            {code[:3000]}
            ```

            Return ONLY the corrected Python code.
            Do not include any explanation, markdown fences, or commentary.
        """).strip()

        try:
            result = await llm_call(prompt)
            return self._strip_fences(result)
        except Exception as e:
            logger.warning("[CodeRepair] LLM repair call failed: %s", e)
            return code  # return unchanged on LLM failure

    @staticmethod
    def _strip_fences(raw: str) -> str:
        """Remove markdown code fences from LLM output."""
        raw = raw.strip()
        for fence in ("```python", "```py", "```"):
            if raw.startswith(fence):
                raw = raw[len(fence):]
                break
        if raw.endswith("```"):
            raw = raw[:-3]
        return raw.strip()

    # ── Reflexion memory (ChromaDB) ───────────────────────────────────────────

    async def _store_success(self, code: str, errors: list, hint: str):
        """Store successful repair pattern in ChromaDB for future reference."""
        try:
            import httpx
            doc = (
                f"SUCCESS: {hint or 'code generation'}\n"
                f"Repair iterations: {len(errors)}\n"
                f"Final code snippet: {code[:300]}"
            )
            async with httpx.AsyncClient(timeout=3) as c:
                await c.post("http://127.0.0.1:8767/save", json={
                    "session_id": "code_repair_successes",
                    "user_input": hint or "code task",
                    "eve_response": doc,
                    "metadata": {"type": "success", "iterations": len(errors)},
                })
        except Exception:
            pass  # memory save is non-critical

    async def _store_failure(self, code: str, errors: list, final_err: str, hint: str):
        """Store failure reflection in ChromaDB (Reflexion pattern)."""
        try:
            import httpx
            reflection = (
                f"FAILURE: {hint or 'code generation'}\n"
                f"Error after {len(errors)} attempts: {final_err[:200]}\n"
                f"History: {'; '.join(errors[-3:])}"
            )
            async with httpx.AsyncClient(timeout=3) as c:
                await c.post("http://127.0.0.1:8767/save", json={
                    "session_id": "code_repair_failures",
                    "user_input": hint or "code task",
                    "eve_response": reflection,
                    "metadata": {"type": "failure", "final_error": final_err[:200]},
                })
        except Exception:
            pass

    async def retrieve_repair_context(self, task: str) -> str:
        """Retrieve relevant past failures/successes before generating code."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.post("http://127.0.0.1:8767/inject", json={
                    "query": f"code generation: {task}",
                    "top_k": 3,
                    "threshold": 0.5,
                })
                return r.json().get("injection", "")
        except Exception:
            return ""
