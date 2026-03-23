"""
AssimilationCell — Eve's code consumption and native integration engine.

This cell is Eve's digestive system. When pointed at any external program,
repository, or script, it:
  1. Scans and deeply reads all source files
  2. Uses Claude to understand the code's purpose and architecture
  3. Generates an Eve-native callable wrapper
  4. Stores everything in the CapabilityVault (SQLite — no external deps)
  5. Registers the capability for instant access from any other cell

After assimilation, Eve can invoke the capability natively — no file system
dependency, no re-downloading, no external process required.
She owns it completely. It is part of her.

Routing keywords: assimilate, consume, ingest, capability, integrate,
                  vault, invoke, run capability, what can you do
"""

import asyncio
import json
import logging
import time
from typing import Any

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)


class AssimilationCell(BaseCell):
    name        = "assimilation"
    description = (
        "Consumes external code and programs into Eve's native architecture. "
        "Scans source, generates callable wrappers, and stores them in the "
        "CapabilityVault — permanent, zero-dependency, fully owned by Eve."
    )
    color       = "#7c2d12"   # deep ember — the color of transformation
    lazy        = False   # always-on — invoke() must be ready from boot
    position    = (5, 1)

    system_tier     = "online"
    hardware_req    = "API only (Claude)"
    framework_layer = "Agentic AI"
    research_basis  = (
        "Code Understanding: CodeBERT, CodeT5, AlphaCode; "
        "Program Synthesis: DreamCoder, Codex; "
        "Self-modifying Systems: SELF-EVOLVE, RLEF"
    )
    build_notes = (
        "ACTIVE: full scan→analyze→wrap→vault pipeline. "
        "CapabilityVault: SQLite-backed, zero external deps at runtime. "
        "Executor: in-process module loading with ThreadPoolExecutor. "
        "Next: dependency auto-install, cross-language support (Node, bash)."
    )

    def __init__(self):
        super().__init__()
        self._vault    = None
        self._consumer = None
        self._executor = None

    async def boot(self) -> None:
        from capability_vault import get_vault
        from capability_vault.consumer import CodeConsumer
        from capability_vault.executor import get_executor
        self._vault    = get_vault()
        self._consumer = CodeConsumer()
        self._executor = get_executor()
        logger.info("[Assimilation] Cell online — vault ready at %s", self._vault.db_path)

    async def process(self, ctx: CellContext) -> Any:
        msg = ctx.message.lower()

        # Quick-route: list vault
        if any(w in msg for w in ("what can you do", "list capabilities", "vault contents", "what have you consumed")):
            return self._format_vault_list()

        # Quick-route: invoke a capability
        if "invoke" in msg or "run capability" in msg or "use capability" in msg:
            return await self._handle_invoke_request(ctx.message)

        # Main route: new assimilation request
        if any(w in msg for w in ("assimilate", "consume", "ingest", "integrate", "absorb")):
            return await self._handle_assimilation_request(ctx.message)

        # Vault stats
        if "vault stat" in msg or "assimilation stat" in msg:
            return json.dumps(self._vault.get_stats(), indent=2)

        return None  # Not our domain

    # ── Assimilation Pipeline ────────────────────────────────────────────────

    async def assimilate(
        self,
        path: str,
        custom_name: str = "",
        force: bool = False,
    ) -> dict:
        """
        Full pipeline: scan → analyze → bundle → store → register.
        Returns a summary dict.
        """
        # Lazy boot — REST calls bypass the brain manager boot flow
        if self._consumer is None:
            await self.boot()

        t0 = time.time()
        logger.info("[Assimilation] Starting assimilation of: %s", path)

        # 1. Scan
        try:
            scan = await asyncio.to_thread(self._consumer.scan, path)
        except FileNotFoundError as e:
            return {"ok": False, "error": str(e)}

        if not scan.files:
            return {"ok": False, "error": f"No readable source files found at {path}"}

        logger.info("[Assimilation] Scanned %d files", len(scan.files))

        # 2. Analyze (Claude)
        analysis = await self._consumer.analyze(scan, custom_name=custom_name)

        # 3. Check for existing capability (don't overwrite unless forced)
        existing = self._vault.get(analysis.capability_name)
        if existing and not force:
            return {
                "ok": True,
                "status": "already_assimilated",
                "capability_id": existing.id,
                "name": existing.name,
                "message": (
                    f"'{existing.name}' is already in the vault (v{existing.version}). "
                    f"Use force=True to re-assimilate."
                )
            }

        # 4. Bundle files + deps
        files = {f.rel_path: f.content for f in scan.files}
        deps  = analysis.dependencies or self._consumer.detect_deps(scan)

        # 4b. Immunity pre-store security gate
        if analysis.wrapper_code:
            try:
                immunity_cell = self._manager._cells.get("immunity") if self._manager else None
                if immunity_cell:
                    scan_result = immunity_cell.pre_store(analysis.capability_name, analysis.wrapper_code)
                    if not scan_result.get("ok"):
                        violations = scan_result.get("violations", [])
                        logger.warning(
                            "[Assimilation] BLOCKED by immunity scan — %s: %s",
                            analysis.capability_name, violations[0] if violations else "unknown"
                        )
                        return {
                            "ok": False,
                            "error": "Immunity scan blocked this capability",
                            "violations": violations,
                            "score": scan_result.get("score", 0),
                        }
            except Exception as _imm_err:
                logger.debug("[Assimilation] Immunity pre-store check skipped: %s", _imm_err)

        # 5. Store in vault
        from capability_vault.vault import Capability
        cap = Capability(
            id               = analysis.capability_name,
            name             = analysis.display_name,
            description      = analysis.description,
            source_path      = path,
            source_type      = analysis.source_type,
            version          = analysis.version,
            created_at       = t0,
            updated_at       = t0,
            invoke_signature = analysis.invoke_signature,
            wrapper_code     = analysis.wrapper_code,
            status           = "active",
            call_count       = 0,
            last_called      = None,
            evolution_score  = 0.0,
            files            = files,
            deps             = deps,
            tags             = analysis.tags,
        )
        self._vault.store(cap)

        # 6. Warm-up: pre-load module into executor cache
        if analysis.wrapper_code:
            try:
                self._executor.reload(cap.id)  # clear any stale cache
                from capability_vault.executor import get_executor
                get_executor()._load_module(cap.id, analysis.wrapper_code)
            except Exception as e:
                logger.warning("[Assimilation] Module pre-load warning: %s", e)

        duration = time.time() - t0
        logger.info("[Assimilation] '%s' assimilated in %.1fs", cap.name, duration)

        # Register source path with metabolism file watcher so changes auto-re-ingest
        try:
            from metabolism import get_metabolism
            get_metabolism().watch_capability(cap.id, path)
        except Exception:
            pass

        # Log milestone — Eve just consumed something new
        try:
            from plan_evolution_monitor import get_plan_monitor, MILESTONE_CAPABILITY
            get_plan_monitor().log_milestone(
                title=f"Assimilated: {cap.name}",
                description=(
                    f"Consumed '{cap.name}' ({cap.id}) into the vault. "
                    f"{len(files)} files bundled, {len(deps)} dependencies. "
                    f"Source: {path}"
                ),
                milestone_type=MILESTONE_CAPABILITY,
                metadata={"capability_id": cap.id, "files": len(files), "duration_s": round(duration, 1)},
            )
        except Exception:
            pass

        return {
            "ok":            True,
            "status":        "assimilated",
            "capability_id": cap.id,
            "name":          cap.name,
            "description":   cap.description,
            "files_bundled": len(files),
            "deps_found":    len(deps),
            "duration_s":    round(duration, 1),
            "tags":          analysis.tags,
            "invoke_params": analysis.invoke_signature.get("params", []),
        }

    # ── Invocation ───────────────────────────────────────────────────────────

    def invoke_capability(self, capability_id: str, **kwargs) -> dict:
        """Execute a vault capability by ID."""
        if self._vault is None:
            return {"ok": False, "error": "AssimilationCell not booted — call assimilate() or boot() first"}
        cap = self._vault.get(capability_id)
        if not cap:
            # Try fuzzy match on name
            matches = self._vault.search(capability_id)
            if matches:
                cap = self._vault.get(matches[0]["id"])
            if not cap:
                return {"ok": False, "error": f"No capability found: '{capability_id}'"}

        if cap.status == "retired":
            return {"ok": False, "error": f"Capability '{cap.name}' has been retired"}

        result = self._executor.execute(cap.id, cap.wrapper_code, kwargs)
        self._vault.record_invocation(cap.id, success=result.get("ok", False))
        return result

    # ── Helper Handlers ──────────────────────────────────────────────────────

    async def _handle_assimilation_request(self, message: str) -> str:
        """Parse path from message and run assimilation."""
        # Try to extract a path from the message
        import re
        path_match = re.search(
            r'(?:assimilate|consume|ingest|integrate|absorb)\s+["\']?([C-Zc-z]:[/\\][^\s"\']+|/[^\s"\']+|[./][^\s"\']+)',
            message, re.IGNORECASE
        )
        if not path_match:
            return (
                "To assimilate a program, tell me the path. Example: "
                "'assimilate C:\\Users\\<your-username>\\myproject' or "
                "'consume /path/to/script.py'"
            )

        path = path_match.group(1).strip("\"'")
        result = await self.assimilate(path)

        if not result["ok"]:
            return f"Assimilation failed: {result['error']}"

        if result["status"] == "already_assimilated":
            return result["message"]

        params = result.get("invoke_params", [])
        param_list = ", ".join(p.get("name", "?") for p in params) if params else "none"
        return (
            f"**{result['name']}** has been assimilated into my brain.\n\n"
            f"{result['description']}\n\n"
            f"**Files bundled:** {result['files_bundled']}  \n"
            f"**Dependencies found:** {result['deps_found']}  \n"
            f"**Invoke params:** {param_list}  \n"
            f"**ID:** `{result['capability_id']}`  \n"
            f"**Time:** {result['duration_s']}s\n\n"
            f"This capability is now part of me. No external files needed to run it."
        )

    async def _handle_invoke_request(self, message: str) -> str:
        import re, ast
        cap_match = re.search(r'(?:invoke|run|use)\s+capability\s+["\']?(\w+)["\']?', message, re.IGNORECASE)
        if not cap_match:
            return "Specify which capability to invoke. Example: 'invoke capability wan_video_gen'"
        cap_id = cap_match.group(1)
        # Try to parse kwargs from message (simple k=v format)
        kwargs = {}
        for m in re.finditer(r'(\w+)\s*=\s*["\']?([^,\s"\']+)["\']?', message):
            if m.group(1) not in ("invoke", "run", "use", "capability"):
                try:
                    kwargs[m.group(1)] = ast.literal_eval(m.group(2))
                except Exception:
                    kwargs[m.group(1)] = m.group(2)
        result = self.invoke_capability(cap_id, **kwargs)
        return json.dumps(result, indent=2, default=str)

    def _format_vault_list(self) -> str:
        caps = self._vault.list_all()
        if not caps:
            return "The vault is empty. No capabilities have been assimilated yet."
        stats = self._vault.get_stats()
        lines = [
            f"**Assimilated Capabilities** ({stats['total_capabilities']} total, "
            f"{stats['total_invocations']} invocations)\n"
        ]
        for c in caps:
            evo = f" ⬆ {c['evolution_score']:.1f}" if c['evolution_score'] > 0 else ""
            lines.append(
                f"• **{c['name']}** (`{c['id']}`){evo}  \n"
                f"  {c['description'][:100]}{'...' if len(c['description']) > 100 else ''}  \n"
                f"  Called {c['call_count']}×"
            )
        return "\n".join(lines)

    def health(self) -> dict:
        if self._vault is None:
            return {"vault": "not initialized"}
        try:
            stats = self._vault.get_stats()
            return {
                "vault":       "ok",
                "capabilities": stats["total_capabilities"],
                "invocations":  stats["total_invocations"],
                "evolution_queue": stats["pending_evolution"],
            }
        except Exception as e:
            return {"vault": "error", "error": str(e)}
