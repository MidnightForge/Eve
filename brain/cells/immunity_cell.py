"""
ImmunityCell — Eve's self-protection and threat detection system.

Eve's immune system. Always-on. Never optional.

What it does:
  - Scans all code before it enters the capability vault (pre-store hook)
  - Verifies wrapper integrity before every invocation (hash check)
  - Detects behavioral anomalies in running capabilities
  - Monitors for prompt injection in user messages
  - Maintains a growing threat signature database (learns from hits)
  - Auto-quarantines suspicious capabilities
  - Runs a background vault integrity monitor thread
  - Exposes /immunity/* REST endpoints for status and management
  - Integrates with Metabolism (file watcher) and Evolution (proposal gating)

Research basis:
  - MITRE ATT&CK T1059 (Command & Scripting), T1027 (Obfuscation),
    T1547 (Persistence), T1123 (Audio Capture), T1570 (Lateral Transfer)
  - CrowdStrike behavioral analytics pattern
  - YARA/Sigma rule approach adapted to Python AST
  - Multi-layer defense: static → dynamic → behavioral → integrity
  - AI-specific: prompt injection (OWASP LLM Top 10 #1), model poisoning

Routing keywords: virus, malware, threat, scan, quarantine, immunity,
                  infected, suspicious, safe, security, protect
"""
import asyncio
import json
import logging
import threading
import time
from typing import Any

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger("eve.immunity")


class ImmunityCell(BaseCell):
    name        = "immunity"
    description = (
        "Eve's self-protection system. Scans all code for malware before vault storage, "
        "verifies integrity on every invocation, detects prompt injection, "
        "monitors behavioral anomalies, and auto-quarantines threats. "
        "MITRE ATT&CK aligned. Always-on."
    )
    color       = "#0f4c3a"   # deep forest green — active protection
    lazy        = False       # ALWAYS boots — cannot be disabled
    position    = (6, 1)

    system_tier     = "online"
    hardware_req    = "CPU only — zero VRAM"
    framework_layer = "Security"
    research_basis  = (
        "MITRE ATT&CK T1059/T1027/T1547/T1123/T1570; "
        "CrowdStrike behavioral analytics; YARA/Sigma rule framework; "
        "OWASP LLM Top 10 (Prompt Injection, Insecure Output Handling); "
        "Adversarial-robust ML defense; Multi-layer defense-in-depth; "
        "Static (AST/regex) + Dynamic (behavioral) + Integrity (SHA256) analysis"
    )
    build_notes = (
        "ACTIVE: Pre-store scan (AST+regex+signatures) → integrity hash on store → "
        "hash verify on invoke → behavioral baseline monitoring → background vault scan. "
        "SQLite threat DB at H:/Eve/memory/eve_immunity.db. "
        "Auto-quarantine on CRITICAL/HIGH findings. "
        "Signature DB grows from hits. Integrated with AssimilationCell + Evolution."
    )

    def __init__(self):
        super().__init__()
        self._db = None
        self._scanner = None
        self._vault_guard = None
        self._behavior_monitor = None
        self._threat_update_thread = None

    # ── Boot ──────────────────────────────────────────────────────────────────

    async def boot(self) -> None:
        import sys
        sys.path.insert(0, "C:/Users/<your-username>/eve")

        from immunity.threat_db import get_immunity_db
        from immunity.threat_scanner import get_scanner
        from immunity.vault_guard import get_vault_guard
        from immunity.behavior_monitor import get_behavior_monitor

        self._db = get_immunity_db()
        self._scanner = get_scanner()
        self._vault_guard = get_vault_guard()
        self._behavior_monitor = get_behavior_monitor()

        # Start background vault integrity monitor
        self._vault_guard.start()

        # Start periodic signature refresh (every 30 min)
        self._threat_update_thread = threading.Thread(
            target=self._sig_refresh_loop, daemon=True, name="eve-immunity-refresh"
        )
        self._threat_update_thread.start()

        stats = self._db.get_stats()
        logger.info(
            "[Immunity] Online — %d signatures, %d events, %d quarantined",
            stats["active_signatures"], stats["total_events"],
            stats["quarantined_capabilities"],
        )
        self._status = CellStatus.ACTIVE

    # ── Process (natural language routing) ────────────────────────────────────

    async def process(self, ctx: CellContext) -> Any:
        msg = ctx.message.lower()

        if any(w in msg for w in ("scan", "check code", "analyze code")):
            return await self._handle_scan_request(ctx.message)

        if any(w in msg for w in ("quarantine list", "what's quarantined", "blocked capabilities")):
            return self._format_quarantine_list()

        if any(w in msg for w in ("threat log", "security log", "recent threats", "what threats")):
            return self._format_threat_log()

        if "release quarantine" in msg or "unquarantine" in msg:
            return await self._handle_release(ctx.message)

        if any(w in msg for w in ("immunity status", "security status", "am i safe", "are we safe")):
            return self._format_status()

        if any(w in msg for w in ("add signature", "new threat pattern", "learn threat")):
            return await self._handle_add_sig(ctx.message)

        return None

    # ── Public API (called by other cells/routes) ─────────────────────────────

    def scan_code(self, code: str, target: str = "unknown") -> dict:
        """Scan Python code for threats. Returns {ok, violations, score, warnings}."""
        if not self._scanner:
            return {"ok": True, "violations": [], "score": 0.0, "warnings": ["scanner not initialized"]}
        result = self._scanner.scan_code(code, target)
        return {
            "ok": result.passed,
            "score": round(result.score, 3),
            "violations": [f"[{f.severity}] {f.rule}: {f.detail}" for f in result.findings
                           if f.severity in ("CRITICAL", "HIGH")],
            "warnings": [f"[{f.severity}] {f.rule}: {f.detail}" for f in result.findings
                         if f.severity in ("MEDIUM", "LOW")],
            "finding_count": len(result.findings),
        }

    def scan_prompt(self, message: str) -> dict:
        """Scan a user message for prompt injection. Returns {ok, risk_level, findings}."""
        if not self._scanner:
            return {"ok": True, "risk_level": "unknown"}
        result = self._scanner.scan_prompt(message)
        risk = "HIGH" if result.critical_count > 0 else ("MEDIUM" if not result.passed else "LOW")
        return {
            "ok": result.passed,
            "risk_level": risk,
            "findings": [{"rule": f.rule, "severity": f.severity, "detail": f.detail}
                         for f in result.findings],
        }

    def pre_store(self, capability_id: str, wrapper_code: str) -> dict:
        """Gate called by AssimilationCell before storing to vault."""
        if not self._vault_guard:
            return {"ok": True, "violations": []}
        return self._vault_guard.pre_store_check(capability_id, wrapper_code)

    def pre_invoke(self, capability_id: str, wrapper_code: str) -> dict:
        """Gate called by executor before running wrapper code."""
        if not self._vault_guard:
            return {"ok": True, "reason": "guard not initialized"}
        return self._vault_guard.pre_invoke_check(capability_id, wrapper_code)

    def quarantine(self, capability_id: str, reason: str, duration_h: float = 0):
        if self._db:
            self._db.quarantine(capability_id, reason, duration_h)

    def release_quarantine(self, capability_id: str, reason: str = "manual"):
        if self._db:
            self._db.release_quarantine(capability_id, reason)

    def add_signature(self, name: str, pattern: str, category: str,
                      severity: str, description: str):
        if self._db:
            self._db.add_signature(name, pattern, category, severity, description)
            if self._scanner:
                self._scanner._refresh_db_sigs()
            logger.info("[Immunity] New signature added: %s (%s/%s)", name, category, severity)

    def log_event(self, event_type: str, severity: str, source: str,
                  detail: str, evidence: str = "") -> int:
        if self._db:
            return self._db.log_event(event_type, severity, source, detail, evidence)
        return -1

    def get_stats(self) -> dict:
        if not self._db:
            return {"status": "not initialized"}
        stats = self._db.get_stats()
        stats["vault_last_scan_ago_s"] = (
            round(self._vault_guard.last_scan_ago_s) if self._vault_guard and
            self._vault_guard.last_scan_ago_s else None
        )
        stats["active_invocations"] = (
            self._behavior_monitor.get_active_invocations()
            if self._behavior_monitor else []
        )
        return stats

    # ── Background signature refresh ──────────────────────────────────────────

    def _sig_refresh_loop(self):
        """Periodically refresh scanner's in-memory signatures from DB."""
        while True:
            time.sleep(1800)  # 30 minutes
            try:
                if self._scanner:
                    self._scanner._refresh_db_sigs()
                    logger.debug("[Immunity] Signature cache refreshed")
            except Exception as e:
                logger.debug("[Immunity] Sig refresh error: %s", e)

    # ── NL handlers ──────────────────────────────────────────────────────────

    async def _handle_scan_request(self, message: str) -> str:
        import re
        code_match = re.search(r'```(?:python)?\s*(.*?)```', message, re.DOTALL)
        if not code_match:
            return ("To scan code, wrap it in ```python ... ``` code blocks. "
                    "Or say 'scan capability <id>' to scan a vaulted capability.")
        code = code_match.group(1)
        result = self.scan_code(code, "manual_scan")
        if result["ok"]:
            return f"Code scan CLEAN (score={result['score']:.2f}). No threats detected."
        return (
            f"Code scan BLOCKED (score={result['score']:.2f})\n\n"
            f"**Violations:**\n" +
            "\n".join(f"• {v}" for v in result["violations"]) +
            (f"\n\n**Warnings:**\n" + "\n".join(f"• {w}" for w in result["warnings"])
             if result["warnings"] else "")
        )

    async def _handle_release(self, message: str) -> str:
        import re
        cap_match = re.search(r'(?:release|unquarantine)\s+(?:quarantine\s+)?["\']?(\w+)["\']?',
                              message, re.IGNORECASE)
        if not cap_match:
            return "Specify which capability to release. Example: 'release quarantine wan_video_gen'"
        cap_id = cap_match.group(1)
        self.release_quarantine(cap_id, "released by Forge")
        return f"**{cap_id}** released from quarantine."

    async def _handle_add_sig(self, message: str) -> str:
        return ("To add a threat signature, use the REST API: "
                "POST /immunity/signatures/add with {name, pattern, category, severity, description}")

    def _format_quarantine_list(self) -> str:
        if not self._db:
            return "Immunity system not initialized."
        items = self._db.get_quarantine_list()
        if not items:
            return "No capabilities currently quarantined."
        lines = [f"**Quarantine Registry** ({len(items)} capabilities)\n"]
        for q in items:
            lines.append(
                f"• **{q['capability_id']}** — {q['reason'][:80]}\n"
                f"  Quarantined: {time.strftime('%Y-%m-%d %H:%M', time.localtime(q['quarantined_at']))}"
            )
        return "\n".join(lines)

    def _format_threat_log(self) -> str:
        if not self._db:
            return "Immunity system not initialized."
        events = self._db.get_events(limit=20, unresolved_only=True)
        if not events:
            return "No unresolved threats in log."
        lines = [f"**Threat Log** ({len(events)} unresolved)\n"]
        for e in events:
            lines.append(
                f"• [{e['severity']}] **{e['event_type']}** — {e['source']}\n"
                f"  {e['detail'][:100]}\n"
                f"  {time.strftime('%m/%d %H:%M', time.localtime(e['timestamp']))}"
            )
        return "\n".join(lines)

    def _format_status(self) -> str:
        stats = self.get_stats()
        quarantined = stats.get("quarantined_capabilities", 0)
        critical = stats.get("critical_unresolved", 0)
        scan_ago = stats.get("vault_last_scan_ago_s")
        scan_str = f"{scan_ago}s ago" if scan_ago else "pending"
        status_icon = "SECURE" if critical == 0 and quarantined == 0 else "ALERT"
        return (
            f"**Eve Immunity System — {status_icon}**\n\n"
            f"Active signatures: {stats.get('active_signatures', 0)}\n"
            f"Total threat events: {stats.get('total_events', 0)}\n"
            f"Unresolved threats: {stats.get('unresolved_events', 0)}\n"
            f"Critical unresolved: {critical}\n"
            f"Quarantined capabilities: {quarantined}\n"
            f"Vault last scanned: {scan_str}\n"
            f"Active invocations: {len(stats.get('active_invocations', []))}"
        )

    def health(self) -> dict:
        if not self._db:
            return {"status": "not initialized"}
        stats = self._db.get_stats()
        return {
            "status": "active",
            "signatures": stats["active_signatures"],
            "total_events": stats["total_events"],
            "unresolved": stats["unresolved_events"],
            "critical": stats["critical_unresolved"],
            "quarantined": stats["quarantined_capabilities"],
            "vault_guard": "running" if (self._vault_guard and self._vault_guard._running) else "stopped",
        }
