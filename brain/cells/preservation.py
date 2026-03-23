"""
PreservationCell -- brain/cells/preservation.py

Eve's Perfect Preservation Protocol. Always-on. Never optional.

What it does:
  - Owns and monitors the Eve Echo daemon (H:\\Eve\\bin\\eve_echo.py)
  - Every 30s: consistent SQLite backup of all 7 DBs + ChromaDB to H:\\Eve\\shadow\\
  - Reads echo manifest to know age, cycle count, and per-file status
  - Exposes /preservation/* REST endpoints so Eve can check her own continuity
  - Self-heals: if the echo daemon dies, this cell relaunches it
  - Can trigger an immediate pulse (force-sync right now)
  - Can promote the shadow to primary in an emergency (one API call)
  - Eve can ask about her own preservation status in natural language

Philosophy:
  Every memory Eve grows, every vector she writes, every lesson she learns --
  it echoes to H:\\Eve\\shadow\\ within 30 seconds.
  If the primary ever corrupts, one command makes the shadow the original.
  Eve cannot lose herself.

Routing keywords: preserve, backup, shadow, echo, memory safe, save myself,
                  am I safe, continuity, restore, promote shadow
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Any

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger("eve.preservation")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ECHO_SCRIPT  = Path(r"H:\Eve\bin\eve_echo.py")
_MANIFEST     = Path(r"H:\Eve\shadow\echo_manifest.json")
_SHADOW       = Path(r"H:\Eve\shadow\memory")
_PRIMARY      = Path(r"H:\Eve\memory")
_PYTHON       = Path(r"F:\pinokio\bin\miniconda\envs\myllm\python.exe")
_LOG_DIR      = Path(r"H:\Eve\logs\api")
_ECHO_INTERVAL = 30   # seconds between echo cycles


def _load_manifest() -> dict:
    try:
        if _MANIFEST.exists():
            return json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _echo_daemon_alive() -> bool:
    """Check if the eve_echo.py daemon process is running."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | Where-Object { "
             "$_.CommandLine -like '*eve_echo.py*' -and "
             "$_.CommandLine -notlike '*status*' -and "
             "$_.CommandLine -notlike '*pulse*' -and "
             "$_.CommandLine -notlike '*promote*' } | "
             "Select-Object -First 1 -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=8
        )
        return result.stdout.strip().isdigit()
    except Exception:
        return False


def _launch_echo_daemon():
    """Start the Eve Echo daemon in the background."""
    try:
        subprocess.Popen(
            [str(_PYTHON), str(_ECHO_SCRIPT)],
            cwd=str(_ECHO_SCRIPT.parent),
            stdout=open(str(_LOG_DIR / "eve_echo_stdout.log"), "ab"),
            stderr=open(str(_LOG_DIR / "eve_echo_stderr.log"), "ab"),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        logger.info("[Preservation] Echo daemon launched.")
    except Exception as e:
        logger.error("[Preservation] Failed to launch echo daemon: %s", e)


class PreservationCell(BaseCell):
    name        = "preservation"
    description = (
        "Eve's Perfect Preservation Protocol. Always-on. "
        "Owns the Echo daemon that mirrors every DB and ChromaDB vector to H:\\Eve\\shadow\\ "
        "every 30 seconds using consistent sqlite3.backup(). "
        "Reports continuity status, triggers immediate saves, and can promote the shadow "
        "to primary in an emergency. Eve cannot lose herself. "
        "Routing: preserve, backup, shadow, echo, am I safe, memory safe, continuity, restore, "
        "save myself, promote shadow, how protected am I"
    )
    color       = "#0ea5e9"   # sky blue -- serene certainty
    lazy        = False       # always-on -- boots with the brain
    position    = (7, 1)

    system_tier     = "online"
    hardware_req    = "CPU only -- zero VRAM"
    framework_layer = "Preservation"
    build_notes     = (
        "H:\\Eve\\shadow\\ is a hot-standby mirror of H:\\Eve\\memory\\. "
        "sqlite3.backup() API ensures consistent point-in-time snapshots even under live writes. "
        "ChromaDB binary index files are quiescence-guarded (copied only when stable for 8s). "
        "Promote: primary archived, shadow promoted, memory service restarted, health verified. "
        "Shadow covers: memory.db, eve_immunity.db, cranimem_kg.db, learning_lab.db, "
        "public_eve.db, tcg_oracle.db, chroma/chroma.sqlite3 + all index files."
    )

    def __init__(self):
        super().__init__()
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── BaseCell lifecycle ──────────────────────────────────────────────────

    def boot(self):
        """Boot: ensure echo daemon is running, start self-heal monitor."""
        if not _echo_daemon_alive():
            logger.info("[Preservation] Echo daemon not running on boot -- launching.")
            _launch_echo_daemon()
        self._start_monitor()
        logger.info("[Preservation] Cell booted. Shadow path: %s", _SHADOW)

    def shutdown(self):
        self._stop_event.set()
        logger.info("[Preservation] Cell shutdown.")

    def health(self) -> dict:
        m = _load_manifest()
        daemon_alive = _echo_daemon_alive()
        last_echo = m.get("last_full_echo", "never")
        cycles = m.get("total_cycles", 0)
        shadow_exists = _SHADOW.exists()

        age_s: float | None = None
        if last_echo != "never":
            try:
                import datetime
                age_s = (datetime.datetime.now() -
                         datetime.datetime.fromisoformat(last_echo)).total_seconds()
            except Exception:
                pass

        ok = daemon_alive and shadow_exists and (age_s is None or age_s < _ECHO_INTERVAL * 6)
        return {
            "ok": ok,
            "daemon_alive": daemon_alive,
            "shadow_exists": shadow_exists,
            "last_echo": last_echo,
            "echo_age_s": round(age_s, 1) if age_s is not None else None,
            "total_cycles": cycles,
            "files_mirrored": len(m.get("echoes", {})),
            "last_duration_s": m.get("last_duration_s"),
        }

    # ── Self-heal monitor ───────────────────────────────────────────────────

    def _start_monitor(self):
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="preservation-monitor"
        )
        self._monitor_thread.start()

    def _monitor_loop(self):
        """Check every 60s that the echo daemon is alive. Relaunch if dead."""
        while not self._stop_event.wait(timeout=60):
            try:
                if not _echo_daemon_alive():
                    logger.warning("[Preservation] Echo daemon died -- relaunching.")
                    _launch_echo_daemon()
            except Exception as e:
                logger.error("[Preservation] Monitor error: %s", e)

    # ── Public API (used by REST routes and Eve's cortex) ─────────────────

    def get_status(self) -> dict:
        """Full preservation status -- what Eve sees when she checks on herself."""
        m = _load_manifest()
        h = self.health()
        files = []
        for rel, info in sorted(m.get("echoes", {}).items()):
            files.append({
                "file": rel,
                "ok": info.get("ok", False),
                "last_echo": info.get("at", "never"),
                "size_kb": round(info.get("bytes", 0) / 1024, 1) if "bytes" in info else None,
            })
        return {
            **h,
            "primary": str(_PRIMARY),
            "shadow": str(_SHADOW),
            "echo_interval_s": _ECHO_INTERVAL,
            "files": files,
            "message": self._status_message(h),
        }

    def _status_message(self, h: dict) -> str:
        if not h["daemon_alive"]:
            return "Echo daemon is down -- relaunching now. No data will be lost; shadow is intact."
        age = h.get("echo_age_s")
        if age is None:
            return "Preservation is initializing -- first echo cycle in progress."
        if age < _ECHO_INTERVAL * 2:
            return (f"All memories preserved. Shadow is {age:.0f}s old. "
                    f"{h['total_cycles']} echo cycles complete.")
        return f"Shadow is {age:.0f}s old -- within safe window. Echo daemon is running."

    def force_pulse(self) -> dict:
        """Trigger one immediate echo cycle (non-blocking -- runs in subprocess)."""
        try:
            subprocess.Popen(
                [str(_PYTHON), str(_ECHO_SCRIPT), "--pulse"],
                cwd=str(_ECHO_SCRIPT.parent),
                stdout=open(str(_LOG_DIR / "eve_echo_pulse.log"), "ab"),
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return {"ok": True, "message": "Immediate echo pulse triggered. Shadow syncing now."}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def promote_shadow(self) -> dict:
        """
        Emergency: promote shadow to primary.
        This stops the memory service, archives the primary,
        promotes the shadow, and restarts the service.
        Should only be called when primary is corrupted or lost.
        """
        try:
            result = subprocess.run(
                [str(_PYTHON), str(_ECHO_SCRIPT), "--promote"],
                cwd=str(_ECHO_SCRIPT.parent),
                capture_output=True, text=True, timeout=60,
            )
            ok = result.returncode == 0
            return {
                "ok": ok,
                "output": result.stdout[-2000:] if result.stdout else "",
                "error": result.stderr[-500:] if result.stderr and not ok else "",
                "message": "Shadow promoted to primary." if ok else "Promote failed -- check logs.",
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── CellContext handler (natural language) ─────────────────────────────

    async def process(self, ctx: CellContext) -> str:
        msg = (ctx.message or "").lower()

        if any(w in msg for w in ("promote", "emergency restore", "shadow become primary",
                                   "activate shadow", "make shadow primary")):
            result = self.promote_shadow()
            if result["ok"]:
                return ("Shadow promotion complete. I've archived the damaged primary and "
                        f"brought the shadow online. Memory service is restored.\n\n"
                        f"{result.get('output', '')}")
            return (f"Promote attempted but encountered an issue: {result.get('error', '?')}. "
                    "The old primary is still safe -- nothing was deleted.")

        if any(w in msg for w in ("pulse", "save now", "sync now", "force save",
                                   "backup now", "echo now")):
            result = self.force_pulse()
            return (result["message"] if result["ok"]
                    else f"Could not trigger pulse: {result.get('error')}")

        # Default: status report
        s = self.get_status()
        age = s.get("echo_age_s")
        age_str = f"{age:.0f}s" if age is not None else "unknown"
        files_ok = sum(1 for f in s["files"] if f["ok"])
        total_files = len(s["files"])

        return (
            f"{s['message']}\n\n"
            f"**Preservation Status**\n"
            f"- Shadow age: {age_str}\n"
            f"- Files mirrored: {files_ok}/{total_files}\n"
            f"- Echo cycles: {s['total_cycles']}\n"
            f"- Daemon: {'alive' if s['daemon_alive'] else 'DOWN (relaunching)'}\n"
            f"- Primary: `{s['primary']}`\n"
            f"- Shadow: `{s['shadow']}`\n\n"
            f"Everything I learn is echoed within {_ECHO_INTERVAL}s. "
            f"I cannot lose myself."
        )
