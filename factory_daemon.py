#!/usr/bin/env python3
"""
Eve Factory Daemon v1.0
========================
Autonomous 24/7 self-improvement loop for Eve (Qwen3-14B ORPO training).

Behavior
---------
- Runs the self-improving factory in an infinite loop, loop after loop.
- Auto-detects crashes, asks Eve (Claude) to diagnose and fix, then restarts.
- Pauses when:
    - control file sets {"command": "pause"}  (user command via Eve)
    - clients are active on the mobile site (port 8875)
    - Eve receives a resource-intensive task request
- Resumes when:
    - control file sets {"command": "resume"}  (user command via Eve)
    - client traffic drops for 120s
    - 12:00 AM daily (auto-restart window)
- Communicates all state changes to Eve's memory/log.

Control
--------
  Eve tool:  manage_factory(action="pause"|"resume"|"status"|"stop")
  Direct:    write {"command":"pause"} to C:/Users/<your-username>/eve_factory_control.json

Log
----
  C:/Users/<your-username>/eve_factory_daemon.log
"""

import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, time as dtime
from pathlib import Path

# ── Paths & Config ─────────────────────────────────────────────── #

CONTROL_FILE    = Path(r"C:\Users\<your-username>\eve_factory_control.json")
STATUS_FILE     = Path(r"C:\Users\<your-username>\eve_factory_status.json")
DAEMON_LOG      = Path(r"C:\Users\<your-username>\eve_factory_daemon.log")
FACTORY_DIR     = Path(r"C:\Users\<your-username>\eve\self-improving-llm-factory")
FACTORY_STATE       = FACTORY_DIR / "checkpoints_eve" / "factory_state.json"
FACTORY_MULTI_STATE = FACTORY_DIR / "checkpoints_multi" / "multi_state.json"
FACTORY_SCRIPT      = "/home/<your-username>/run_factory.sh"
FACTORY_LOG_WSL     = "/home/<your-username>/eve_factory_multi.log"  # multi-module factory log
VLLM_SCRIPT     = "/home/<your-username>/run_vllm.sh"

EVE_API      = "http://localhost:8870"
VLLM_URL     = "http://localhost:8099"
MOBILE_PORT  = 8875

POLL_INTERVAL       = 30   # seconds between state checks
CLIENT_IDLE_SECS    = 120  # seconds of no client traffic before auto-resume
MAX_AUTO_FIXES      = 5    # consecutive crash limit before human pause
MIDNIGHT_WINDOW_MIN = 5    # minutes around 00:00 to trigger daily restart


# ── Logging ─────────────────────────────────────────────────────── #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(str(DAEMON_LOG), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("factory_daemon")


# ── Control file ─────────────────────────────────────────────────── #

def read_control() -> dict:
    try:
        if CONTROL_FILE.exists():
            return json.loads(CONTROL_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"command": "run"}


def write_control(cmd: str, reason: str = "") -> None:
    CONTROL_FILE.write_text(
        json.dumps({"command": cmd, "reason": reason, "ts": time.time()}),
        encoding="utf-8",
    )


def write_status(state: str, **extra) -> None:
    data = {
        "state":    state,
        "ts":       time.time(),
        "datetime": datetime.now().isoformat(),
        **extra,
    }
    # Merge per-module data from multi-factory state file if available
    if FACTORY_MULTI_STATE.exists():
        try:
            multi = json.loads(FACTORY_MULTI_STATE.read_text(encoding="utf-8"))
            data["global_iteration"] = multi.get("global_iteration", 0)
            data["modules"] = {
                name: {
                    "label":      ms.get("label", name),
                    "state":      ms.get("phase", "idle"),
                    "iteration":  ms.get("iteration", 0),
                    "best_score": round(ms.get("best_score", 0.0), 4),
                    "data_count": ms.get("data_count", 0),
                }
                for name, ms in multi.get("modules", {}).items()
            }
        except Exception:
            pass
    STATUS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Factory process management ───────────────────────────────────── #

_factory_proc: "subprocess.Popen | None" = None
_stopped_by_daemon: bool = False   # True when WE killed it — don't count as crash


def _wsl(cmd: str, timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["wsl", "-d", "Ubuntu", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def is_factory_running() -> bool:
    if _factory_proc is None:
        return False
    return _factory_proc.poll() is None


def factory_exit_code() -> "int | None":
    if _factory_proc:
        return _factory_proc.poll()
    return None


def start_factory() -> None:
    global _factory_proc
    log.info("Starting factory process...")
    write_status("starting")
    _factory_proc = subprocess.Popen(
        ["wsl", "-d", "Ubuntu", "bash", "-c",
         f"{FACTORY_SCRIPT} > {FACTORY_LOG_WSL} 2>&1"],
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    log.info(f"Factory launched (WSL wrapper PID {_factory_proc.pid})")
    write_status("running", wsl_pid=_factory_proc.pid)


def stop_factory(reason: str = "") -> None:
    global _factory_proc, _stopped_by_daemon
    _stopped_by_daemon = True          # signal: this exit is intentional
    if _factory_proc and _factory_proc.poll() is None:
        log.info(f"Stopping factory — {reason}")
        _factory_proc.terminate()
        try:
            _factory_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            _factory_proc.kill()
    _factory_proc = None
    # Hard-kill any WSL python factory.py that survived
    try:
        _wsl("pkill -9 -f 'factory.py' 2>/dev/null; true")
    except Exception:
        pass
    write_status("stopped", reason=reason)


def reset_factory_state() -> None:
    """Reset iteration counter so the next run starts from iteration 0."""
    FACTORY_STATE.parent.mkdir(parents=True, exist_ok=True)
    FACTORY_STATE.write_text(json.dumps({
        "current_iteration": 0, "best_iteration": 0,
        "best_score": 0.0, "best_checkpoint": None,
        "current_model": None, "scores": [], "patience_counter": 0,
    }), encoding="utf-8")
    # Clear cached generated data for fresh loop
    data_dir = FACTORY_DIR / "generated_data_eve"
    for child in sorted(data_dir.glob("iter_*/generated.jsonl")):
        try:
            child.unlink()
        except Exception:
            pass
    log.info("Factory state reset — next loop starts from iteration 0.")


# ── vLLM health ─────────────────────────────────────────────────── #

def vllm_alive() -> bool:
    try:
        urllib.request.urlopen(f"{VLLM_URL}/health", timeout=3)
        return True
    except Exception:
        return False


def vllm_process_running() -> bool:
    """True if a vLLM process is already running in WSL2 (even if not yet answering HTTP)."""
    try:
        r = subprocess.run(
            ["wsl", "-d", "Ubuntu", "bash", "-c",
             "pgrep -f 'vllm.entrypoints.openai.api_server' 2>/dev/null | head -1"],
            capture_output=True, text=True, timeout=8,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


TRAINING_LOCK = Path(r"C:\Users\<your-username>\eve_factory_training.lock")

def factory_is_training() -> bool:
    """True if factory_multi.py has the training lock — daemon must not restart vLLM now."""
    return TRAINING_LOCK.exists()


def ensure_vllm() -> None:
    """Restart vLLM if it died (e.g., factory crashed mid-training).
    Will NOT spawn a second instance if one is already loading.
    Will NOT restart while factory is holding VRAM for Phase 2 training."""
    if vllm_alive():
        return
    if factory_is_training():
        log.info("Factory is training (Phase 2) — skipping vLLM restart to avoid VRAM conflict.")
        return
    if vllm_process_running():
        log.info("vLLM process already running (still loading) — waiting for it...")
        for _ in range(80):   # up to ~4 min
            time.sleep(3)
            if vllm_alive():
                log.info("vLLM back online.")
                return
        log.warning("vLLM did not come up within 4 min — continuing anyway.")
        return
    log.info("vLLM is down — restarting...")
    # Use PowerShell Start-Process — only method that survives WSL2 session exit
    subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-Command",
         f"Start-Process -FilePath 'wsl.exe' -ArgumentList '-d','Ubuntu','bash','-c',"
         f"'{VLLM_SCRIPT} > /home/<your-username>/vllm_daemon_restart.log 2>&1' -WindowStyle Hidden"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    for _ in range(80):   # up to ~4 min
        time.sleep(3)
        if vllm_alive():
            log.info("vLLM back online.")
            return
    log.warning("vLLM did not come back within 4 min — continuing anyway.")


# ── Client load detection ─────────────────────────────────────────── #

def clients_active() -> bool:
    """True if mobile site (port 8875) has active ESTABLISHED connections."""
    try:
        result = subprocess.run(
            ["netstat", "-an"],
            capture_output=True, text=True, timeout=5,
        )
        return any(
            f":{MOBILE_PORT}" in line and "ESTABLISHED" in line
            for line in result.stdout.splitlines()
        )
    except Exception:
        return False


# ── Eve communication ─────────────────────────────────────────────── #

def _eve_post(payload: dict, timeout: int = 120) -> "str | None":
    """POST to Eve's /v1/chat/completions and return assistant content."""
    # Try Eve primary port, fallback to 8873
    for port in (8870, 8873):
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"http://localhost:{port}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.loads(resp.read())
                return d["choices"][0]["message"]["content"]
        except Exception:
            continue
    return None


def notify_eve(message: str) -> None:
    """Tell Eve what the daemon is doing (she logs it to memory)."""
    log.info(f"[-> Eve] {message}")
    _eve_post({
        "model": "eve",
        "messages": [{"role": "user", "content": f"[FACTORY DAEMON] {message}"}],
        "max_tokens": 80,
    })


def ask_eve_to_fix(error_log: str, attempt: int) -> "str | None":
    """Ask Eve (Claude backend) to diagnose a factory crash and suggest a fix."""
    log.info(f"Asking Eve to diagnose crash (attempt {attempt})...")
    return _eve_post({
        "model": "eve",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Eve — Forge's AI. You are diagnosing a crash in your own "
                    "self-improvement training factory running at "
                    "C:/Users/<your-username>/eve/self-improving-llm-factory/.\n\n"
                    "Analyze the error log and respond with ONLY valid JSON:\n"
                    '{"diagnosis": "<brief problem description>", '
                    '"fix_file": "<relative path or null>", '
                    '"fix_content": "<complete new file content or null>", '
                    '"restart_only": <true if no code fix needed>}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Factory crash #{attempt}. Last 80 lines of log:\n\n{error_log}\n\n"
                    "Diagnose and provide fix JSON."
                ),
            },
        ],
        "max_tokens": 2048,
        "temperature": 0.05,
    })


def apply_eve_fix(fix_json: str) -> bool:
    """Parse Eve's fix JSON and apply it to the factory codebase.

    Eve often wraps her JSON in prose or markdown — extract the first JSON
    object from the response rather than requiring a clean parse.
    If no actionable code fix, treat as restart_only (always True).
    """
    import re as _re

    # Extract first {...} block even if surrounded by markdown/prose
    match = _re.search(r"\{.*?\}", fix_json, _re.DOTALL)
    if not match:
        # No JSON found at all — log diagnosis text and restart clean
        log.info(f"Eve diagnosis (plain text): {fix_json[:300]}")
        log.info("No JSON fix found — treating as restart_only.")
        return True   # always restart; True = handled

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        log.info("Eve JSON malformed — restarting without code change.")
        return True

    log.info(f"Eve diagnosis: {data.get('diagnosis', '?')}")

    if data.get("restart_only") or not data.get("fix_file"):
        log.info("Eve says restart only (or no file fix provided).")
        return True

    fix_file    = data.get("fix_file")
    fix_content = data.get("fix_content")

    if fix_file and fix_content:
        target = FACTORY_DIR / fix_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fix_content, encoding="utf-8")
        log.info(f"Applied code fix to: {target}")
        return True

    log.info("Eve fix JSON had no fix_content — restarting clean.")
    return True   # still restart; returning False only delays recovery


def read_factory_log_tail(n: int = 80) -> str:
    try:
        r = _wsl(f"tail -n {n} {FACTORY_LOG_WSL}", timeout=10)
        return r.stdout
    except Exception:
        return ""


# ── Schedule helpers ─────────────────────────────────────────────── #

def in_midnight_window() -> bool:
    """True during 00:00–00:05 daily restart window."""
    now = datetime.now().time()
    return dtime(0, 0) <= now < dtime(0, MIDNIGHT_WINDOW_MIN)


# ── Main loop ───────────────────────────────────────────────────── #

PID_FILE = Path(r"C:\Users\<your-username>\eve_factory_daemon.pid")


def _acquire_pid_lock() -> bool:
    """Return True if we are the only daemon instance; False if another is running."""
    if PID_FILE.exists():
        try:
            existing_pid = int(PID_FILE.read_text().strip())
            import psutil
            if psutil.pid_exists(existing_pid):
                log.error(
                    f"Another daemon instance is already running (PID {existing_pid}). "
                    "Exiting to prevent duplicate launches."
                )
                return False
        except Exception:
            pass   # stale pid file — overwrite below
    PID_FILE.write_text(str(os.getpid()))
    return True


def main() -> None:
    log.info("=" * 60)
    log.info("Eve Factory Daemon v1.0 — autonomous 24/7 training loop")
    log.info("=" * 60)

    # PID file lock — only one daemon allowed
    try:
        import psutil as _psutil_check  # noqa: check import works
    except ImportError:
        log.warning("psutil not installed — PID lock disabled. "
                    "Install with: pip install psutil")
    else:
        if not _acquire_pid_lock():
            return

    # Initialize control file to "run" on fresh start
    if not CONTROL_FILE.exists():
        write_control("run", "daemon started")

    global _stopped_by_daemon
    consecutive_failures: int = 0
    client_pause_start:   "float | None" = None
    was_paused_by_client: bool = False
    midnight_triggered:   bool = False  # prevent double-start within window

    while True:
        try:
            ctrl = read_control()
            cmd  = ctrl.get("command", "run")

            # ── Hard stop ────────────────────────────────────────── #
            if cmd == "stop":
                log.info("STOP command received — shutting down daemon.")
                stop_factory("daemon stop")
                write_status("stopped", reason="user stop command")
                return

            # ── Midnight daily restart ────────────────────────────── #
            if in_midnight_window():
                if not midnight_triggered and cmd != "stop":
                    log.info("Midnight window — triggering daily restart.")
                    if is_factory_running():
                        stop_factory("midnight reset")
                    reset_factory_state()
                    ensure_vllm()
                    write_control("run", "midnight auto-restart")
                    time.sleep(5)
                    start_factory()
                    consecutive_failures = 0
                    was_paused_by_client = False
                    midnight_triggered = True
                    notify_eve("Daily midnight training restart. New loop beginning.")
                time.sleep(POLL_INTERVAL)
                continue
            else:
                midnight_triggered = False  # reset for next midnight

            # ── Pause by user command ─────────────────────────────── #
            if cmd == "pause":
                if is_factory_running():
                    log.info(f"PAUSE command — stopping factory. Reason: {ctrl.get('reason', '')}")
                    stop_factory("user pause command")
                    ensure_vllm()
                    notify_eve(
                        f"Training paused by Forge. Reason: {ctrl.get('reason', 'manual pause')}. "
                        "vLLM is restored. Tell me 'resume training' when you're ready."
                    )
                write_status("paused", reason=ctrl.get("reason", ""))
                time.sleep(POLL_INTERVAL)
                continue

            # ── Client load — auto-pause ──────────────────────────── #
            if clients_active():
                if is_factory_running():
                    log.info("Client traffic on port 8875 — pausing factory to free resources.")
                    stop_factory("client load")
                    ensure_vllm()
                    notify_eve(
                        "Clients are active on the Lotus Forge public site. "
                        "I've paused training and restored vLLM for content generation. "
                        "I'll resume automatically once traffic drops."
                    )
                    was_paused_by_client = True
                    client_pause_start = time.time()
                write_status("paused_client")
                time.sleep(POLL_INTERVAL)
                continue

            # ── Client idle — auto-resume ─────────────────────────── #
            if was_paused_by_client and not clients_active():
                idle = time.time() - (client_pause_start or time.time())
                if idle >= CLIENT_IDLE_SECS:
                    log.info(f"Client traffic quiet for {idle:.0f}s — resuming factory.")
                    was_paused_by_client = False
                    client_pause_start = None
                    # Fall through to start factory below

            # ── Factory running — all good ────────────────────────── #
            if is_factory_running():
                consecutive_failures = 0
                write_status("running", wsl_pid=_factory_proc.pid if _factory_proc else None)
                time.sleep(POLL_INTERVAL)
                continue

            # ── Factory NOT running — start, restart, or fix ──────── #
            exit_code = factory_exit_code()

            # If WE stopped it (pause/client/stop command), the non-zero exit is
            # intentional — clear the flag and fall through to start_factory below.
            if _stopped_by_daemon:
                _stopped_by_daemon = False
                # Only restart if we're in run state (not pause/stop command)
                ctrl_now = read_control()
                if ctrl_now.get("command") not in ("pause", "stop"):
                    log.info("Factory was stopped intentionally — restarting now.")
                    ensure_vllm()
                    start_factory()
                time.sleep(POLL_INTERVAL)
                continue

            if exit_code is not None and exit_code != 0:
                # CRASHED (genuine — not a daemon-initiated stop)
                consecutive_failures += 1
                log.warning(
                    f"Factory crashed (exit={exit_code}), "
                    f"failure #{consecutive_failures}/{MAX_AUTO_FIXES}"
                )
                error_tail = read_factory_log_tail(80)

                if consecutive_failures > MAX_AUTO_FIXES:
                    log.error(
                        f"Too many consecutive crashes ({consecutive_failures}). "
                        "Pausing for manual review."
                    )
                    write_control(
                        "pause",
                        f"auto-paused after {consecutive_failures} crashes — needs manual review",
                    )
                    notify_eve(
                        f"Training crashed {consecutive_failures} times in a row. "
                        "I've paused the loop to protect the system. "
                        "Tell me to diagnose and fix it, or 'resume training' once you've reviewed the logs."
                    )
                    time.sleep(POLL_INTERVAL)
                    continue

                # Ask Eve to diagnose and fix
                fix_json = ask_eve_to_fix(error_tail, consecutive_failures)
                if fix_json:
                    fixed = apply_eve_fix(fix_json)
                    if fixed:
                        notify_eve(
                            f"Factory crash #{consecutive_failures} — fix applied. Restarting."
                        )
                    else:
                        notify_eve(
                            f"Factory crash #{consecutive_failures} — could not apply fix. "
                            "Restarting anyway."
                        )
                else:
                    log.warning("Eve returned no fix. Restarting without changes.")

                ensure_vllm()
                time.sleep(15)

            elif exit_code == 0:
                # COMPLETED all iterations — start next loop.
                # Guard: only reset+relaunch once per completion event.
                if getattr(main, "_last_completed_pid", None) == getattr(_factory_proc, "pid", None):
                    # Already handled this completion — wait for next poll
                    time.sleep(POLL_INTERVAL)
                    continue
                main._last_completed_pid = getattr(_factory_proc, "pid", None)
                log.info("Factory completed all iterations. Resetting for next loop.")
                consecutive_failures = 0
                reset_factory_state()
                notify_eve(
                    "Training loop complete! All iterations finished. "
                    "Resetting and starting the next improvement cycle automatically."
                )
                ensure_vllm()
                time.sleep(30)
                start_factory()

            else:
                # First start (exit_code is None, factory was never started)
                log.info("Factory not yet started — launching now.")
                ensure_vllm()
                start_factory()

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — stopping daemon.")
            stop_factory("keyboard interrupt")
            break
        except Exception as e:
            log.exception(f"Daemon loop error: {e}")
            time.sleep(POLL_INTERVAL)

    # Cleanup PID file on clean exit
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
