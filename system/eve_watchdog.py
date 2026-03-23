"""
Eve Process Watchdog — H:\Eve\bin\eve_watchdog.py
===================================================
Windows-native Python process supervisor.  Manages all Eve services with:
  - Priority-ordered startup (memory → backend → tools → sidecars)
  - Health-check-based liveness (HTTP or process-alive)
  - Automatic restart on crash with cooldown backoff
  - Crash envelope written to H:\Eve\logs\crash\ on every death
  - All stdout/stderr redirected to H:\Eve\logs\api\<service>.log

Run at login via Task Scheduler (see H:\Eve\bin\README_WATCHDOG.txt).
Kill with Ctrl-C or by stopping the process — all children are cleaned up.
"""

import datetime
import faulthandler
import json
import logging
import os
import signal
import subprocess
import sys
import time
import threading
import urllib.request
from pathlib import Path

# ── Crash safety for the watchdog itself ──────────────────────────────────────
CRASH_DIR = Path("H:/Eve/logs/crash")
CRASH_DIR.mkdir(parents=True, exist_ok=True)
_fault_path = CRASH_DIR / "watchdog_fault.log"
faulthandler.enable(file=open(_fault_path, "a"), all_threads=True)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path("H:/Eve/logs/api")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "watchdog.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("eve.watchdog")

# ── Paths ─────────────────────────────────────────────────────────────────────
PYTHON      = r"F:\pinokio\bin\miniconda\envs\myllm\python.exe"
PYTHON_IRIS = r"C:\Users\<your-username>\eve\eve_env\Scripts\python.exe"
PYTHON_BASE = r"F:\pinokio\bin\miniconda\python.exe"
EVE_DIR     = r"C:\Users\<your-username>\eve"


# ── Service registry ──────────────────────────────────────────────────────────
# Fields:
#   name        — human label
#   cmd         — list[str] command to launch
#   cwd         — working directory (or None for default)
#   env_extra   — dict of extra env vars to merge
#   health_url  — HTTP URL to GET (200 = alive).  None = process-alive only
#   port        — port to check (logged only, not required)
#   priority    — startup order (lower = first)
#   depends_on  — name of service that must be healthy first
#   restart_delay_s  — seconds to wait before restarting after a crash
#   max_restarts    — max restarts in rolling window; 0 = unlimited
#   critical         — True = halt watchdog if this service never comes up

SERVICES = [
    {
        "name": "memory",
        "cmd":  [PYTHON, r"H:\Eve\bin\memory_service.py"],
        "cwd":  r"H:\Eve\bin",
        "env_extra": {},
        "health_url": "http://127.0.0.1:8767/health",
        "port": 8767,
        "priority": 1,
        "depends_on": None,
        "restart_delay_s": 5,
        "max_restarts": 0,
        "critical": True,
    },
    {
        "name": "tool_server",
        "cmd":  [PYTHON, "eve_tool_server.py"],
        "cwd":  EVE_DIR + r"\EveToolServer",
        "env_extra": {},
        "health_url": "http://127.0.0.1:8769/health",
        "port": 8769,
        "priority": 2,
        "depends_on": "memory",
        "restart_delay_s": 5,
        "max_restarts": 0,
        "critical": False,
    },
    {
        "name": "eve_backend",
        "cmd":  [PYTHON, "-m", "uvicorn", "app:app",
                 "--host", "0.0.0.0", "--port", "8870", "--loop", "asyncio"],
        "cwd":  EVE_DIR,
        "env_extra": {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        "health_url": "http://127.0.0.1:8870/health",
        "port": 8870,
        "priority": 3,
        "depends_on": "memory",
        "restart_delay_s": 10,
        "max_restarts": 0,
        "critical": True,
    },
    {
        "name": "voice_sidecar",
        "cmd":  [PYTHON, "voice_sidecar.py"],
        "cwd":  EVE_DIR,
        "env_extra": {},
        "health_url": "http://127.0.0.1:8766/health",
        "port": 8766,
        "priority": 4,
        "depends_on": "eve_backend",
        "restart_delay_s": 8,
        "max_restarts": 0,
        "critical": False,
    },
    {
        "name": "lotus_forge_public",
        "cmd":  [PYTHON, "-m", "uvicorn", "public_app:app",
                 "--host", "0.0.0.0", "--port", "8875"],
        "cwd":  EVE_DIR,
        "env_extra": {"FORGE_PUBLIC_PASS": "Finalfantasy3181!"},
        "health_url": "http://127.0.0.1:8875/",
        "port": 8875,
        "priority": 5,
        "depends_on": "eve_backend",
        "restart_delay_s": 10,
        "max_restarts": 0,
        "critical": False,
    },
    {
        "name": "factory_daemon",
        "cmd":  [PYTHON, "factory_daemon.py"],
        "cwd":  EVE_DIR,
        "env_extra": {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        "health_url": None,  # process-alive only
        "port": None,
        "priority": 6,
        "depends_on": "eve_backend",
        "restart_delay_s": 30,
        "max_restarts": 10,
        "critical": False,
    },
    {
        "name": "eve_echo",
        "cmd":  [PYTHON, r"H:\Eve\bin\eve_echo.py"],
        "cwd":  r"H:\Eve\bin",
        "env_extra": {},
        "health_url": None,  # process-alive only; status via manifest
        "port": None,
        "priority": 10,      # last to start — memory must be live first
        "depends_on": "memory",
        "restart_delay_s": 15,
        "max_restarts": 0,
        "critical": False,
    },
]

# ── State ─────────────────────────────────────────────────────────────────────
_procs: dict[str, subprocess.Popen] = {}
_restart_counts: dict[str, list[float]] = {s["name"]: [] for s in SERVICES}
_shutdown = threading.Event()

ROLLING_WINDOW_S = 600   # 10-minute window for max_restarts count
CHECK_INTERVAL_S = 20    # how often to poll all services
STARTUP_TIMEOUT_S = 120  # max seconds to wait for a service to become healthy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _http_ok(url: str, timeout: int = 4) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _proc_alive(proc: subprocess.Popen) -> bool:
    return proc is not None and proc.poll() is None


def _is_healthy(svc: dict) -> bool:
    proc = _procs.get(svc["name"])
    # If proc is None the service was either not started yet or started externally.
    # In that case fall through to the health_url check.
    if proc is not None and not _proc_alive(proc):
        return False   # process we own has died
    if svc["health_url"]:
        return _http_ok(svc["health_url"])
    # No health URL and no owned proc — can't determine health, assume up
    return proc is not None


def _write_crash_envelope(svc_name: str, proc: subprocess.Popen, returncode: int):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CRASH_DIR / f"{svc_name}_{ts}.json"
    envelope = {
        "service": svc_name,
        "timestamp": ts,
        "returncode": returncode,
        "pid": proc.pid if proc else None,
        "restart_count": len(_restart_counts.get(svc_name, [])),
    }
    try:
        path.write_text(json.dumps(envelope, indent=2))
        log.warning("[CRASH] %s exited rc=%s — envelope: %s", svc_name, returncode, path)
    except Exception as e:
        log.error("[CRASH] Could not write envelope for %s: %s", svc_name, e)


def _open_log(svc_name: str, suffix: str = "") -> "open":
    tag = f"{svc_name}{suffix}"
    return open(str(LOG_DIR / f"{tag}.log"), "ab")


def _launch(svc: dict) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(svc.get("env_extra") or {})
    stdout_f = _open_log(svc["name"], "_out")
    stderr_f = _open_log(svc["name"], "_err")
    proc = subprocess.Popen(
        svc["cmd"],
        cwd=svc.get("cwd"),
        env=env,
        stdout=stdout_f,
        stderr=stderr_f,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    log.info("[START] %s  pid=%s  port=%s", svc["name"], proc.pid, svc.get("port", "–"))
    return proc


def _wait_healthy(svc: dict, timeout: int = STARTUP_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_healthy(svc):
            return True
        if not _proc_alive(_procs.get(svc["name"])):
            return False   # process died while waiting
        time.sleep(2)
    return False


def _throttled(svc: dict) -> bool:
    """Returns True if this service has exceeded max_restarts in rolling window."""
    max_r = svc.get("max_restarts", 0)
    if max_r == 0:
        return False
    now = time.time()
    counts = _restart_counts[svc["name"]]
    # Purge old timestamps
    _restart_counts[svc["name"]] = [t for t in counts if now - t < ROLLING_WINDOW_S]
    return len(_restart_counts[svc["name"]]) >= max_r


def _record_restart(svc_name: str):
    _restart_counts[svc_name].append(time.time())


# ── Startup ───────────────────────────────────────────────────────────────────

def _already_up(svc: dict) -> bool:
    """Check if a service is already healthy externally (before watchdog owned it)."""
    url = svc.get("health_url")
    if url:
        return _http_ok(url)
    return False


def startup_all():
    ordered = sorted(SERVICES, key=lambda s: s["priority"])
    for svc in ordered:
        if _shutdown.is_set():
            break
        name = svc["name"]

        # Skip if already healthy (e.g. launched by the bat before the watchdog started)
        if _already_up(svc):
            log.info("[SKIP] %s already healthy — watchdog will monitor it", name)
            # Register a sentinel so monitor loop tracks liveness via health_url
            _procs[name] = None
            continue

        # Wait for dependency
        dep = svc.get("depends_on")
        if dep:
            dep_svc = next((s for s in SERVICES if s["name"] == dep), None)
            if dep_svc:
                log.info("[WAIT] %s depends on %s — waiting...", name, dep)
                deadline = time.monotonic() + STARTUP_TIMEOUT_S
                while time.monotonic() < deadline and not _shutdown.is_set():
                    if _is_healthy(dep_svc):
                        break
                    time.sleep(2)
                else:
                    if not _is_healthy(dep_svc):
                        log.error("[SKIP] %s: dependency %s not healthy — skipping", name, dep)
                        continue

        log.info("[LAUNCH] %s ...", name)
        _procs[name] = _launch(svc)
        ok = _wait_healthy(svc, timeout=STARTUP_TIMEOUT_S)
        if ok:
            log.info("[READY] %s  ✓", name)
        else:
            log.warning("[TIMEOUT] %s did not become healthy within %ds", name, STARTUP_TIMEOUT_S)
            if svc.get("critical"):
                log.error("[CRITICAL] %s failed — watchdog continuing (will retry in monitor loop)", name)


# ── Monitor loop ──────────────────────────────────────────────────────────────

def monitor_loop():
    while not _shutdown.is_set():
        for svc in SERVICES:
            name = svc["name"]
            proc = _procs.get(name)

            if _is_healthy(svc):
                continue   # all good

            # Dead or unhealthy — write crash envelope if process exited
            if proc is not None and proc.poll() is not None:
                _write_crash_envelope(name, proc, proc.returncode)
                _procs[name] = None

            if _throttled(svc):
                log.warning("[THROTTLE] %s has crashed too many times — pausing restarts", name)
                continue

            delay = svc.get("restart_delay_s", 5)
            log.info("[RESTART] %s in %ds...", name, delay)
            time.sleep(delay)

            _record_restart(name)
            _procs[name] = _launch(svc)

        _shutdown.wait(timeout=CHECK_INTERVAL_S)


# ── Shutdown ──────────────────────────────────────────────────────────────────

def shutdown_all(signum=None, frame=None):
    log.info("[SHUTDOWN] Stopping all services...")
    _shutdown.set()
    for name, proc in _procs.items():
        if _proc_alive(proc):
            log.info("  Terminating %s (pid=%s)", name, proc.pid)
            try:
                proc.terminate()
            except Exception:
                pass
    time.sleep(3)
    for name, proc in _procs.items():
        if _proc_alive(proc):
            try:
                proc.kill()
            except Exception:
                pass
    log.info("[SHUTDOWN] Done.")
    sys.exit(0)


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, shutdown_all)
    signal.signal(signal.SIGINT,  shutdown_all)

    log.info("=" * 60)
    log.info("  Eve Watchdog starting  —  %s", datetime.datetime.now())
    log.info("  Python: %s", sys.executable)
    log.info("  Log dir: %s", LOG_DIR)
    log.info("  Crash dir: %s", CRASH_DIR)
    log.info("=" * 60)

    startup_all()
    monitor_loop()
