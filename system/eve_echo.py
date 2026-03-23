"""
Eve Echo * Live Mirror + Instant Restore System
================================================
H:\\Eve\\bin\\eve_echo.py

Runs as a background daemon alongside the watchdog.
Every N seconds it takes a consistent snapshot of every live Eve database
and writes it to H:\\Eve\\shadow\\ * a hot-standby mirror that can become
the real primary in seconds.

Architecture
------------
  Primary  : H:\\Eve\\memory\\          * live (ChromaDB + all SQLite DBs)
  Shadow   : H:\\Eve\\shadow\\memory\\  * echo (consistent point-in-time copies)
  Manifest : H:\\Eve\\shadow\\echo_manifest.json  * what was echoed and when

SQLite safety
-------------
  Uses sqlite3.backup() API * produces a byte-perfect consistent snapshot
  even while the DB is being written to.  Never touches the WAL mid-transaction.

ChromaDB safety
---------------
  chroma.sqlite3   -> sqlite3.backup()
  binary index files (*.bin, *.pickle) -> copied only when no write is detected
  for 2 consecutive polls (quiescence guard).

Modes
-----
  python eve_echo.py               * start daemon (loops forever)
  python eve_echo.py --status      * show last echo times + divergence
  python eve_echo.py --promote     * make shadow the primary (swaps dirs, restarts memory svc)
  python eve_echo.py --pulse       * run one echo cycle and exit

Usage in watchdog
-----------------
  Add to SERVICES in eve_watchdog.py (priority 10, no health_url, no restart needed).
  Or launch from H:\\Eve\\bin\\launch_eve.bat after watchdog starts.
"""

import argparse
import datetime
import faulthandler
import json
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# -- Crash safety --------------------------------------------------------------
_CRASH_DIR = Path("H:/Eve/logs/crash")
_CRASH_DIR.mkdir(parents=True, exist_ok=True)
faulthandler.enable(file=open(str(_CRASH_DIR / "eve_echo_fault.log"), "a"), all_threads=True)

# -- Logging -------------------------------------------------------------------
LOG_DIR = Path("H:/Eve/logs/api")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "eve_echo.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("eve.echo")

# -- Paths ---------------------------------------------------------------------
PRIMARY      = Path("H:/Eve/memory")
SHADOW       = Path("H:/Eve/shadow/memory")
MANIFEST     = Path("H:/Eve/shadow/echo_manifest.json")
PYTHON       = Path(r"F:\pinokio\bin\miniconda\envs\myllm\python.exe")
MEMORY_SVC   = Path(r"H:\Eve\bin\memory_service.py")
MEMORY_SVC_CWD = Path(r"H:\Eve\bin")

# Echo interval: how often to run a full sync cycle (seconds)
ECHO_INTERVAL_S = 30

# Quiescence guard for ChromaDB binary files:
# if mtime changed between two polls this many seconds apart, skip this cycle
CHROMA_QUIESCE_S = 8

# -- SQLite databases to mirror (relative to PRIMARY) -------------------------
SQLITE_DBS = [
    "memory.db",
    "eve_immunity.db",
    "cranimem_kg.db",
    "learning_lab.db",
    "public_eve.db",
    "tcg_oracle.db",
    "chroma/chroma.sqlite3",
]

# -- ChromaDB binary index files -----------------------------------------------
CHROMA_BINARY_GLOBS = [
    "chroma/**/*.bin",
    "chroma/**/*.pickle",
]

# -- Manifest ------------------------------------------------------------------
def _load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text())
        except Exception:
            pass
    return {"echoes": {}, "last_full_echo": None, "total_cycles": 0}


def _save_manifest(m: dict):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, default=str))


# -- SQLite backup -------------------------------------------------------------
def _backup_sqlite(src: Path, dst: Path) -> bool:
    """Consistent backup using sqlite3.backup() API. Returns True on success."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        dst_conn = sqlite3.connect(str(dst))
        with dst_conn:
            src_conn.backup(dst_conn)
        src_conn.close()
        dst_conn.close()
        return True
    except Exception as e:
        log.warning("[ECHO] SQLite backup failed %s -> %s: %s", src.name, dst, e)
        return False


# -- Binary file copy (quiescence-guarded) ------------------------------------
def _snapshot_binary(src: Path, dst: Path) -> bool:
    """Copy a binary file, but only if it hasn't changed recently."""
    if not src.exists():
        return False
    mtime_now = src.stat().st_mtime
    time.sleep(CHROMA_QUIESCE_S)
    mtime_after = src.stat().st_mtime
    if mtime_after != mtime_now:
        log.debug("[ECHO] %s still changing * skipping this cycle", src.name)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(str(src), str(dst))
        return True
    except Exception as e:
        log.warning("[ECHO] Binary copy failed %s: %s", src.name, e)
        return False


# -- Full echo cycle -----------------------------------------------------------
def echo_cycle(manifest: dict) -> dict:
    """Run one complete mirror cycle. Returns updated manifest."""
    cycle_start = time.time()
    successes = 0
    failures = 0

    # 1. Mirror all SQLite DBs
    for rel in SQLITE_DBS:
        src = PRIMARY / rel
        dst = SHADOW / rel
        ok = _backup_sqlite(src, dst)
        ts = datetime.datetime.now().isoformat()
        if ok:
            sz = dst.stat().st_size if dst.exists() else 0
            manifest["echoes"][rel] = {"ok": True, "at": ts, "bytes": sz}
            successes += 1
            log.debug("[ECHO] OK %s  (%d KB)", rel, sz // 1024)
        else:
            manifest["echoes"][rel] = {"ok": False, "at": ts}
            failures += 1

    # 2. Mirror ChromaDB binary index files (quiescence-guarded)
    for glob_pat in CHROMA_BINARY_GLOBS:
        for src in PRIMARY.glob(glob_pat.replace("chroma/", "", 1).replace("chroma\\", "", 1)):
            # Reconstruct relative path under chroma/
            rel_parts = src.relative_to(PRIMARY)
            dst = SHADOW / rel_parts
            ok = _snapshot_binary(src, dst)
            key = str(rel_parts)
            ts = datetime.datetime.now().isoformat()
            if ok:
                manifest["echoes"][key] = {"ok": True, "at": ts}
                successes += 1
            else:
                failures += 1

    manifest["last_full_echo"] = datetime.datetime.now().isoformat()
    manifest["total_cycles"] = manifest.get("total_cycles", 0) + 1
    manifest["last_duration_s"] = round(time.time() - cycle_start, 2)
    manifest["successes"] = successes
    manifest["failures"] = failures

    if failures == 0:
        log.info("[ECHO] Cycle %d complete * %d files mirrored in %.1fs",
                 manifest["total_cycles"], successes, manifest["last_duration_s"])
    else:
        log.warning("[ECHO] Cycle %d * %d ok, %d FAILED",
                    manifest["total_cycles"], successes, failures)
    return manifest


# -- Status --------------------------------------------------------------------
def show_status():
    m = _load_manifest()
    print("\n== Eve Echo Status ==========================================")
    print(f"  Primary  : {PRIMARY}")
    print(f"  Shadow   : {SHADOW}")
    print(f"  Cycles   : {m.get('total_cycles', 0)}")
    print(f"  Last echo: {m.get('last_full_echo', 'never')}")
    print(f"  Duration : {m.get('last_duration_s', '?')}s")
    print()
    print("  File                         Last Echo              OK")
    print("  " + "-" * 60)
    for rel, info in sorted(m.get("echoes", {}).items()):
        at   = info.get("at", "never")[:19]
        ok   = "OK" if info.get("ok") else "X"
        sz   = f"  {info['bytes']//1024:>6} KB" if "bytes" in info else ""
        print(f"  {rel:<30} {at}  {ok}{sz}")
    print()

    # Divergence: time since last successful echo cycle
    print("  Divergence check:")
    last_echo_str = m.get("last_full_echo")
    if last_echo_str:
        import datetime as _dt
        last_echo = _dt.datetime.fromisoformat(last_echo_str)
        age_s = (_dt.datetime.now() - last_echo).total_seconds()
        if age_s < ECHO_INTERVAL_S * 3:
            print(f"  OK  Shadow is {age_s:.0f}s old -- within echo window ({ECHO_INTERVAL_S}s interval)")
        else:
            print(f"  WARN  Shadow is {age_s:.0f}s old -- echo daemon may be down")
    else:
        print("  WARN  No echo cycles recorded yet -- run daemon or --pulse")

    # Quick file presence check
    missing = [r for r in SQLITE_DBS if not (SHADOW / r).exists()]
    if missing:
        print(f"  WARN  Not yet mirrored: {', '.join(missing)}")
    print("=" * 54 + "\n")


# -- Promote shadow -> primary --------------------------------------------------
def promote():
    """
    Swap shadow into the primary slot.
    Steps:
      1. Stop memory service (port 8767)
      2. Rename H:\\Eve\\memory -> H:\\Eve\\memory.pre_promote_<ts>
      3. Copy H:\\Eve\\shadow\\memory -> H:\\Eve\\memory
      4. Restart memory service
      5. Verify health
    """
    import urllib.request

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = Path(f"H:/Eve/memory.pre_promote_{ts}")

    print(f"\n== Eve Echo PROMOTE ================================")
    print(f"  Shadow  : {SHADOW}")
    print(f"  Primary : {PRIMARY}  ->  {backup_name}  (old backup)")
    print(f"  New     : {SHADOW}   ->  {PRIMARY}")
    print()

    if not SHADOW.exists():
        print("  X Shadow does not exist yet * run at least one echo cycle first")
        sys.exit(1)

    # 1. Kill memory service
    print("  [1/5] Stopping memory service (port 8767)...")
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-NetTCPConnection -LocalPort 8767 -EA SilentlyContinue | "
         "Select-Object -ExpandProperty OwningProcess | "
         "ForEach-Object { Stop-Process -Id $_ -Force -EA SilentlyContinue }"],
        capture_output=True
    )
    time.sleep(3)

    # 2. Move primary aside
    print(f"  [2/5] Archiving primary -> {backup_name.name}...")
    if PRIMARY.exists():
        PRIMARY.rename(backup_name)

    # 3. Copy shadow into primary slot
    print(f"  [3/5] Promoting shadow -> {PRIMARY}...")
    shutil.copytree(str(SHADOW), str(PRIMARY))

    # 4. Restart memory service
    print("  [4/5] Starting memory service...")
    subprocess.Popen(
        [str(PYTHON), str(MEMORY_SVC)],
        cwd=str(MEMORY_SVC_CWD),
        stdout=open(str(LOG_DIR / "memory_out.log"), "ab"),
        stderr=open(str(LOG_DIR / "memory_err.log"), "ab"),
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    time.sleep(5)

    # 5. Health check
    print("  [5/5] Verifying health...")
    for attempt in range(10):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8767/health", timeout=4) as r:
                if r.status == 200:
                    data = json.loads(r.read())
                    print(f"  OK Memory service UP * {data.get('vectors', '?')} vectors")
                    print(f"\n  PROMOTE COMPLETE.  Old primary archived at {backup_name}")
                    print("  Eve is now running on the promoted shadow.\n")
                    return
        except Exception:
            pass
        time.sleep(2)

    print("  X Memory service did not respond after promote.")
    print(f"    Old primary is safe at: {backup_name}")
    print("    To rollback: rename memory -> memory.failed, rename backup -> memory, restart svc\n")
    sys.exit(1)


# -- Daemon --------------------------------------------------------------------
_running = True

def _handle_signal(signum, frame):
    global _running
    log.info("[ECHO] Signal %s received * stopping after current cycle", signum)
    _running = False

def run_daemon():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    log.info("=" * 60)
    log.info("  Eve Echo daemon starting * %s", datetime.datetime.now())
    log.info("  Primary : %s", PRIMARY)
    log.info("  Shadow  : %s", SHADOW)
    log.info("  Interval: %ds", ECHO_INTERVAL_S)
    log.info("=" * 60)

    SHADOW.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()

    while _running:
        try:
            manifest = echo_cycle(manifest)
            _save_manifest(manifest)
        except Exception as e:
            log.error("[ECHO] Cycle error: %s", e, exc_info=True)
        if _running:
            time.sleep(ECHO_INTERVAL_S)

    log.info("[ECHO] Daemon stopped cleanly.")


# -- Entry ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eve Echo * live mirror daemon")
    parser.add_argument("--status",  action="store_true", help="Show mirror status and exit")
    parser.add_argument("--promote", action="store_true", help="Promote shadow to primary and exit")
    parser.add_argument("--pulse",   action="store_true", help="Run one echo cycle and exit")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.promote:
        promote()
    elif args.pulse:
        SHADOW.mkdir(parents=True, exist_ok=True)
        m = _load_manifest()
        m = echo_cycle(m)
        _save_manifest(m)
        show_status()
    else:
        run_daemon()
