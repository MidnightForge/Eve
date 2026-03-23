"""
LearningLabCell — brain/cells/learning_lab.py

Handles the Learning Lab UI:
  - Dream injection queue (thoughts injected into tonight's dreams)
  - School challenges (submitted to the training factory)
  - Training suggestions (what Eve should learn next)

SQLite DB: H:/Eve/memory/learning_lab.db

REST endpoints (added via router.py):
  GET  /learning/queue           — returns dream queue + suggestion list
  POST /learning/inject-dream    — injects content into dream queue
  POST /learning/challenge       — adds challenge to training factory
  GET  /learning/status          — factory + dream queue status summary
  GET  /learning/suggestions     — list of suggestions
  POST /learning/suggestion      — add a new suggestion
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

_DB_PATH = Path("H:/Eve/memory/learning_lab.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS dream_queue (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            content  TEXT NOT NULL,
            tags     TEXT DEFAULT '',
            created  REAL NOT NULL,
            injected INTEGER DEFAULT 0,
            injected_at REAL
        );
        CREATE TABLE IF NOT EXISTS suggestions (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            content  TEXT NOT NULL,
            tags     TEXT DEFAULT '',
            created  REAL NOT NULL,
            status   TEXT DEFAULT 'pending'
        );
        CREATE TABLE IF NOT EXISTS challenges (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt     TEXT NOT NULL,
            difficulty TEXT DEFAULT 'medium',
            domain     TEXT DEFAULT 'reasoning',
            created    REAL NOT NULL,
            status     TEXT DEFAULT 'pending'
        );
        """)


# Initialise DB on import
try:
    _init_db()
except Exception as e:
    logger.warning(f"LearningLabCell DB init failed: {e}")


class LearningLabCell(BaseCell):
    name        = "learning_lab"
    description = "Learning Lab — dream injection, school challenges, training suggestions"
    color       = "#a855f7"
    lazy        = True
    position    = (5, 3)

    # ── Dream queue ────────────────────────────────────────────────────────

    def inject_dream(self, content: str, tags: str = "") -> dict:
        try:
            with _get_conn() as conn:
                cur = conn.execute(
                    "INSERT INTO dream_queue (content, tags, created) VALUES (?, ?, ?)",
                    (content.strip(), tags.strip(), time.time()),
                )
                row_id = cur.lastrowid
            logger.info(f"LearningLab: dream injected id={row_id}")
            return {"ok": True, "id": row_id, "message": "Dream queued for tonight."}
        except Exception as e:
            logger.error(f"LearningLab inject_dream error: {e}")
            return {"ok": False, "error": str(e)}

    def get_dream_queue(self) -> list[dict]:
        try:
            with _get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM dream_queue ORDER BY created DESC LIMIT 50"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"LearningLab get_dream_queue error: {e}")
            return []

    # ── Suggestions ────────────────────────────────────────────────────────

    def add_suggestion(self, content: str, tags: str = "") -> dict:
        try:
            with _get_conn() as conn:
                cur = conn.execute(
                    "INSERT INTO suggestions (content, tags, created) VALUES (?, ?, ?)",
                    (content.strip(), tags.strip(), time.time()),
                )
                row_id = cur.lastrowid
            return {"ok": True, "id": row_id, "message": "Suggestion saved."}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_suggestions(self) -> list[dict]:
        try:
            with _get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM suggestions ORDER BY created DESC LIMIT 100"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            return []

    # ── Challenges ─────────────────────────────────────────────────────────

    def add_challenge(self, prompt: str, difficulty: str = "medium", domain: str = "reasoning") -> dict:
        try:
            with _get_conn() as conn:
                cur = conn.execute(
                    "INSERT INTO challenges (prompt, difficulty, domain, created) VALUES (?, ?, ?, ?)",
                    (prompt.strip(), difficulty, domain, time.time()),
                )
                row_id = cur.lastrowid
            return {"ok": True, "id": row_id, "message": "Challenge submitted."}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_challenges(self) -> list[dict]:
        try:
            with _get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM challenges ORDER BY created DESC LIMIT 50"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            return []

    # ── Status summary ──────────────────────────────────────────────────────

    def get_status(self) -> dict:
        try:
            with _get_conn() as conn:
                dream_total   = conn.execute("SELECT COUNT(*) FROM dream_queue").fetchone()[0]
                dream_pending = conn.execute("SELECT COUNT(*) FROM dream_queue WHERE injected=0").fetchone()[0]
                sugg_total    = conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0]
                chal_total    = conn.execute("SELECT COUNT(*) FROM challenges").fetchone()[0]
                latest_dream  = conn.execute(
                    "SELECT content, created FROM dream_queue ORDER BY created DESC LIMIT 1"
                ).fetchone()
            return {
                "ok": True,
                "dream_queue": {"total": dream_total, "pending": dream_pending},
                "suggestions": {"total": sugg_total},
                "challenges":  {"total": chal_total},
                "latest_dream": dict(latest_dream) if latest_dream else None,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── BaseCell interface ─────────────────────────────────────────────────

    async def process(self, ctx: CellContext) -> Any:
        # Routing support: if the message mentions dream/challenge/suggest keywords
        msg = ctx.message.lower()
        if "inject" in msg or "dream" in msg:
            return "Dream injection is handled through the Learning Lab UI tab."
        elif "challenge" in msg or "school" in msg:
            return "School challenges can be submitted in the Learning Lab tab."
        return {"learning_lab": "ready"}

    def health(self) -> dict:
        try:
            status = self.get_status()
            return {"status": "active", "db": str(_DB_PATH), **status}
        except Exception as e:
            return {"status": "error", "error": str(e)}
