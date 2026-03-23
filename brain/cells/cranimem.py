"""
CraniMemCell — Gated Bounded Memory with Knowledge Graph Integration.

Architecture:
  - Bounded pool: max 500 memory slots with strength-based eviction
  - Strength decay: 0.995 per turn without access
  - Knowledge Graph: SQLite nodes+edges, entity/relation extraction via Haiku
  - On memory save: entities + relations → KG
  - On retrieve: vector search + 1-hop KG traversal
  - Scheduled replay: every 50 saves, replay top-10 by strength through KG extractor

REST endpoints:
  GET  /brain/cranimem/graph   — KG stats (node/edge counts, top nodes)
  GET  /brain/cranimem/slots   — memory pool stats
  POST /brain/cranimem/query   — direct query
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import anthropic

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

_HAIKU    = "claude-haiku-4-5-20251001"
_KG_DB    = Path(r"H:\Eve\memory\cranimem_kg.db")
_MEMORY_URL = "http://127.0.0.1:8767"
_MAX_SLOTS  = 500
_DECAY_RATE = 0.995
_REPLAY_EVERY = 50


# ── Memory slot ───────────────────────────────────────────────────────────────

@dataclass
class MemorySlot:
    id:           str
    content:      str
    strength:     float
    last_access:  float
    access_count: int
    created_at:   float
    session_id:   str = "cranimem"


# ── Knowledge Graph ───────────────────────────────────────────────────────────

class KnowledgeGraph:
    """SQLite-backed knowledge graph with nodes and directed edges."""

    def __init__(self, db_path: Path):
        self._db = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db))

    def _init_db(self) -> None:
        with self._lock, self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id              TEXT PRIMARY KEY,
                    label           TEXT NOT NULL,
                    type            TEXT DEFAULT 'entity',
                    properties_json TEXT DEFAULT '{}'
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id          TEXT PRIMARY KEY,
                    from_node   TEXT NOT NULL,
                    to_node     TEXT NOT NULL,
                    relation    TEXT NOT NULL,
                    weight      REAL DEFAULT 1.0,
                    created_at  REAL NOT NULL
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_node)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_edges_to   ON edges(to_node)")

    def upsert_node(self, label: str, node_type: str = "entity", props: dict = None) -> str:
        """Add or update a node. Returns node ID."""
        node_id = "n_" + uuid.uuid5(uuid.NAMESPACE_DNS, label.lower()).hex[:12]
        with self._lock, self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO nodes (id, label, type, properties_json)
                VALUES (?, ?, ?, ?)
            """, (node_id, label, node_type, json.dumps(props or {})))
        return node_id

    def add_edge(self, from_label: str, to_label: str, relation: str, weight: float = 1.0) -> None:
        from_id = self.upsert_node(from_label)
        to_id   = self.upsert_node(to_label)
        edge_id = "e_" + uuid.uuid5(uuid.NAMESPACE_DNS, f"{from_id}{relation}{to_id}").hex[:12]
        with self._lock, self._conn() as c:
            existing = c.execute("SELECT weight FROM edges WHERE id=?", (edge_id,)).fetchone()
            if existing:
                c.execute("UPDATE edges SET weight=? WHERE id=?", (min(existing[0] + 0.1, 5.0), edge_id))
            else:
                c.execute("""
                    INSERT INTO edges (id, from_node, to_node, relation, weight, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (edge_id, from_id, to_id, relation, weight, time.time()))

    def get_neighbors(self, label: str, max_hops: int = 1) -> list[dict]:
        """Get 1-hop neighbors of a node by label."""
        node_id = "n_" + uuid.uuid5(uuid.NAMESPACE_DNS, label.lower()).hex[:12]
        results = []
        with self._lock, self._conn() as c:
            rows = c.execute("""
                SELECT n.label, e.relation, e.weight
                FROM edges e
                JOIN nodes n ON (e.to_node = n.id OR e.from_node = n.id)
                WHERE (e.from_node = ? OR e.to_node = ?) AND n.id != ?
                ORDER BY e.weight DESC
                LIMIT 20
            """, (node_id, node_id, node_id)).fetchall()
            for label, rel, weight in rows:
                results.append({"label": label, "relation": rel, "weight": weight})
        return results

    def stats(self) -> dict:
        with self._lock, self._conn() as c:
            n_nodes = c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            n_edges = c.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            top = c.execute("""
                SELECT n.label, COUNT(e.id) as degree
                FROM nodes n
                LEFT JOIN edges e ON (e.from_node = n.id OR e.to_node = n.id)
                GROUP BY n.id ORDER BY degree DESC LIMIT 10
            """).fetchall()
        return {
            "node_count": n_nodes,
            "edge_count": n_edges,
            "top_nodes": [{"label": r[0], "degree": r[1]} for r in top],
        }


# ── CraniMemCell ──────────────────────────────────────────────────────────────

class CraniMemCell(BaseCell):
    name        = "cranimem"
    description = (
        "Gated bounded memory (500 slots) with strength decay + eviction. "
        "SQLite knowledge graph: entity/relation extraction from every save. "
        "Retrieval = vector search + 1-hop KG traversal. "
        "Scheduled replay reinforces top-10 memories."
    )
    color       = "#6d28d9"
    lazy        = False     # always-on
    position    = (4, 6)

    system_tier     = "online"
    hardware_req    = "CPU (SQLite KG) + ChromaDB (vector) + Claude Haiku (extraction)"
    research_basis  = (
        "Complementary Learning Systems (McClelland 1995), "
        "Bounded memory with LRU/strength-based eviction, "
        "Knowledge Graph integration for structured memory traversal"
    )
    build_notes     = (
        "LIVE: 500-slot pool, strength decay 0.995/turn, KG extraction via Haiku. "
        "Replay every 50 saves reinforces top-10 strongest memories. "
        "GET /brain/cranimem/graph | GET /brain/cranimem/slots | POST /brain/cranimem/query"
    )
    framework_layer = "AI & ML"

    def __init__(self):
        super().__init__()
        self._slots:   dict[str, MemorySlot] = {}   # id → slot
        self._kg:      Optional[KnowledgeGraph] = None
        self._client:  Optional[anthropic.Anthropic] = None
        self._lock     = threading.Lock()
        self._save_count = 0
        self._decay_thread: Optional[threading.Thread] = None

    async def boot(self) -> None:
        _KG_DB.parent.mkdir(parents=True, exist_ok=True)
        self._kg     = KnowledgeGraph(_KG_DB)
        self._client = anthropic.Anthropic()
        self._start_decay_loop()
        logger.info("[CraniMem] Cell online — KG at %s", _KG_DB)

    def _start_decay_loop(self) -> None:
        """Background decay: every 60 seconds apply strength decay to all non-accessed slots."""
        def _loop():
            while True:
                time.sleep(60)
                try:
                    self._apply_decay()
                except Exception as e:
                    logger.debug("[CraniMem] Decay error: %s", e)
        t = threading.Thread(target=_loop, daemon=True, name="cranimem-decay")
        t.start()
        self._decay_thread = t

    def _apply_decay(self) -> None:
        with self._lock:
            for slot in self._slots.values():
                if time.time() - slot.last_access > 60:
                    slot.strength *= _DECAY_RATE
                    slot.strength = max(slot.strength, 0.001)

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, content: str, session_id: str = "cranimem") -> str:
        """Save a memory. Evict weakest slot if pool is full. Returns slot ID."""
        slot_id = "c_" + uuid.uuid4().hex[:12]
        surprise = self._compute_surprise(content)

        slot = MemorySlot(
            id=slot_id, content=content,
            strength=surprise, last_access=time.time(),
            access_count=0, created_at=time.time(),
            session_id=session_id,
        )

        with self._lock:
            # Evict if full
            if len(self._slots) >= _MAX_SLOTS:
                weakest_id = min(self._slots, key=lambda k: self._slots[k].strength)
                del self._slots[weakest_id]
                logger.debug("[CraniMem] Evicted slot %s", weakest_id)
            self._slots[slot_id] = slot
            self._save_count += 1
            count = self._save_count

        # KG extraction (background)
        threading.Thread(
            target=self._extract_to_kg,
            args=(content,),
            daemon=True,
        ).start()

        # Replay every N saves
        if count % _REPLAY_EVERY == 0:
            threading.Thread(target=self._replay_top_memories, daemon=True).start()

        # Also push to ChromaDB memory service
        try:
            import requests
            requests.post(
                f"{_MEMORY_URL}/save",
                json={"user_input": "[CraniMem]", "eve_response": content,
                      "session_id": session_id},
                timeout=2,
            )
        except Exception:
            pass

        return slot_id

    def _compute_surprise(self, content: str) -> float:
        """Simple surprise: fraction of novel words vs existing slot contents."""
        if not self._slots:
            return 0.8

        import re
        new_words = set(re.findall(r"\b[a-z]{3,}\b", content.lower()))
        if not new_words:
            return 0.5

        all_words: set[str] = set()
        with self._lock:
            for s in list(self._slots.values())[-50:]:
                all_words |= set(re.findall(r"\b[a-z]{3,}\b", s.content.lower()))

        novel = new_words - all_words
        return min(len(novel) / max(len(new_words), 1), 1.0)

    def _extract_to_kg(self, content: str) -> None:
        """Extract entities and relations from content → add to KG."""
        if not self._client or not self._kg:
            return
        try:
            msg = self._client.messages.create(
                model=_HAIKU,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Extract entities and relationships from this text.\n\n{content[:1000]}\n\n"
                        "Return ONLY a JSON array of: "
                        '[{"from": "entity1", "relation": "relation_name", "to": "entity2"}]\n'
                        "Max 10 triples. Use short, lowercase labels."
                    ),
                }],
            )
            raw = msg.content[0].text.strip()
            # Strip code fences
            import re
            if "```" in raw:
                m = re.search(r"\[.*\]", raw, re.DOTALL)
                raw = m.group(0) if m else "[]"
            triples = json.loads(raw)
            for t in triples:
                if isinstance(t, dict) and "from" in t and "to" in t and "relation" in t:
                    self._kg.add_edge(t["from"], t["to"], t["relation"])
        except Exception as e:
            logger.debug("[CraniMem] KG extraction error: %s", e)

    def _replay_top_memories(self) -> None:
        """Replay top-10 strongest memories through KG extractor to reinforce connections."""
        with self._lock:
            top = sorted(self._slots.values(), key=lambda s: -s.strength)[:10]
        for slot in top:
            self._extract_to_kg(slot.content)
            time.sleep(0.5)
        logger.info("[CraniMem] Replay pass completed for top %d memories", len(top))

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = 5) -> dict:
        """
        Retrieve memories: vector search + KG traversal.
        Returns {memories: [...], kg_context: [...]}
        """
        # Vector search from ChromaDB
        chroma_results = ""
        try:
            import requests
            r = requests.post(
                f"{_MEMORY_URL}/inject",
                json={"query": query_text, "top_k": top_k, "threshold": 0.3},
                timeout=3,
            )
            chroma_results = r.json().get("injection", "")
        except Exception:
            pass

        # Warm up slot access counts
        import re
        query_words = set(re.findall(r"\b[a-z]{3,}\b", query_text.lower()))
        matching_slots = []
        with self._lock:
            for slot in self._slots.values():
                slot_words = set(re.findall(r"\b[a-z]{3,}\b", slot.content.lower()))
                overlap = len(query_words & slot_words) / max(len(query_words), 1)
                if overlap > 0.2:
                    slot.last_access = time.time()
                    slot.access_count += 1
                    slot.strength = min(slot.strength + 0.1, 1.0)
                    matching_slots.append({"content": slot.content[:200], "strength": slot.strength})

        matching_slots.sort(key=lambda x: -x["strength"])

        # KG traversal: extract key entities from query, find 1-hop neighbors
        kg_context = []
        if self._kg:
            import re as _re
            nouns = [w for w in _re.findall(r"\b[A-Z][a-z]+\b", query_text)][:3]
            for noun in nouns:
                neighbors = self._kg.get_neighbors(noun)
                if neighbors:
                    kg_context.append({"entity": noun, "neighbors": neighbors[:5]})

        return {
            "chroma_injection": chroma_results,
            "slot_matches":     matching_slots[:top_k],
            "kg_context":       kg_context,
        }

    # ── Stats ──────────────────────────────────────────────────────────────────

    def pool_stats(self) -> dict:
        with self._lock:
            slots = list(self._slots.values())
        if not slots:
            return {"total": 0, "avg_strength": 0, "min_strength": 0, "max_strength": 0, "save_count": self._save_count}
        strengths = [s.strength for s in slots]
        return {
            "total":        len(slots),
            "capacity":     _MAX_SLOTS,
            "avg_strength": round(sum(strengths) / len(strengths), 4),
            "min_strength": round(min(strengths), 4),
            "max_strength": round(max(strengths), 4),
            "save_count":   self._save_count,
        }

    async def process(self, ctx: CellContext) -> Any:
        result = self.query(ctx.message)
        ctx.memory_injection = (ctx.memory_injection or "") + "\n\n[CraniMem]\n" + result.get("chroma_injection", "")
        return result

    def health(self) -> dict:
        return {
            "status":     self._status.value,
            "pool_slots": len(self._slots),
            "kg_ready":   self._kg is not None,
            "save_count": self._save_count,
        }
