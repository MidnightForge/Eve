"""
MemoryCell — ChromaDB + Mem0AI + HEMA semantic memory.

Three-layer memory architecture:
  Layer 1 (ChromaDB service @ :8767): raw conversation storage + vector retrieval
  Layer 2 (Mem0AI): intelligent consolidation — extracts salient facts, deduplicates,
    compresses long conversation history into clean structured memories.
    26% accuracy boost, 90% token reduction vs raw ChromaDB retrieval alone.
  Layer 3 (HEMA): Hippocampal Episodic Memory Architecture — running conversation
    summary (compact, always-injected) + surprise gate (skip storing boring exchanges).
    Based on arXiv:2504.16754 — dual compact+vector memory.

Surprise Gate (arXiv:2501.00663 Titans inspiration):
  Before saving, compute surprise score = semantic novelty vs recent exchanges.
  Only store exchanges with surprise > threshold (saves VRAM, reduces noise).

HEMA Running Summary:
  Maintains a rolling ~200-word summary of the current conversation.
  Updated every N turns via Haiku. Always injected as context prefix.
  Acts as the "hippocampal index" pointing to relevant long-term memories.
"""

import logging
import asyncio
import hashlib
import threading
import time
import requests as _req
from typing import Optional

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

MEMORY_URL = "http://127.0.0.1:8767"

# ── HEMA Config ───────────────────────────────────────────────────────────────
_HEMA_SUMMARY_TURNS   = 6     # regenerate running summary every N turns
_HEMA_SUMMARY_WORDS   = 220   # target words for running summary
_SURPRISE_THRESHOLD   = 0.18  # below this = boring, skip full storage
_SURPRISE_VOCAB_SIZE  = 200   # top-N words for surprise scoring
_HEMA_MODEL           = "claude-haiku-4-5-20251001"

# ── Mem0AI (optional — graceful degradation if unavailable) ───────────────────
_mem0 = None

def _init_mem0():
    global _mem0
    if _mem0 is not None:
        return _mem0
    try:
        from mem0 import Memory
        import os
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "eve_mem0_consolidated",
                    "path": os.path.join(os.path.expanduser("~"), "eve", "mem0_chroma"),
                }
            },
            "llm": {
                "provider": "anthropic",
                "config": {
                    "model": "claude-haiku-4-5-20251001",
                    "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "multi-qa-MiniLM-L6-cos-v1"}
            }
        }
        _mem0 = Memory.from_config(config)
        logger.info("[MemoryCell] Mem0AI consolidation layer online")
    except Exception as e:
        logger.debug("[MemoryCell] Mem0AI unavailable (graceful): %s", e)
        _mem0 = False   # False = tried and failed, don't retry
    return _mem0


class MemoryCell(BaseCell):
    name        = "memory"
    description = (
        "Three-layer semantic memory — ChromaDB raw retrieval + "
        "Mem0AI consolidation + HEMA running summary with surprise gate."
    )
    color       = "#0891b2"
    lazy        = False      # always prefetch
    position    = (1, 1)

    # ── HEMA state ────────────────────────────────────────────────────────────
    _hema_summary:      str  = ""    # rolling conversation summary
    _hema_turn_count:   int  = 0     # turns since last summary update
    _hema_recent:       list = []    # last 20 exchange fingerprints for surprise
    _hema_vocab:        dict = {}    # word → count for surprise TF scoring

    # ── Surprise Gate statistics ───────────────────────────────────────────────
    _saved_count:       int  = 0
    _skipped_count:     int  = 0

    # ── MemoryOS Heat Score system ─────────────────────────────────────────────
    # Heat scores keyed by memory fingerprint (md5[:8])
    # HOT  > 0.7 : always inject
    # WARM 0.3–0.7: normal ChromaDB retrieval
    # COLD < 0.3 : stored but ranked lower
    _heat_scores:       dict = {}    # fingerprint → float[0,1]
    _heat_lock:         object = None  # set in __init__

    _HOT_THRESHOLD  = 0.7
    _WARM_THRESHOLD = 0.3
    _HEAT_DECAY     = 0.97   # per-turn decay multiplier
    _HEAT_ACCESS    = 0.10   # bump on retrieval
    _HEAT_DECAY_INTERVAL = 60  # seconds

    def __init__(self):
        # MemoryCell uses class-level state for some fields (legacy pattern from codebase).
        # We add instance-level heat lock here.
        self._heat_lock = threading.Lock()
        super().__init__()
        self._heat_decay_thread = None
        self._start_heat_decay()

    def _start_heat_decay(self) -> None:
        """Background thread: decay heat scores every 60 seconds."""
        def _loop():
            while True:
                time.sleep(self._HEAT_DECAY_INTERVAL)
                try:
                    self._decay_all_heat()
                except Exception:
                    pass
        t = threading.Thread(target=_loop, daemon=True, name="memory-heat-decay")
        t.start()
        self._heat_decay_thread = t

    def _decay_all_heat(self) -> None:
        with self._heat_lock:
            for fp in list(MemoryCell._heat_scores.keys()):
                MemoryCell._heat_scores[fp] = max(
                    MemoryCell._heat_scores[fp] * self._HEAT_DECAY, 0.0
                )

    async def boot(self) -> None:
        # Quick health check in thread — must not block event loop
        try:
            r = await asyncio.to_thread(_req.get, f"{MEMORY_URL}/health", timeout=2)
            if r.status_code == 200:
                logger.info("[MemoryCell] ChromaDB memory service online")
            else:
                self._status = CellStatus.DEGRADED
        except Exception as exc:
            self._status = CellStatus.DEGRADED
            logger.warning("[MemoryCell] Memory service unreachable: %s", exc)

    # ── HEMA: Running summary ─────────────────────────────────────────────────

    def _update_vocab(self, text: str) -> None:
        """Update word frequency vocab for surprise scoring."""
        import re
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        for w in words:
            self._hema_vocab[w] = self._hema_vocab.get(w, 0) + 1
        # Keep only top _SURPRISE_VOCAB_SIZE words to bound memory
        if len(self._hema_vocab) > _SURPRISE_VOCAB_SIZE * 2:
            sorted_vocab = sorted(self._hema_vocab.items(), key=lambda x: -x[1])
            self._hema_vocab = dict(sorted_vocab[:_SURPRISE_VOCAB_SIZE])

    def _surprise_score(self, user_input: str, eve_response: str) -> float:
        """
        Compute surprise score in [0, 1] using TF-based novelty.
        High score = novel content (should store). Low score = repetitive (skip).

        Formula: fraction of words in this exchange that are rare in recent history.
        Rare = below median frequency in _hema_vocab.
        """
        import re
        if not self._hema_vocab:
            return 1.0  # First exchange — always surprising

        text = (user_input + " " + eve_response).lower()
        words = re.findall(r"\b[a-z]{3,}\b", text)
        if not words:
            return 0.5

        median_freq = sorted(self._hema_vocab.values())[len(self._hema_vocab) // 2]
        novel_count = sum(
            1 for w in words
            if self._hema_vocab.get(w, 0) <= median_freq
        )
        return novel_count / len(words)

    def _fingerprint(self, text: str) -> str:
        return hashlib.md5(text[:200].encode()).hexdigest()[:8]

    async def _refresh_hema_summary(self, recent_exchanges: list[dict]) -> None:
        """
        Regenerate the running conversation summary via Haiku.
        Called every _HEMA_SUMMARY_TURNS turns.
        """
        if not recent_exchanges:
            return
        try:
            import anthropic
            client = anthropic.Anthropic()
            exchange_text = "\n".join(
                f"User: {ex['user']}\nEve: {ex['eve']}"
                for ex in recent_exchanges[-8:]
            )
            if self._hema_summary:
                prompt = (
                    f"Previous summary:\n{self._hema_summary}\n\n"
                    f"New exchanges:\n{exchange_text}\n\n"
                    f"Update the summary to include the new exchanges. "
                    f"Keep it under {_HEMA_SUMMARY_WORDS} words. "
                    "Preserve key facts, decisions, and context."
                )
            else:
                prompt = (
                    f"Conversation:\n{exchange_text}\n\n"
                    f"Write a concise summary under {_HEMA_SUMMARY_WORDS} words "
                    "capturing the key topics, decisions, and important context."
                )
            loop = asyncio.get_event_loop()
            def _call():
                r = client.messages.create(
                    model=_HEMA_MODEL,
                    max_tokens=300,
                    system=(
                        "You are a conversation summarizer. Create a dense, "
                        "factual summary preserving all important context. "
                        "No filler, no pleasantries — just key information."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return r.content[0].text.strip()

            new_summary = await asyncio.wait_for(
                loop.run_in_executor(None, _call), timeout=8.0
            )
            self._hema_summary = new_summary
            logger.debug("[MemoryCell] HEMA summary updated (%d words)", len(new_summary.split()))
        except Exception as e:
            logger.debug("[MemoryCell] HEMA summary refresh failed: %s", e)

    async def process(self, ctx: CellContext) -> str:
        """
        Returns blended memory injection (three layers):
          - HEMA running summary (hippocampal index — always injected first)
          - Mem0AI consolidated facts (intelligent deduplication)
          - ChromaDB raw retrieval (vector similarity)
        """
        loop = asyncio.get_event_loop()
        # Run ChromaDB + Mem0AI retrievals in parallel
        chroma_task = loop.run_in_executor(None, self._fetch, ctx.message)
        mem0_task   = loop.run_in_executor(None, self.mem0_search, ctx.message)
        try:
            chroma_result, mem0_result = await asyncio.wait_for(
                asyncio.gather(chroma_task, mem0_task, return_exceptions=True),
                timeout=3.5,
            )
        except Exception:
            chroma_result, mem0_result = "", ""

        chroma_str = chroma_result if isinstance(chroma_result, str) else ""
        mem0_str   = mem0_result   if isinstance(mem0_result,   str) else ""

        # Layer 3 (HEMA): running summary as hippocampal index prefix
        hema_str = ""
        if self._hema_summary:
            hema_str = f"[Conversation Context]\n{self._hema_summary}"

        # Blend: HEMA summary → Mem0AI facts → ChromaDB context
        parts = [p for p in [hema_str, mem0_str, chroma_str] if p]
        combined = "\n\n".join(parts)
        ctx.memory_injection = combined
        return combined

    def _fetch(self, query: str) -> str:
        try:
            r = _req.post(
                f"{MEMORY_URL}/inject",
                json={"query": query, "top_k": 5, "threshold": 0.45},
                timeout=2.0,
            )
            injection = r.json().get("injection", "")
        except Exception:
            injection = ""

        # ── Heat score: warm up retrieved memories ────────────────────────────
        if injection:
            fp = self._fingerprint(injection[:200])
            with self._heat_lock:
                current = MemoryCell._heat_scores.get(fp, 0.5)
                MemoryCell._heat_scores[fp] = min(current + self._HEAT_ACCESS, 1.0)

        # ── Inject HOT memories at top (always inject regardless of similarity) ─
        hot_injections = self._get_hot_memories(query)
        if hot_injections:
            return "[HOT MEMORIES]\n" + hot_injections + "\n\n" + injection
        return injection

    def _get_hot_memories(self, query: str) -> str:
        """Fetch HOT (heat > 0.7) memories — always injected at the top."""
        # Identify hot fingerprints
        with self._heat_lock:
            hot_fps = [fp for fp, heat in MemoryCell._heat_scores.items()
                       if heat >= self._HOT_THRESHOLD]

        if not hot_fps:
            return ""

        # We don't have a "fetch by fingerprint" endpoint, so we re-query
        # ChromaDB with a slightly modified query for HOT context.
        # This is a best-effort approach — the heat is mainly used for reranking.
        try:
            r = _req.post(
                f"{MEMORY_URL}/inject",
                json={"query": f"important recurring context: {query}", "top_k": 3,
                      "threshold": 0.2},
                timeout=1.5,
            )
            return r.json().get("injection", "")
        except Exception:
            return ""

    def _heat_set_from_surprise(self, fingerprint: str, surprise: float) -> None:
        """Set initial heat score for a new memory based on surprise."""
        with self._heat_lock:
            MemoryCell._heat_scores[fingerprint] = float(surprise)

    def get_heat_tier(self, text: str) -> str:
        """Return heat tier (HOT/WARM/COLD) for a piece of text."""
        fp = self._fingerprint(text)
        with self._heat_lock:
            heat = MemoryCell._heat_scores.get(fp, 0.5)
        if heat >= self._HOT_THRESHOLD:
            return "HOT"
        if heat >= self._WARM_THRESHOLD:
            return "WARM"
        return "COLD"

    def save(self, user_input: str, eve_response: str, session_id: str) -> None:
        """
        Persist an exchange to ChromaDB + Mem0AI with surprise gate + HEMA update.

        Surprise Gate: exchanges below _SURPRISE_THRESHOLD are semantically
        redundant — skip full vector storage to reduce noise.
        HEMA update: always update vocab + recent list; periodically refresh summary.
        """
        # ── Surprise gate ─────────────────────────────────────────────────────
        surprise = self._surprise_score(user_input, eve_response)
        self._update_vocab(user_input + " " + eve_response)

        # Track recent exchange fingerprints
        fp = self._fingerprint(user_input)
        self._hema_recent.append({
            "user": user_input[:300],
            "eve":  eve_response[:300],
            "fp":   fp,
            "surprise": surprise,
        })
        if len(self._hema_recent) > 30:
            self._hema_recent = self._hema_recent[-30:]

        self._hema_turn_count += 1

        # Schedule HEMA summary refresh every N turns (background, non-blocking)
        if self._hema_turn_count % _HEMA_SUMMARY_TURNS == 0:
            asyncio.create_task(self._refresh_hema_summary(self._hema_recent))

        # ── Titans neural memory update (ALWAYS — Titans has its own internal gate) ─
        if self._manager:
            _titans = self._manager._cells.get("titans")
            if _titans:
                try:
                    _titans.learn(user_input, eve_response)
                except Exception:
                    pass

        if surprise < _SURPRISE_THRESHOLD:
            # Exchange is not novel enough for vector databases — skip ChromaDB/Mem0AI
            self._skipped_count += 1
            logger.debug(
                "[MemoryCell] Surprise gate: SKIP (surprise=%.2f < %.2f)",
                surprise, _SURPRISE_THRESHOLD,
            )
            return

        self._saved_count += 1
        logger.debug("[MemoryCell] Surprise gate: SAVE (surprise=%.2f)", surprise)

        # Assign initial heat score from surprise value
        self._heat_set_from_surprise(fp, surprise)

        # Layer 1: raw ChromaDB
        try:
            _req.post(
                f"{MEMORY_URL}/save",
                json={
                    "user_input":  user_input,
                    "eve_response": eve_response,
                    "session_id":  session_id,
                },
                timeout=3.0,
            )
        except Exception:
            pass
        # Layer 2: Mem0AI consolidation (background, non-blocking)
        try:
            m = _init_mem0()
            if m and m is not False:
                messages = [
                    {"role": "user",      "content": user_input},
                    {"role": "assistant", "content": eve_response},
                ]
                m.add(messages, user_id="forge", metadata={"session": session_id})
        except Exception as e:
            logger.debug("[MemoryCell] Mem0AI save error: %s", e)

    def mem0_search(self, query: str, top_k: int = 5) -> str:
        """
        Mem0AI consolidated memory search — returns intelligently compressed facts.
        Falls back to empty string gracefully.
        """
        try:
            m = _init_mem0()
            if not m or m is False:
                return ""
            results = m.search(query, user_id="forge", limit=top_k)
            if not results or not results.get("results"):
                return ""
            facts = [r.get("memory", "") for r in results["results"] if r.get("memory")]
            if not facts:
                return ""
            return "[Consolidated Memory]\n" + "\n".join(f"• {f}" for f in facts)
        except Exception as e:
            logger.debug("[MemoryCell] Mem0AI search error: %s", e)
            return ""

    def health(self) -> dict:
        svc_ok = False
        try:
            r = _req.get(f"{MEMORY_URL}/health", timeout=1)
            svc_ok = r.status_code == 200
        except Exception:
            pass
        total = max(self._saved_count + self._skipped_count, 1)
        # Heat stats
        with self._heat_lock:
            all_heats = list(MemoryCell._heat_scores.values())
        hot_count  = sum(1 for h in all_heats if h >= self._HOT_THRESHOLD)
        warm_count = sum(1 for h in all_heats if self._WARM_THRESHOLD <= h < self._HOT_THRESHOLD)
        cold_count = sum(1 for h in all_heats if h < self._WARM_THRESHOLD)
        return {
            "memory_service":     svc_ok,
            "hema_summary_len":   len(self._hema_summary.split()),
            "hema_turns":         self._hema_turn_count,
            "surprise_saved":     self._saved_count,
            "surprise_skipped":   self._skipped_count,
            "surprise_save_rate": round(self._saved_count / total, 3),
            "heat_hot":           hot_count,
            "heat_warm":          warm_count,
            "heat_cold":          cold_count,
            "heat_total":         len(all_heats),
        }
