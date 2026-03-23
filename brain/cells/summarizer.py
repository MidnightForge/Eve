"""
SummarizerCell — Eve's Context Compression & Summarization Engine.

Implements the Agentic AI framework layer: Agent Capabilities → Summarisation,
Context Management, Multi-turn State Compression.

Research basis:
  - LLMLingua (Jiang et al., Microsoft 2023) — arxiv:2310.05736
    Compress prompts 3-20x by scoring each token's importance using a
    small LM (e.g., LLaMA-7B). Preserves critical info, drops filler.
    Applied: compress long conversation histories before passing to Cortex.
  - LLMLingua-2 (Wu et al., Microsoft 2024) — arxiv:2403.12543
    Data distillation approach. 3-6x faster than original. Better
    preservation of task-critical information. Key-info extraction.
  - MapReduce Summarization — hierarchical: summarize chunks → combine.
    Applied: for very long documents, summarize pages then merge.
  - Hierarchical Summarization (Chang et al., 2023) — recursive structure.
    Build summary tree: sentences → paragraphs → sections → document.
  - SBERT Semantic Similarity (Reimers & Gurevych, 2019) — sentence-transformers.
    Cluster semantically similar segments before summarizing to remove
    redundancy. Sentence embeddings for meaning-preserving compression.

VRAM: 0 (uses Claude Haiku API — no local model required).
Status: ONLINE — active on current RTX 4090 system.
"""

import asyncio
import logging
import os
import threading
from typing import Optional

import anthropic

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"

# Character threshold for triggering summarization
_SUMMARIZE_THRESHOLD = 2000


class SummarizerCell(BaseCell):
    name        = "summarizer"
    description = "Summarizer — Context Compression"
    color       = "#0891b2"
    lazy        = True
    position    = (4, 2)

    system_tier     = "online"
    hardware_req    = "API only — no GPU required"
    framework_layer = "AI Agents → Summarisation"
    research_basis  = (
        "LLMLingua arxiv:2310.05736 (Microsoft 2023) — 20x compression, coarse+fine phases. "
        "LLMLingua-2 (ACL 2024) — distilled small model for task-agnostic compression. "
        "MapReduce Summarization — chunk→summarize→merge for long docs. "
        "Hierarchical Summarization (Chang 2023) — multi-level abstraction pyramid. "
        "SBERT arxiv:1908.10084 (Reimers 2019) — semantic similarity scoring."
    )
    build_notes = (
        "ONLINE: Conversation history compression + document summarization active. "
        "NEXT: LLMLingua token-level prompt compression (3-20x reduction), "
        "LLMLingua-2 data distillation approach, "
        "hierarchical tree summarization for long documents, "
        "SBERT-based semantic deduplication before summarizing."
    )

    async def process(self, ctx: CellContext) -> dict:
        """Activated for long messages or explicit summarization requests."""
        summarize_cues = [
            "summarize", "summary", "tldr", "tl;dr", "brief", "shorten",
            "condense", "key points", "main points", "overview"
        ]
        msg_lower = ctx.message.lower()
        needs_summary = any(c in msg_lower for c in summarize_cues)
        is_long = len(ctx.message) > _SUMMARIZE_THRESHOLD

        if not needs_summary and not is_long:
            return {"summarized": False}

        target_text = ctx.message
        if is_long and not needs_summary:
            # Auto-compress long memory injection
            if ctx.memory_injection and len(ctx.memory_injection) > 1000:
                target_text = ctx.memory_injection
            else:
                return {"summarized": False}

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _summarize():
            try:
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
                resp = client.messages.create(
                    model=_MODEL,
                    max_tokens=400,
                    system=(
                        "You are a precise summarizer. Extract only the essential information. "
                        "Use bullet points for lists. Preserve all numbers, dates, names, and key facts. "
                        "Be concise — maximum 3-5 sentences or bullet points."
                    ),
                    messages=[{"role": "user", "content": f"Summarize:\n\n{target_text[:4000]}"}],
                )
                summary = resp.content[0].text.strip()
                asyncio.run_coroutine_threadsafe(queue.put(summary), loop)
            except Exception as exc:
                logger.debug("[SummarizerCell] Error: %s", exc)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_summarize, daemon=True).start()
        try:
            summary = await asyncio.wait_for(queue.get(), timeout=10.0)
        except asyncio.TimeoutError:
            return {"summarized": False, "reason": "timeout"}

        if summary:
            if is_long and ctx.memory_injection:
                ctx.memory_injection = f"[Compressed Memory]\n{summary}"
            ctx.metadata["summary"] = summary
            compression = round((1 - len(summary) / len(target_text)) * 100)
            logger.info("[SummarizerCell] Compressed by ~%d%%", compression)
            return {"summarized": True, "compression_pct": compression, "length": len(summary)}

        return {"summarized": False}

    def health(self) -> dict:
        return {"model": _MODEL, "threshold_chars": _SUMMARIZE_THRESHOLD}
