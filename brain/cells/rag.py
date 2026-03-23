"""
RAGCell — Eve's Retrieval-Augmented Generation Engine.

Implements the Agentic AI framework layer: Gen AI → RAG,
Multi-cognition, Transformation Mitigation.

Research basis:
  - RAG (Lewis et al., Meta 2020) — arxiv:2005.11401
    Original RAG: retrieves documents, feeds to generator. Foundation.
  - HyDE: Hypothetical Document Embeddings (Gao et al., 2022) — arxiv:2212.10496
    Generate a FAKE answer first, embed it, use embedding to retrieve
    real documents. Closes vocabulary gap between queries and documents.
  - Corrective RAG (Yan et al., 2024) — arxiv:2401.15884
    Evaluates retrieved documents for relevance. If irrelevant → web search.
    If ambiguous → knowledge refinement. Dramatically improves RAG accuracy.
  - Self-RAG (Asai et al., 2023) — arxiv:2310.11511
    Adaptive retrieval: model decides WHEN to retrieve (not always).
    Uses special tokens: [Retrieve], [IsRel], [IsSup], [IsUse].
  - GraphRAG (Edge et al., Microsoft 2024) — arxiv:2404.16130
    Builds knowledge graph from documents. Community detection on graph.
    Enables global questions that standard RAG cannot answer.
  - LLMLingua (Jiang et al., Microsoft 2023) — arxiv:2310.05736
    Compresses prompts by 3-20x using a small model to identify
    unimportant tokens. Reduces cost and latency with minimal quality loss.

VRAM: 0 (uses existing ChromaDB + Claude API).
Status: ONLINE — active on current RTX 4090 system.
"""

import asyncio
import logging
import os
import threading
from typing import Optional

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

# ChromaDB connection (reuses Eve's memory service)
_CHROMA_HOST = "http://127.0.0.1:8767"
_COLLECTION  = "eve_memory"


class RAGCell(BaseCell):
    name        = "rag"
    description = "RAG — Retrieval-Augmented Generation"
    color       = "#059669"
    lazy        = True
    position    = (5, 0)

    system_tier     = "online"
    hardware_req    = "ChromaDB + API only — no GPU required"
    framework_layer = "Gen AI → RAG"
    research_basis  = (
        "RAG arxiv:2005.11401 (Lewis/Meta 2020) — foundation. "
        "HyDE arxiv:2212.10496 (Gao 2022) — fake-doc embedding closes vocab gap. "
        "Corrective RAG arxiv:2401.15884 (Yan 2024) — bad retrieval → web fallback. "
        "Self-RAG arxiv:2310.11511 (Asai 2023) — [Retrieve]/[IsRel] reflection tokens. "
        "GraphRAG arxiv:2404.16130 (Microsoft 2024) — KG community detection. "
        "LLMLingua arxiv:2310.05736 (Microsoft 2023) — 20x prompt compression."
    )
    build_notes = (
        "ONLINE: Standard RAG retrieval from ChromaDB active. "
        "NEXT: HyDE hypothetical doc embedding, Corrective RAG with web fallback, "
        "Self-RAG adaptive retrieval tokens, GraphRAG knowledge graph construction, "
        "LLMLingua context compression for long retrievals."
    )

    def __init__(self):
        super().__init__()
        self._chroma_client = None
        self._collection = None

    async def boot(self) -> None:
        """Connect to ChromaDB."""
        try:
            import chromadb
            self._chroma_client = chromadb.HttpClient(host="127.0.0.1", port=8767)
            collections = self._chroma_client.list_collections()
            collection_names = [c.name for c in collections]
            # Use eve_memory if available, otherwise first collection
            target = "eve_memory" if "eve_memory" in collection_names else (collection_names[0] if collection_names else None)
            if target:
                self._collection = self._chroma_client.get_collection(target)
                logger.info("[RAGCell] Connected to ChromaDB collection: %s", target)
            else:
                logger.warning("[RAGCell] No ChromaDB collections found")
        except Exception as exc:
            logger.warning("[RAGCell] ChromaDB connection failed: %s", exc)

    async def process(self, ctx: CellContext) -> dict:
        if not self._collection:
            return {"retrieved": False, "reason": "ChromaDB not connected"}

        # Self-RAG style: decide if retrieval is needed
        retrieval_cues = [
            "remember", "recall", "did you", "have i", "what did", "history",
            "last time", "before", "earlier", "previously", "told you", "said",
            "know about", "find", "search", "look up", "what is", "who is"
        ]
        msg_lower = ctx.message.lower()
        should_retrieve = any(cue in msg_lower for cue in retrieval_cues)

        if not should_retrieve:
            return {"retrieved": False, "reason": "no retrieval cue detected"}

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _retrieve():
            try:
                results = self._collection.query(
                    query_texts=[ctx.message],
                    n_results=5,
                    include=["documents", "distances", "metadatas"],
                )
                docs      = results.get("documents", [[]])[0]
                distances = results.get("distances", [[]])[0]
                metas     = results.get("metadatas", [[]])[0]

                # Corrective RAG: filter by relevance (distance < 1.5 = relevant)
                relevant = [
                    {"doc": d, "distance": dist, "meta": m}
                    for d, dist, m in zip(docs, distances, metas)
                    if dist < 1.5
                ]

                if relevant:
                    context_text = "\n---\n".join(r["doc"] for r in relevant[:3])
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"retrieved": True, "context": context_text, "count": len(relevant)}),
                        loop
                    )
                else:
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"retrieved": False, "reason": "no relevant documents found"}),
                        loop
                    )
            except Exception as exc:
                logger.debug("[RAGCell] Retrieval error: %s", exc)
                asyncio.run_coroutine_threadsafe(
                    queue.put({"retrieved": False, "reason": str(exc)}),
                    loop
                )

        threading.Thread(target=_retrieve, daemon=True).start()
        try:
            result = await asyncio.wait_for(queue.get(), timeout=4.0)
        except asyncio.TimeoutError:
            result = {"retrieved": False, "reason": "timeout"}

        if result.get("retrieved") and result.get("context"):
            # Inject into memory_injection so Cortex uses it
            ctx.memory_injection += f"\n[RAG Retrieved Memory]\n{result['context']}"
            logger.info("[RAGCell] Retrieved %d documents for context", result.get("count", 0))

        return result

    def health(self) -> dict:
        connected = self._collection is not None
        count = 0
        if connected:
            try:
                count = self._collection.count()
            except Exception:
                pass
        return {"chroma_connected": connected, "document_count": count}
