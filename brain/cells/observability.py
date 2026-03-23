"""
ObservabilityCell — Eve's Tracing, Metrics & Feedback Loop Engine.

Implements the Agentic AI framework layer: Observability & Tracing,
Feedback Loops & Evaluations, Cost & Resource Management.

Research basis:
  - OpenTelemetry (CNCF 2023) — distributed tracing standard. Spans, traces,
    metrics. Applied to LLM systems for token cost tracking, latency,
    error rates. Eve's brain uses this pattern.
  - Arize Phoenix / LangSmith patterns — LLM-specific observability.
    Track: prompt quality, retrieval relevance, hallucination rate,
    user satisfaction signals, token cost per turn.
  - RLHF feedback loop patterns — convert implicit signals (retry rate,
    session length, rephrasing) into preference data for fine-tuning.
  - LLM-as-Judge (Zheng et al., 2023) — arxiv:2306.05685
    Use a strong LLM to automatically evaluate response quality.
    Creates scalable automated feedback without human annotation.
  - HELM (Liang et al., Stanford 2022) — Holistic Evaluation of LMs.
    Multi-dimensional scoring: accuracy, calibration, robustness, fairness,
    efficiency, bias, toxicity. Eve evaluates herself against these.

VRAM: 0 (pure metrics collection — no local model required).
Status: ONLINE — active on current RTX 4090 system.
"""

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)


class ObservabilityCell(BaseCell):
    name        = "observability"
    description = "Observability — Tracing, Metrics & Feedback"
    color       = "#d97706"
    lazy        = False   # always-on — collects data from first request
    position    = (4, 1)

    system_tier     = "online"
    hardware_req    = "CPU only — no GPU required"
    framework_layer = "Agentic AI → Governance"
    research_basis  = (
        "OpenTelemetry (CNCF 2023) — spans/metrics/traces standard. "
        "LLM-as-Judge arxiv:2306.05685 (Zheng 2023) — MT-Bench auto eval. "
        "HELM arxiv:2211.09110 (Stanford 2022) — holistic model evaluation. "
        "RLHF feedback loops — human preference as reward signal. "
        "Arize Phoenix — LLM observability & drift detection patterns."
    )
    build_notes = (
        "ONLINE: Turn metrics, latency tracking, error rates, cell call counts active. "
        "NEXT: Full OpenTelemetry span tracing, LLM-as-Judge auto-scoring per turn, "
        "RLHF signal extraction from user behavior, cost-per-turn tracking, "
        "Grafana-compatible metrics export, anomaly detection alerts."
    )

    def __init__(self):
        super().__init__()
        self._turns:          list  = []       # per-turn records
        self._cell_latencies: dict  = defaultdict(list)
        self._error_log:      deque = deque(maxlen=200)
        self._cost_tracker:   dict  = defaultdict(float)  # model → estimated $
        self._session_start:  float = time.time()
        self._quality_scores: deque = deque(maxlen=100)

        # Token cost estimates (per 1M tokens, USD) — Claude Haiku
        self._COSTS = {
            "claude-haiku-4-5-20251001":  {"input": 0.25, "output": 1.25},
            "claude-sonnet-4-6":          {"input": 3.00, "output": 15.00},
        }

    async def process(self, ctx: CellContext) -> dict:
        """Record this turn's metrics."""
        turn_record = {
            "timestamp":    time.time(),
            "message_len":  len(ctx.message),
            "is_complex":   ctx.is_complex,
            "voice_mode":   ctx.voice_mode,
            "active_cells": list(ctx.active_cells),
            "user_id":      ctx.user_id,
        }
        self._turns.append(turn_record)

        # Keep only last 500 turns in memory
        if len(self._turns) > 500:
            self._turns = self._turns[-500:]

        return {"recorded": True, "total_turns": len(self._turns)}

    def record_cell_latency(self, cell_name: str, ms: float) -> None:
        self._cell_latencies[cell_name].append(ms)
        if len(self._cell_latencies[cell_name]) > 200:
            self._cell_latencies[cell_name] = self._cell_latencies[cell_name][-200:]

    def record_error(self, cell_name: str, error: str) -> None:
        self._error_log.append({
            "cell": cell_name, "error": error[:200], "time": time.time()
        })

    def record_quality(self, score: float) -> None:
        """Score 0-10 from LLM-as-Judge or user signal."""
        self._quality_scores.append({"score": score, "time": time.time()})

    def get_dashboard(self) -> dict:
        """Full metrics snapshot."""
        uptime = time.time() - self._session_start
        avg_quality = (
            sum(q["score"] for q in self._quality_scores) / len(self._quality_scores)
            if self._quality_scores else None
        )
        cell_avg_ms = {
            cell: round(sum(lats) / len(lats), 1)
            for cell, lats in self._cell_latencies.items() if lats
        }
        recent_errors = list(self._error_log)[-10:]
        return {
            "uptime_s":      round(uptime, 0),
            "total_turns":   len(self._turns),
            "turns_per_hr":  round(len(self._turns) / max(uptime / 3600, 0.001), 1),
            "avg_quality":   round(avg_quality, 2) if avg_quality else "N/A",
            "cell_latencies_ms": cell_avg_ms,
            "recent_errors": recent_errors,
            "total_errors":  len(self._error_log),
        }

    def health(self) -> dict:
        dash = self.get_dashboard()
        return {
            "total_turns":  dash["total_turns"],
            "uptime_s":     dash["uptime_s"],
            "total_errors": dash["total_errors"],
            "avg_quality":  dash["avg_quality"],
        }
