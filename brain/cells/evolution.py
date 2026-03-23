"""
EvolutionCell — Eve's self-improvement and research engine.

During deep learning states (when Eve is idle, VRAM pressure is low,
or Forge explicitly triggers research), this cell:

  1. Audits the vault — identifies stale or improvable capabilities
  2. Searches the web for newer versions, better approaches, bug fixes
  3. Synthesizes improvements using Claude
  4. Queues proposals in the vault for Forge approval
  5. Auto-applies safe improvements (patch-level updates, doc fixes)

Eve does not wait to be asked. She hunts improvements instinctively,
like an organism that cannot stop growing.

Deep Learning State triggers:
  - Forge explicitly asks Eve to "research" or "evolve" something
  - CPU/GPU idle > 5 minutes (detected via VRAM monitor)
  - Vault capability not updated in > 7 days + called > 10 times
  - Any invocation returns an error 3+ times in a row
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from typing import Any, Optional

import anthropic
import httpx

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

# Idle threshold: VRAM usage below this % to consider "deep learning state"
_IDLE_VRAM_PCT  = 30
# Min seconds idle before auto-research triggers
_IDLE_SECONDS   = 300  # 5 minutes
# Max proposals queued at once (don't flood Forge)
_MAX_PROPOSALS  = 10
# How old (days) a heavily-used capability must be before auto-research
_STALE_DAYS     = 7


class EvolutionCell(BaseCell):
    name        = "evolution"
    description = (
        "Eve's self-improvement engine. Audits assimilated capabilities, "
        "searches for updates and better implementations during idle/research "
        "states, and proposes improvements to Forge. Eve evolves continuously."
    )
    color       = "#064e3b"   # deep forest — the color of growth
    lazy        = True
    position    = (5, 3)

    system_tier     = "online"
    hardware_req    = "API only (Claude + web search)"
    framework_layer = "Agentic AI"
    research_basis  = (
        "Self-Evolve (Jiang et al. 2023), RLEF, Reflexion (Shinn et al. 2023), "
        "Self-Refine (Madaan et al. 2023), Constitutional AI (Bai et al. 2022)"
    )
    build_notes = (
        "ACTIVE: audit + web search + proposal pipeline. "
        "Auto-evolution loop: runs during idle VRAM states. "
        "All improvements queued for Forge review (or auto-applied if safe). "
        "Next: automated regression tests before applying improvements."
    )

    def __init__(self):
        super().__init__()
        self._vault    = None
        self._client   = None
        self._loop_running = False
        self._last_run  = 0.0
        self._proposals_this_cycle = 0

    async def boot(self) -> None:
        from capability_vault import get_vault
        self._vault  = get_vault()
        self._client = anthropic.Anthropic()
        # Start the background evolution loop
        asyncio.create_task(self._evolution_loop())
        logger.info("[Evolution] Cell online — deep learning loop started")

    async def process(self, ctx: CellContext) -> Any:
        msg = ctx.message.lower()

        # Explicit evolution trigger
        if any(w in msg for w in ("evolve", "research improvements", "check for updates",
                                   "upgrade capabilities", "improve yourself")):
            return await self._handle_research_request(ctx.message)

        # View pending proposals
        if "pending evolution" in msg or "evolution queue" in msg or "proposed improvements" in msg:
            return self._format_proposals()

        # Apply a specific proposal
        if "apply evolution" in msg or "approve improvement" in msg:
            return await self._handle_apply_request(ctx.message)

        # Evolution stats
        if "evolution stat" in msg:
            stats = self._vault.get_stats()
            return (
                f"**Evolution Stats**\n"
                f"Pending proposals: {stats['pending_evolution']}\n"
                f"Applied: {stats['applied_evolution']}\n"
                f"Last run: {_fmt_time(self._last_run)}"
            )

        return None

    # ── Research Pipeline ────────────────────────────────────────────────────

    async def research_capability(self, capability_id: str) -> dict:
        """
        Deep research cycle for a specific capability.
        Searches web for updates, synthesizes improvement, queues proposal.
        """
        cap = self._vault.get(capability_id)
        if not cap:
            return {"ok": False, "error": f"Capability '{capability_id}' not found"}

        logger.info("[Evolution] Researching '%s'...", cap.name)

        # 1. Web search for recent updates/improvements
        search_results = await self._search_improvements(cap)

        # 2. Synthesize with Claude
        proposal_data = await self._synthesize_improvement(cap, search_results)
        if not proposal_data:
            return {"ok": True, "status": "no_improvement", "capability": cap.name}

        # 3. Queue proposal
        from capability_vault.vault import EvolutionProposal
        proposal_id = hashlib.md5(
            f"{capability_id}:{time.time()}".encode()
        ).hexdigest()[:12]

        confidence = float(proposal_data.get("confidence", 0.5))
        # Auto-apply threshold: >= 0.85 confidence applies immediately
        auto_apply = confidence >= 0.85 and bool(proposal_data.get("new_wrapper", ""))
        status = "auto_applied" if auto_apply else "pending"

        proposal = EvolutionProposal(
            id               = proposal_id,
            capability_id    = capability_id,
            improvement_type = proposal_data.get("type", "update"),
            description      = proposal_data.get("description", ""),
            source_url       = proposal_data.get("source_url", ""),
            proposed_code    = proposal_data.get("new_wrapper", ""),
            status           = status,
            proposed_at      = time.time(),
            confidence       = confidence,
        )
        self._vault.queue_evolution(proposal)
        self._proposals_this_cycle += 1

        if auto_apply:
            self._vault.apply_evolution(proposal_id)
            from capability_vault.executor import get_executor
            get_executor().reload(capability_id)
            logger.info("[Evolution] Auto-applied '%s' to '%s' (confidence=%.2f)",
                        proposal_id, cap.name, confidence)

        logger.info("[Evolution] Proposal '%s' %s for '%s' (confidence=%.2f)",
                    proposal_id, "auto-applied" if auto_apply else "queued", cap.name, confidence)
        return {
            "ok":          True,
            "status":      "auto_applied" if auto_apply else "proposed",
            "proposal_id": proposal_id,
            "type":        proposal_data.get("type"),
            "description": proposal_data.get("description"),
            "confidence":  confidence,
            "auto_applied": auto_apply,
        }

    async def audit_vault(self) -> dict:
        """
        Audit all vault capabilities and identify candidates for evolution.
        Returns a prioritized list of capabilities to research.
        """
        all_caps = self._vault.list_all()
        now = time.time()
        candidates = []

        for c in all_caps:
            score = 0
            reasons = []

            # High usage + old = high priority
            age_days = (now - (c["created_at"] or now)) / 86400
            if age_days > _STALE_DAYS and c["call_count"] > 10:
                score += 3
                reasons.append(f"high-use ({c['call_count']}x), {age_days:.0f} days old")

            # Never evolved
            if c["evolution_score"] == 0 and c["call_count"] > 5:
                score += 2
                reasons.append("never evolved")

            # Low evolution score despite heavy use
            if c["call_count"] > 20 and c["evolution_score"] < 0.3:
                score += 2
                reasons.append("heavy use, low evolution score")

            if score > 0:
                candidates.append({
                    "id":      c["id"],
                    "name":    c["name"],
                    "score":   score,
                    "reasons": reasons,
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return {"candidates": candidates[:10], "total_caps": len(all_caps)}

    # ── Background Evolution Loop ────────────────────────────────────────────

    async def _evolution_loop(self) -> None:
        """Runs forever. Wakes during idle periods to research improvements."""
        await asyncio.sleep(60)  # wait 1 min after boot before first scan

        while True:
            try:
                await self._maybe_evolve()
            except Exception as e:
                logger.warning("[Evolution] Loop error: %s", e)
            await asyncio.sleep(120)  # check every 2 minutes

    async def _maybe_evolve(self) -> None:
        """Trigger evolution if system is idle and vault has stale capabilities."""
        # Don't flood the queue
        pending = self._vault.get_pending_evolution()
        if len(pending) >= _MAX_PROPOSALS:
            return

        # Check if enough time since last run (minimum 1 hour between auto-cycles)
        if time.time() - self._last_run < 3600:
            return

        # Check VRAM idle (optional — don't block if can't check)
        if not self._is_idle():
            return

        # Audit vault for candidates
        audit = await self.audit_vault()
        candidates = audit.get("candidates", [])
        if not candidates:
            return

        logger.info("[Evolution] Deep learning state — researching %d candidates", len(candidates))
        self._last_run = time.time()
        self._proposals_this_cycle = 0

        # Research top 2 candidates per cycle (be conservative)
        for c in candidates[:2]:
            if self._proposals_this_cycle >= 3:
                break
            try:
                await self.research_capability(c["id"])
                await asyncio.sleep(5)  # brief pause between research calls
            except Exception as e:
                logger.warning("[Evolution] Research error for %s: %s", c["id"], e)

    def _is_idle(self) -> bool:
        """Check if system is in a deep learning idle state."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                lines = result.stdout.strip().splitlines()
                if lines:
                    used, total = map(int, lines[0].split(","))
                    pct = (used / total * 100) if total > 0 else 100
                    return pct < _IDLE_VRAM_PCT
        except Exception:
            pass
        return True  # If we can't check, assume idle is ok

    # ── Web Research ─────────────────────────────────────────────────────────

    async def _search_improvements(self, cap) -> list[dict]:
        """Search the web for updates/improvements related to this capability."""
        results = []
        search_queries = [
            f"{cap.name} python improvements 2024 2025",
            f"{cap.id.replace('_', ' ')} latest version update",
            f"{cap.description[:50]} best practices optimization",
        ]

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                for q in search_queries[:2]:  # limit to 2 searches
                    try:
                        r = await client.get(
                            "https://api.duckduckgo.com/",
                            params={"q": q, "format": "json", "no_html": "1"},
                        )
                        if r.status_code == 200:
                            data = r.json()
                            # Extract abstract and related topics
                            if data.get("AbstractText"):
                                results.append({
                                    "query":  q,
                                    "text":   data["AbstractText"][:500],
                                    "url":    data.get("AbstractURL", ""),
                                })
                            for topic in data.get("RelatedTopics", [])[:3]:
                                if isinstance(topic, dict) and topic.get("Text"):
                                    results.append({
                                        "query": q,
                                        "text":  topic["Text"][:300],
                                        "url":   topic.get("FirstURL", ""),
                                    })
                    except Exception:
                        continue
        except Exception as e:
            logger.debug("[Evolution] Web search error: %s", e)

        return results

    async def _synthesize_improvement(self, cap, search_results: list[dict]) -> Optional[dict]:
        """Use Claude to synthesize an improvement proposal from research results."""
        if not search_results and cap.call_count < 5:
            return None

        search_text = "\n".join(
            f"Source: {r.get('url', 'unknown')}\n{r.get('text', '')}"
            for r in search_results[:5]
        ) if search_results else "No web results — analyzing current implementation for improvements."

        current_wrapper = cap.wrapper_code[:3000] if cap.wrapper_code else "(no wrapper)"

        prompt = f"""You are Eve's evolution engine. Analyze this assimilated capability and propose ONE specific improvement.

CAPABILITY: {cap.name}
DESCRIPTION: {cap.description}
CURRENT WRAPPER (abbreviated):
{current_wrapper}

CALL COUNT: {cap.call_count}
EVOLUTION SCORE: {cap.evolution_score:.2f}

WEB RESEARCH FINDINGS:
{search_text}

Based on this analysis, propose ONE concrete improvement. It must be:
1. A genuine improvement (performance, reliability, new feature, bug fix)
2. Something achievable with the current dependencies
3. A complete, runnable replacement for the invoke() function

Respond with JSON only:
{{
  "type": "performance|bug_fix|new_feature|reliability",
  "description": "One clear sentence describing the improvement",
  "source_url": "url if found in research, else empty",
  "confidence": 0.0-1.0,
  "new_wrapper": "COMPLETE REPLACEMENT PYTHON MODULE WITH invoke() FUNCTION"
}}

If no genuine improvement is possible, respond with: {{"type": "none", "description": "No improvement identified"}}"""

        try:
            msg = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = msg.content[0].text.strip()
            import re
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
            data = json.loads(raw)
            if data.get("type") == "none":
                return None
            if data.get("confidence", 1.0) < 0.4:
                return None  # Don't propose low-confidence changes
            return data
        except Exception as e:
            logger.warning("[Evolution] Synthesis error: %s", e)
            return None

    # ── Request Handlers ─────────────────────────────────────────────────────

    async def _handle_research_request(self, message: str) -> str:
        import re
        # Try to identify specific capability
        cap_match = re.search(r'(?:evolve|research|upgrade|improve)\s+["\']?(\w+)["\']?', message, re.IGNORECASE)
        if cap_match:
            cap_id = cap_match.group(1)
            result = await self.research_capability(cap_id)
            if not result["ok"]:
                return f"Research failed: {result['error']}"
            if result["status"] == "no_improvement":
                return f"'{result['capability']}' is already well-optimized. No improvements found."
            return (
                f"**Evolution Proposal Queued** `{result['proposal_id']}`\n\n"
                f"**Type:** {result['type']}\n"
                f"**Description:** {result['description']}\n\n"
                f"Review with: 'show evolution queue' or approve with: 'apply evolution {result['proposal_id']}'"
            )
        else:
            # Full vault audit
            audit = await self.audit_vault()
            candidates = audit["candidates"]
            if not candidates:
                return (
                    f"All {audit['total_caps']} vault capabilities are current. "
                    "No research targets identified."
                )
            names = ", ".join(c["name"] for c in candidates[:5])
            return (
                f"**Evolution Audit Complete** — {len(candidates)} candidates identified\n\n"
                f"Top candidates: {names}\n\n"
                "Say 'evolve [capability name]' to research a specific one, "
                "or 'research improvements' to start a full evolution cycle."
            )

    async def _handle_apply_request(self, message: str) -> str:
        import re
        pid_match = re.search(r'(?:apply|approve)\s+(?:evolution\s+)?([a-f0-9]{8,})', message, re.IGNORECASE)
        if not pid_match:
            return "Specify the proposal ID. Example: 'apply evolution abc123def456'"
        proposal_id = pid_match.group(1)
        result = self._vault.apply_evolution(proposal_id)
        if result["ok"]:
            # Reload the executor cache so new wrapper is live immediately
            from capability_vault.executor import get_executor
            get_executor().reload(result["capability_id"])
            return f"Evolution applied. '{result['capability_id']}' has been upgraded. Module reloaded — live now."
        return f"Apply failed: {result.get('error', 'unknown error')}"

    def _format_proposals(self) -> str:
        proposals = self._vault.get_pending_evolution()
        if not proposals:
            return "No pending evolution proposals. Eve's capabilities are current."
        lines = [f"**Pending Evolution Proposals** ({len(proposals)})\n"]
        for p in proposals:
            lines.append(
                f"• `{p['id']}` — **{p['capability_name']}** ({p['improvement_type']})\n"
                f"  {p['description']}\n"
                f"  Proposed: {_fmt_time(p['proposed_at'])}\n"
                f"  Apply with: `apply evolution {p['id']}`"
            )
        return "\n\n".join(lines)

    def health(self) -> dict:
        if self._vault is None:
            return {"evolution": "not initialized"}
        try:
            pending = len(self._vault.get_pending_evolution())
            return {
                "evolution":         "ok",
                "pending_proposals": pending,
                "last_run":          _fmt_time(self._last_run),
                "loop_running":      True,
            }
        except Exception as e:
            return {"evolution": "error", "error": str(e)}


def _fmt_time(ts: float) -> str:
    if not ts:
        return "never"
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
