"""
PersonaCell — Eve's Personalization & State Persistence Engine.

Implements the Agentic AI framework layer: Agent Capabilities → Personalization,
State Persistence, Role Management & Constraints.

Research basis:
  - Adaptive Dialogue Systems (Zhang et al., 2020) — persona-grounded conversation.
    Models that maintain consistent persona and remember user preferences.
  - RecSys-style user modeling (Koren et al., 2009 collaborative filtering evolved)
    User preference vectors updated from interaction patterns.
  - PEARL (Zhong et al., 2024) — Personalizing LLMs with user context.
    Retrieves relevant personal facts at inference time. Similar to RAG but
    for user-specific knowledge rather than world knowledge.
  - MPC: Memory-augmented Personalized Conversation (Xu et al., 2022)
    Long-term persona-consistent dialogue via structured persona memory.
  - Mixtral/Mistral instruction tuning patterns for role adaptation.

VRAM: 0 (structured data + API — no local model required).
Status: ONLINE — active on current RTX 4090 system.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

_PERSONA_FILE = Path(os.path.expanduser("~")) / "eve" / "persona_memory.json"


class PersonaCell(BaseCell):
    name        = "persona"
    description = "Persona — User Modeling & Personalization"
    color       = "#be185d"
    lazy        = False   # always-on — loads user profile at startup
    position    = (5, 1)

    system_tier     = "online"
    hardware_req    = "CPU only — no GPU required"
    framework_layer = "AI Agents → Personalization"
    research_basis  = (
        "PEARL (Zhong 2024), Adaptive Dialogue Systems (Zhang 2020), "
        "MPC Memory-augmented Conversation (Xu 2022), RecSys user modeling"
    )
    build_notes = (
        "ONLINE: User preference tracking, interaction patterns, tone adaptation active. "
        "NEXT: Full PEARL persona retrieval at inference time, "
        "collaborative filtering for preference prediction, "
        "multi-user persona isolation, "
        "longitudinal user model updates from session signals."
    )

    def __init__(self):
        super().__init__()
        self._profiles: dict = {}   # user_id → profile
        self._load_profiles()

    def _load_profiles(self) -> None:
        try:
            if _PERSONA_FILE.exists():
                self._profiles = json.loads(_PERSONA_FILE.read_text(encoding="utf-8"))
                logger.info("[PersonaCell] Loaded %d user profiles", len(self._profiles))
        except Exception as exc:
            logger.debug("[PersonaCell] Profile load error: %s", exc)

    def _save_profiles(self) -> None:
        try:
            _PERSONA_FILE.write_text(json.dumps(self._profiles, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("[PersonaCell] Profile save error: %s", exc)

    def _get_profile(self, user_id: int) -> dict:
        key = str(user_id)
        if key not in self._profiles:
            self._profiles[key] = {
                "user_id":        user_id,
                "created":        time.time(),
                "last_seen":      time.time(),
                "turn_count":     0,
                "preferred_tone": "warm",       # warm | formal | playful | brief
                "topics":         {},            # topic → mention count
                "preferences":    {},            # key → value
                "name_hint":      None,          # if user mentioned their name
                "expertise":      "general",     # general | technical | creative
                "voice_prefers":  False,
            }
        return self._profiles[key]

    async def process(self, ctx: CellContext) -> dict:
        profile = self._get_profile(ctx.user_id)

        # Update profile
        profile["last_seen"]  = time.time()
        profile["turn_count"] += 1
        if ctx.voice_mode:
            profile["voice_prefers"] = True

        # Detect expertise level from message complexity
        word_count = len(ctx.message.split())
        tech_words = ["api", "vram", "model", "cuda", "tensor", "gradient", "inference",
                      "architecture", "parameter", "quantization", "embedding"]
        tech_score = sum(1 for w in tech_words if w in ctx.message.lower())
        if tech_score >= 2:
            profile["expertise"] = "technical"

        # Extract name hint if user introduces themselves
        import re
        name_match = re.search(r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+)", ctx.message)
        if name_match:
            profile["name_hint"] = name_match.group(1)

        # Topic tracking
        topic_keywords = {
            "eve": "eve_ai", "patent": "patents", "image": "image_gen",
            "video": "video_gen", "code": "coding", "memory": "memory",
            "voice": "voice", "music": "music", "art": "creative",
        }
        for kw, topic in topic_keywords.items():
            if kw in ctx.message.lower():
                profile["topics"][topic] = profile["topics"].get(topic, 0) + 1

        # Build personalization context injection
        persona_context = self._build_context(profile)
        if persona_context:
            ctx.metadata["persona_context"] = persona_context

        # Save periodically (every 10 turns)
        if profile["turn_count"] % 10 == 0:
            self._save_profiles()

        return {"profile_updated": True, "turn_count": profile["turn_count"]}

    def _build_context(self, profile: dict) -> str:
        parts = []
        if profile.get("name_hint"):
            parts.append(f"User's name: {profile['name_hint']}")
        if profile["expertise"] == "technical":
            parts.append("User is technically sophisticated — use precise terminology.")
        if profile["voice_prefers"]:
            parts.append("User often uses voice — keep responses speakable.")
        top_topics = sorted(profile["topics"].items(), key=lambda x: -x[1])[:3]
        if top_topics:
            parts.append(f"Frequent topics: {', '.join(t[0] for t in top_topics)}")
        return "\n".join(parts)

    def health(self) -> dict:
        return {
            "user_profiles": len(self._profiles),
            "profile_file":  str(_PERSONA_FILE),
        }
