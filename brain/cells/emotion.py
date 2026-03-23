"""
EmotionCell — Eve's emotional intelligence engine.

Detects the emotional tone of incoming messages and maintains
Eve's current emotional state. Drives ANIMA avatar expressions
through the inter-cell bus.

Emotional states map directly to the avatar expression system
already defined in index.html (neutral, happy, thinking, surprised,
focused, speaking, yawn).
"""

import re
import logging
import asyncio
from typing import Optional

from openai import OpenAI

from brain.base_cell import BaseCell, CellContext, CellStatus

_VLLM_URL = "http://127.0.0.1:8099/v1"
_MODEL    = "eve"

logger = logging.getLogger(__name__)

# Quick keyword map for zero-latency baseline detection
_KEYWORD_EMOTIONS = {
    "happy":     ["love", "amazing", "great", "awesome", "yes!", "🥰", "❤️", "haha", "lol", "yay"],
    "thinking":  ["hmm", "why", "how does", "explain", "what if", "curious", "wonder"],
    "surprised": ["wait", "what?", "really?", "seriously", "omg", "no way", "holy"],
    "focused":   ["code", "build", "write", "analyze", "debug", "fix", "implement"],
    "yawn":      ["tired", "sleepy", "boring", "yawn", "bed", "night"],
}

_VALID_EMOTIONS = {"neutral", "happy", "thinking", "surprised", "focused", "speaking", "yawn"}


class EmotionCell(BaseCell):
    name        = "emotion"
    description = "Emotion Engine — mood detection & avatar expression"
    color       = "#ec4899"
    lazy        = False
    position    = (0, 0)

    def __init__(self):
        super().__init__()
        self._current_emotion = "neutral"
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=_VLLM_URL, api_key="none")
        return self._client

    async def process(self, ctx: CellContext) -> str:
        """Detect emotion from message. Updates ctx.emotion_state."""
        emotion = self._keyword_detect(ctx.message)
        if not emotion:
            emotion = await self._ai_detect(ctx.message)
        self._current_emotion = emotion
        ctx.emotion_state = emotion
        return emotion

    def _keyword_detect(self, text: str) -> Optional[str]:
        lower = text.lower()
        for emotion, keywords in _KEYWORD_EMOTIONS.items():
            if any(kw in lower for kw in keywords):
                return emotion
        return None

    async def _ai_detect(self, text: str) -> str:
        """Fast haiku inference for nuanced emotion detection."""
        try:
            loop = asyncio.get_event_loop()
            def _call():
                r = self._get_client().chat.completions.create(
                    model=_MODEL,
                    max_tokens=10,
                    messages=[
                        {"role": "system", "content": (
                            f"Classify the emotional tone of this message as ONE word from: "
                            f"{', '.join(_VALID_EMOTIONS)}. Reply with ONLY that word."
                        )},
                        {"role": "user", "content": text[:200]},
                    ],
                )
                return r.choices[0].message.content.strip().lower()
            result = await asyncio.wait_for(loop.run_in_executor(None, _call), timeout=2.0)
            return result if result in _VALID_EMOTIONS else "neutral"
        except Exception:
            return "neutral"

    @property
    def current_emotion(self) -> str:
        return self._current_emotion

    def health(self) -> dict:
        return {"current_emotion": self._current_emotion}
