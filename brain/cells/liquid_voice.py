"""
LiquidVoiceCell — Liquid Time-Constant adaptive voice parameters.

LTC concept: neurons with time constants that adapt to input, unlike static RNNs.
Here: voice parameters (speed, pitch, energy) don't snap instantly — they decay
toward the target with time constant τ=0.3, producing smooth transitions.

Emotion → voice parameter mapping:
  happy:   speed=1.05, pitch=+1, energy=1.1
  sad:     speed=0.90, pitch=-1, energy=0.8
  excited: speed=1.10, pitch=+2, energy=1.2
  calm:    speed=0.95, pitch=0,  energy=0.9
  angry:   speed=1.00, pitch=-2, energy=1.3
  neutral: speed=1.00, pitch=0,  energy=1.0

LTC update rule:
  param(t+1) = param(t) + τ * (target - param(t))

REST endpoint:
  GET /brain/liquid_voice/state — current voice parameters + target + emotion
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

_TAU = 0.3   # time constant — higher = faster transitions

_EMOTION_TARGETS = {
    "happy":   {"speed": 1.05, "pitch": 1.0,  "energy": 1.1},
    "sad":     {"speed": 0.90, "pitch": -1.0, "energy": 0.8},
    "excited": {"speed": 1.10, "pitch": 2.0,  "energy": 1.2},
    "calm":    {"speed": 0.95, "pitch": 0.0,  "energy": 0.9},
    "angry":   {"speed": 1.00, "pitch": -2.0, "energy": 1.3},
    "neutral": {"speed": 1.00, "pitch": 0.0,  "energy": 1.0},
    "focused": {"speed": 0.97, "pitch": -0.5, "energy": 0.95},
    "playful": {"speed": 1.07, "pitch": 1.5,  "energy": 1.15},
    "serious": {"speed": 0.92, "pitch": -1.0, "energy": 0.88},
    "surprised":{"speed": 1.05,"pitch": 1.5,  "energy": 1.1},
}

_DEFAULT_TARGET = _EMOTION_TARGETS["neutral"]


class LiquidVoiceCell(BaseCell):
    """
    Liquid Time-Constant voice parameter smoother.

    Reference: Hasani et al. (2021) "Liquid Time-constant Networks" AAAI 2021.
    LTC neurons: dx/dt = -x/τ(x,t) + f(x,I)
    Here simplified to discrete LTC update: x_{t+1} = x_t + τ*(target - x_t)

    Reads EmotionCell state, smoothly interpolates voice parameters, and
    feeds them to VoiceCell when generating speech.
    """

    name        = "liquid_voice"
    description = (
        "Liquid Time-Constant adaptive voice. Reads emotion state, "
        "smoothly interpolates speed/pitch/energy toward target with τ=0.3. "
        "No abrupt voice changes — parameters flow like water."
    )
    color       = "#0891b2"
    lazy        = False   # always tracking, minimal cost
    position    = (3, 7)

    system_tier     = "online"
    hardware_req    = "CPU only (parameter math)"
    research_basis  = (
        "Hasani et al. (2021) 'Liquid Time-constant Networks' AAAI 2021. "
        "Neural ODEs (Chen et al. 2018). Smooth parameter interpolation for TTS."
    )
    build_notes     = (
        "LIVE: always-on parameter tracker. Reads EmotionCell each turn. "
        "τ=0.3 LTC update: param += 0.3*(target-param) each call. "
        "GET /brain/liquid_voice/state"
    )
    framework_layer = "AI & ML"

    def __init__(self):
        super().__init__()
        self._current = {"speed": 1.0, "pitch": 0.0, "energy": 1.0}
        self._target  = dict(_DEFAULT_TARGET)
        self._current_emotion = "neutral"
        self._lock    = threading.Lock()
        self._update_count = 0

    async def boot(self) -> None:
        logger.info("[LiquidVoice] Cell online — LTC parameter smoother active")

    def _ltc_step(self) -> None:
        """Apply one LTC update step: x += τ*(target - x)."""
        with self._lock:
            for param in ("speed", "pitch", "energy"):
                curr = self._current[param]
                tgt  = self._target[param]
                self._current[param] = round(curr + _TAU * (tgt - curr), 4)
            self._update_count += 1

    def update_emotion(self, emotion: str) -> dict:
        """Update target parameters based on new emotion state."""
        emotion_key = emotion.lower().split()[0] if emotion else "neutral"
        target = _EMOTION_TARGETS.get(emotion_key, _DEFAULT_TARGET)

        with self._lock:
            self._target         = dict(target)
            self._current_emotion = emotion_key

        self._ltc_step()
        return self.get_state()

    def get_state(self) -> dict:
        """Return current voice parameters + target + emotion."""
        with self._lock:
            return {
                "current":   dict(self._current),
                "target":    dict(self._target),
                "emotion":   self._current_emotion,
                "tau":       _TAU,
                "update_count": self._update_count,
            }

    def get_tts_params(self) -> dict:
        """Get current voice params formatted for TTS calls."""
        with self._lock:
            return {
                "speed":  self._current["speed"],
                "pitch":  self._current["pitch"],
                "energy": self._current["energy"],
            }

    async def process(self, ctx: CellContext) -> Any:
        """Update emotion from context and return current voice params."""
        emotion = ctx.emotion_state or "neutral"
        state = self.update_emotion(emotion)

        # Inject voice params hint into metadata
        ctx.metadata["liquid_voice"] = self.get_tts_params()

        return state

    def health(self) -> dict:
        with self._lock:
            return {
                "status":       self._status.value,
                "emotion":      self._current_emotion,
                "speed":        self._current["speed"],
                "pitch":        self._current["pitch"],
                "energy":       self._current["energy"],
                "update_count": self._update_count,
            }
