"""
VoiceCell — Whisper STT + Kokoro TTS + AudioEngine enhancement.

Handles all audio I/O. Now integrates the native AudioEngine for:
- Post-TTS normalization (target -14 dBFS broadcast standard)
- Emotion-aware pitch shifting (happy = +1 semitone, serious = -0.5)
- Audio analysis of incoming voice (BPM-equivalent energy, tone)
"""

import logging
import tempfile
from pathlib import Path
import requests as _req

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

VOICE_URL = "http://127.0.0.1:8766"

# Emotion → pitch shift mapping (semitones)
_EMOTION_PITCH = {
    "happy":     +1.0,
    "excited":   +1.5,
    "playful":   +0.8,
    "serious":   -0.5,
    "sad":       -1.0,
    "focused":    0.0,
    "neutral":    0.0,
    "surprised": +0.5,
}


class VoiceCell(BaseCell):
    name        = "voice"
    description = "Voice Sidecar — Whisper STT + Kokoro TTS + AudioEngine mastering"
    color       = "#16a34a"
    lazy        = True
    position    = (3, 2)

    def __init__(self):
        super().__init__()
        self._audio_engine = None

    async def boot(self) -> None:
        try:
            r = _req.get(f"{VOICE_URL}/health", timeout=2)
            if r.status_code == 200:
                logger.info("[VoiceCell] Voice sidecar online")
            else:
                self._status = CellStatus.DEGRADED
        except Exception as exc:
            self._status = CellStatus.DEGRADED
            logger.warning("[VoiceCell] Voice sidecar unreachable: %s", exc)
        # Attach AudioEngine
        try:
            from eve_creative_engine import get_engine
            self._audio_engine = get_engine().audio
            logger.info("[VoiceCell] AudioEngine attached — normalization + pitch active")
        except Exception as exc:
            logger.warning("[VoiceCell] AudioEngine unavailable: %s", exc)

    async def process(self, ctx: CellContext) -> dict:
        return {
            "voice_mode":       ctx.voice_mode,
            "sidecar_url":      VOICE_URL,
            "audio_engine":     self._audio_engine is not None,
        }

    def enhance_tts_audio(self, audio_path: str, emotion: str = "neutral") -> str:
        """
        Post-process a TTS output file:
        1. Normalize to -14 dBFS (broadcast standard)
        2. Apply emotion-based pitch shift
        Returns path to enhanced file (or original on failure).
        """
        if not self._audio_engine:
            return audio_path
        try:
            p = Path(audio_path)
            normalized = str(p.with_stem(p.stem + "_norm"))
            result = self._audio_engine.normalize(audio_path, normalized, target_dbfs=-14.0)
            if result.get("status") != "ok":
                return audio_path

            semitones = _EMOTION_PITCH.get(emotion, 0.0)
            if abs(semitones) > 0.01:
                pitched = str(p.with_stem(p.stem + "_pitch"))
                result2 = self._audio_engine.pitch_shift(normalized, pitched, semitones)
                if result2.get("status") == "ok":
                    return pitched
            return normalized
        except Exception as e:
            logger.debug("[VoiceCell] Audio enhancement failed: %s", e)
            return audio_path

    def analyze_input_audio(self, audio_path: str) -> dict:
        """
        Analyze incoming voice audio (e.g. from Whisper input).
        Returns energy level, estimated pitch, duration.
        Used to detect emotional tone from voice before transcription.
        """
        if not self._audio_engine:
            return {}
        try:
            return self._audio_engine.analyze(audio_path)
        except Exception:
            return {}

    def health(self) -> dict:
        try:
            r = _req.get(f"{VOICE_URL}/health", timeout=1)
            return {
                "sidecar_online": r.status_code == 200,
                "audio_engine":   self._audio_engine is not None,
            }
        except Exception:
            return {"sidecar_online": False, "audio_engine": self._audio_engine is not None}
