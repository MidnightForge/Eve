"""
MusicCell — AI music, sound, and advanced voice synthesis brain cell.

Integrates:
  - MusicGen (Meta) — local text→music, up to 30s, melody conditioning
  - AudioCraft / MAGNeT — parallel music gen from Meta
  - Suno v4 (API) — full song generation with lyrics and style
  - Udio (API) — competing song gen with unique style control
  - CosyVoice 2 — 150ms streaming TTS with emotion parameter control
  - Fish Speech 1.5 — fast local TTS with voice cloning
  - Zonos v0.1 — 12-emotion parameter voice cloning (Apache 2.0)
  - F5-TTS — zero-shot voice cloning, 10x realtime
  - Stable Audio 2.0 — high-quality 44kHz stereo music gen (API)
  - AudioSep — text-queried audio source separation
  - Demucs (Meta) — stem separation (vocals, bass, drums, other)

Activated when cortex detects music composition, voice cloning, or advanced audio intent
beyond basic Audition automation.
"""

import logging
import requests as _req
import os

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

EVE_URL       = "http://127.0.0.1:8870"
MUSICGEN_URL  = "http://127.0.0.1:7970"   # MusicGen local gradio
COSYVOICE_URL = "http://127.0.0.1:7971"   # CosyVoice 2 local server
FISHSPEECH_URL= "http://127.0.0.1:7972"   # Fish Speech local
ZONOS_URL     = "http://127.0.0.1:7973"   # Zonos local server
DEMUCS_URL    = "http://127.0.0.1:7974"   # Demucs local server

_MUSIC_KEYWORDS = {
    # Music generation
    "compose", "composition", "melody", "chord", "chord progression", "harmony",
    "song", "music", "beat", "drum", "bass", "guitar", "piano", "synth",
    "lofi", "edm", "jazz", "classical", "hip hop", "trap", "ambient",
    "bgm", "background music", "soundtrack", "score", "jingle", "theme",
    "musicgen", "suno", "udio", "stable audio", "audiocraft",
    # Advanced voice
    "voice clone", "voice cloning", "clone voice", "voice synthesis",
    "tts style", "emotion voice", "crying voice", "whispering", "singing voice",
    "cosyvoice", "fish speech", "zonos", "f5-tts", "xtts", "styletts",
    "zero-shot voice", "few-shot voice",
    # Audio separation
    "stem", "stem separation", "vocal removal", "instrumental", "karaoke",
    "demucs", "audiosep", "isolate", "extract vocal", "extract drums",
    "source separation", "unmix",
    # Sound design
    "sound effect", "sfx", "foley", "sound design", "audio fx",
    "sample", "sample pack", "one shot", "loop",
}


def _detect_music_intent(message: str) -> bool:
    low = message.lower()
    return any(kw in low for kw in _MUSIC_KEYWORDS)


class MusicCell(BaseCell):
    name        = "music"
    description = "AI music & voice — MusicGen, CosyVoice 2, Fish Speech, Zonos, Demucs, Suno"
    color       = "#ec4899"
    lazy        = True
    position    = (4, 2)

    def __init__(self):
        super().__init__()
        self._suno_key:   str = ""
        self._udio_key:   str = ""
        self._fal_key:    str = ""
        self._backends:   dict = {}

    async def boot(self) -> None:
        self._suno_key = os.environ.get("SUNO_API_KEY", "")
        self._udio_key = os.environ.get("UDIO_API_KEY", "")
        self._fal_key  = os.environ.get("FAL_KEY", "")
        self._backends = self._probe_backends()
        online = [k for k, v in self._backends.items() if v]
        if not online:
            self._status = CellStatus.DEGRADED
            logger.warning("[MusicCell] No music backends available")
        else:
            logger.info("[MusicCell] Online: %s", online)

    def _probe_backends(self) -> dict:
        return {
            "musicgen_local":  self._ping(f"{MUSICGEN_URL}/"),
            "cosyvoice_local": self._ping(f"{COSYVOICE_URL}/"),
            "fishspeech_local":self._ping(f"{FISHSPEECH_URL}/"),
            "zonos_local":     self._ping(f"{ZONOS_URL}/"),
            "demucs_local":    self._ping(f"{DEMUCS_URL}/"),
            "suno_api":        bool(self._suno_key),
            "udio_api":        bool(self._udio_key),
            "fal_audio":       bool(self._fal_key),
        }

    def _ping(self, url: str) -> bool:
        try:
            return _req.get(url, timeout=2).status_code < 400
        except Exception:
            return False

    async def process(self, ctx: CellContext) -> dict:
        is_music = _detect_music_intent(ctx.message)
        return {
            "music_intent_detected": is_music,
            "backends": self._backends,
            "suggested_tool": self._suggest_tool(ctx.message) if is_music else None,
        }

    def _suggest_tool(self, message: str) -> str:
        low = message.lower()
        if any(x in low for x in ["stem", "separate", "demucs", "isolate vocal", "karaoke"]):
            return "demucs"
        if any(x in low for x in ["voice clone", "clone", "my voice", "cosyvoice", "fish speech"]):
            return "cosyvoice2" if self._backends.get("cosyvoice_local") else "fish_speech"
        if any(x in low for x in ["emotion", "cry", "whisper", "zonos", "12 emotion"]):
            return "zonos"
        if any(x in low for x in ["full song", "lyrics", "suno", "udio", "radio"]):
            return "suno_api" if self._suno_key else "musicgen"
        if any(x in low for x in ["compose", "melody", "chord", "instrumental", "bgm"]):
            return "musicgen_local" if self._backends.get("musicgen_local") else "fal_musicgen"
        return "musicgen"

    def health(self) -> dict:
        return {
            "backends": self._probe_backends(),
            "suno_key": bool(self._suno_key),
            "udio_key": bool(self._udio_key),
        }
