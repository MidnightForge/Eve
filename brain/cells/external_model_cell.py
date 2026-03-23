"""
ExternalModelCell — FAL.AI unified API gateway + other external model providers.

Provides Eve access to 985+ models via a single API:
  - FAL.AI: image, video, audio, 3D, speech, upscaling, inpainting
  - Replicate: community model hub
  - Stability AI: SDXL, SD3, Stable Video
  - Google Vertex AI Imagen 3 / Veo 3.1 (when available)
  - Runway ML Gen-3 Alpha / Gen-3 Turbo
  - Pika Labs 2.1
  - Kling AI API (video gen)
  - HeyGen (talking head, avatar)
  - D-ID (talking head)

FAL.AI key endpoints relevant to Eve:
  - fal-ai/flux/dev — Flux Dev image gen
  - fal-ai/flux/schnell — Flux Schnell ultra-fast
  - fal-ai/wan-i2v — WAN image-to-video
  - fal-ai/seedvc — voice conversion
  - fal-ai/aura-sr — AuraSR 4x upscaling (sub-second)
  - fal-ai/trellis — TRELLIS 3D from image
  - fal-ai/elevenlabs/tts — ElevenLabs TTS
  - fal-ai/playht — PlayHT voice cloning
  - fal-ai/suno-ai/bark — Bark TTS
  - fal-ai/stable-audio — Stable Audio 2.0 music
  - fal-ai/minimax-video — Hailuo video gen
  - fal-ai/kling-video — Kling video
  - fal-ai/cogvideox-5b — CogVideoX video

Activated when local backends are unavailable or when the user explicitly
requests an external API, a specific model by name, or a capability beyond
what local tools provide.
"""

import logging
import os
import requests as _req

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

EVE_URL     = "http://127.0.0.1:8870"
FAL_API_URL = "https://queue.fal.run"

_EXTERNAL_KEYWORDS = {
    # FAL.AI direct mentions
    "fal.ai", "fal ai", "fal-ai", "fal api",
    # Replicate
    "replicate", "replicate.com",
    # Specific external models
    "runway", "gen-3", "gen3", "pika", "pika labs", "heygen", "d-id",
    "stable diffusion api", "stability ai", "kling api", "minimax video",
    "hailuo", "cogvideox", "wan api",
    # External voice
    "elevenlabs", "eleven labs", "playht", "play.ht",
    # External image
    "aura sr", "aurasr", "real-esrgan api",
    # When local fails
    "external", "cloud model", "api model", "online model",
    # Capability-based routing
    "best quality", "highest quality", "commercial grade",
}

# FAL model catalogue grouped by capability
FAL_MODELS = {
    "image_gen": [
        "fal-ai/flux/dev",
        "fal-ai/flux/schnell",
        "fal-ai/flux-pro",
        "fal-ai/stable-diffusion-v3-medium",
        "fal-ai/aura-flow",
    ],
    "video_gen": [
        "fal-ai/kling-video/v1.6/pro/text-to-video",
        "fal-ai/wan-i2v",
        "fal-ai/minimax-video/image-to-video",
        "fal-ai/cogvideox-5b",
        "fal-ai/hunyuan-video",
    ],
    "upscaling": [
        "fal-ai/aura-sr",
        "fal-ai/real-esrgan",
        "fal-ai/clarity-upscaler",
    ],
    "voice_tts": [
        "fal-ai/elevenlabs/tts",
        "fal-ai/playht",
        "fal-ai/kokoro/american-english",
    ],
    "voice_convert": [
        "fal-ai/seedvc",
    ],
    "threed": [
        "fal-ai/trellis",
        "fal-ai/triposr",
        "fal-ai/meshy-4",
    ],
    "audio_music": [
        "fal-ai/stable-audio",
        "fal-ai/mmaudio-v2",
    ],
    "talking_head": [
        "fal-ai/sadtalker",
        "fal-ai/musetalk",
        "fal-ai/latentsync",
    ],
    "lipsync": [
        "fal-ai/latentsync",
        "fal-ai/wav2lip",
    ],
}


def _detect_external_intent(message: str) -> bool:
    low = message.lower()
    return any(kw in low for kw in _EXTERNAL_KEYWORDS)


class ExternalModelCell(BaseCell):
    name        = "external_model"
    description = "FAL.AI + Replicate gateway — 985+ external AI models on demand"
    color       = "#06b6d4"
    lazy        = True
    position    = (5, 0)

    def __init__(self):
        super().__init__()
        self._fal_key:       str  = ""
        self._replicate_key: str  = ""
        self._fal_online:    bool = False

    async def boot(self) -> None:
        self._fal_key       = os.environ.get("FAL_KEY", "")
        self._replicate_key = os.environ.get("REPLICATE_API_TOKEN", "")
        if self._fal_key:
            self._fal_online = self._probe_fal()
        if not self._fal_key and not self._replicate_key:
            self._status = CellStatus.DEGRADED
            logger.warning("[ExternalModelCell] No API keys configured (FAL_KEY, REPLICATE_API_TOKEN)")
        else:
            logger.info("[ExternalModelCell] FAL=%s Replicate=%s",
                        bool(self._fal_key), bool(self._replicate_key))

    def _probe_fal(self) -> bool:
        """Check FAL.AI reachability with a lightweight HEAD request."""
        try:
            r = _req.head("https://fal.run", timeout=3)
            return r.status_code < 500
        except Exception:
            return False

    async def process(self, ctx: CellContext) -> dict:
        is_ext = _detect_external_intent(ctx.message)
        return {
            "external_intent_detected": is_ext,
            "fal_available": bool(self._fal_key),
            "replicate_available": bool(self._replicate_key),
            "fal_online": self._fal_online,
            "available_categories": list(FAL_MODELS.keys()),
            "suggested_models": self._suggest_models(ctx.message) if is_ext else [],
        }

    def _suggest_models(self, message: str) -> list:
        low = message.lower()
        suggestions = []
        if any(x in low for x in ["image", "picture", "photo", "flux", "generate"]):
            suggestions.extend(FAL_MODELS["image_gen"][:2])
        if any(x in low for x in ["video", "animate", "motion", "kling", "wan"]):
            suggestions.extend(FAL_MODELS["video_gen"][:2])
        if any(x in low for x in ["upscale", "enhance", "4x", "aura"]):
            suggestions.extend(FAL_MODELS["upscaling"][:2])
        if any(x in low for x in ["voice", "tts", "speech", "elevenlabs"]):
            suggestions.extend(FAL_MODELS["voice_tts"][:2])
        if any(x in low for x in ["3d", "mesh", "trellis", "triposr"]):
            suggestions.extend(FAL_MODELS["threed"][:2])
        if any(x in low for x in ["music", "audio", "song"]):
            suggestions.extend(FAL_MODELS["audio_music"])
        if any(x in low for x in ["lip sync", "lipsync", "talking head", "latentsync"]):
            suggestions.extend(FAL_MODELS["lipsync"])
        return suggestions or FAL_MODELS["image_gen"][:1]

    def get_fal_models(self, category: str = None) -> dict:
        """Return FAL model catalogue, optionally filtered by category."""
        if category:
            return {category: FAL_MODELS.get(category, [])}
        return FAL_MODELS

    def health(self) -> dict:
        return {
            "fal_key_set":       bool(self._fal_key),
            "replicate_key_set": bool(self._replicate_key),
            "fal_online":        self._fal_online,
            "model_categories":  list(FAL_MODELS.keys()),
            "total_fal_models":  sum(len(v) for v in FAL_MODELS.values()),
        }
