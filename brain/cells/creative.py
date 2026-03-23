"""
CreativeCell — Lotus Forge / FNE image & video generation + Native Creative Engine.

Wraps the Forge Node Engine (FNE), ComfyUI, and the full native
eve_creative_engine suite (PhotoshopEngine, VideoEngine, AudioEngine,
LightroomEngine, VectorEngine, LayoutEngine, TrackingEngine, AssetBrowser).

Activated when cortex routes to "creative" or "design" intent.
"""

import logging
import requests as _req

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

FNE_URL     = "http://127.0.0.1:8873"
COMFYUI_URL = "http://127.0.0.1:8001"

# ── Design intent keywords used by cortex routing ────────────────────────────
DESIGN_KEYWORDS = {
    # Photoshop / raster
    "edit", "crop", "resize", "filter", "grade", "colour", "color", "enhance",
    "sharpen", "blur", "composite", "layer", "psd", "photoshop", "inpaint",
    "selection", "mask", "smart object", "adjustment layer", "neural filter",
    # Illustrator / vector
    "illustrator", "vector", "svg", "artboard", "pen tool", "pathfinder",
    # InDesign / layout
    "indesign", "layout", "print", "publication", "idml", "text frame",
    # Premiere / video editing
    "premiere", "sequence", "timeline", "cut", "splice", "color grade",
    "multicam", "lumetri", "video edit",
    # After Effects / motion
    "after effects", "composition", "motion", "keyframe", "expression",
    "visual effect", "vfx", "motion graphic", "mograph", "chroma key",
    "rotoscope", "tracking", "3d layer", "aerender",
    # Animate
    "animate", "flash", "timeline animation", "symbol", "tween", "jsfl",
    # Audition / audio
    "audition", "audio", "mix", "normalize", "pitch", "waveform", "track",
    "beat", "podcast", "noise reduction", "eq", "multitrack", "stem",
    "loudness", "lufs", "spectral",
    # Character Animator / VTube
    "character animator", "puppet", "rigging", "performance capture",
    "live puppet", "vtube drive",
    # Lightroom / RAW
    "lightroom", "raw", "develop", "preset", "catalog", "grade", "exposure",
    # Bridge / asset management
    "bridge", "asset", "rename", "browse", "portfolio", "metadata", "xmp",
    # Media Encoder
    "media encoder", "ame", "encode", "transcode", "convert", "batch",
    "watch folder",
    # General creative ops
    "pdf", "export", "upscale", "thumbnail", "face detect", "pose",
    "landmark", "tracking", "mediapipe",
}


def _detect_design_intent(message: str) -> bool:
    """Return True if message contains native creative engine keywords."""
    low = message.lower()
    return any(kw in low for kw in DESIGN_KEYWORDS)


class CreativeCell(BaseCell):
    name        = "creative"
    description = "Lotus Forge — image/video generation, native Adobe-replacement creative suite"
    color       = "#ea580c"
    lazy        = True
    position    = (3, 0)

    def __init__(self):
        super().__init__()
        self._engine_caps: dict = {}

    async def boot(self) -> None:
        fne_ok = self._ping(f"{FNE_URL}/fne/queue")
        cui_ok = self._ping(f"{COMFYUI_URL}/system_stats")
        # Check native creative engine
        nce_ok = self._ping("http://127.0.0.1:8870/creative/capabilities")
        if not fne_ok and not cui_ok and not nce_ok:
            self._status = CellStatus.DEGRADED
            logger.warning("[CreativeCell] No creative pipeline responding")
        else:
            logger.info("[CreativeCell] Creative pipeline online (FNE:%s ComfyUI:%s NativeEngine:%s)",
                        fne_ok, cui_ok, nce_ok)
        # Cache engine capabilities
        if nce_ok:
            try:
                r = _req.get("http://127.0.0.1:8870/creative/capabilities", timeout=3)
                self._engine_caps = r.json()
            except Exception:
                pass

    def _ping(self, url: str) -> bool:
        try:
            return _req.get(url, timeout=2).status_code < 400
        except Exception:
            return False

    async def process(self, ctx: CellContext) -> dict:
        """
        Signals what creative tools are available for this context.
        Includes native engine capability list so Cortex can suggest
        the right tool (e.g. AudioEngine for audio edits, not FNE).
        """
        is_design = _detect_design_intent(ctx.message)
        fne_ok    = self._ping(f"{FNE_URL}/fne/queue")
        nce_ok    = self._ping("http://127.0.0.1:8870/creative/capabilities")

        result = {
            "fne_available":    fne_ok,
            "comfyui_available": self._ping(f"{COMFYUI_URL}/system_stats"),
            "native_engine_available": nce_ok,
            "design_intent_detected": is_design,
        }

        # Surface relevant native engine modules for design requests
        if is_design and self._engine_caps:
            result["suggested_engines"] = _suggest_engines(ctx.message, self._engine_caps)

        return result

    def health(self) -> dict:
        nce_ok = self._ping("http://127.0.0.1:8870/creative/capabilities")
        caps = {}
        if nce_ok and self._engine_caps:
            caps = {mod: all(v for k, v in info.items() if k != "features")
                    for mod, info in self._engine_caps.items()}
        return {
            "fne_online":           self._ping(f"{FNE_URL}/fne/queue"),
            "comfyui_online":       self._ping(f"{COMFYUI_URL}/system_stats"),
            "native_engine_online": nce_ok,
            "engine_modules":       caps,
        }


def _suggest_engines(message: str, caps: dict) -> list:
    """Map message keywords to the most relevant native engine modules."""
    low = message.lower()
    suggestions = []
    keyword_map = {
        "photoshop": ["psd", "layer", "composite", "photoshop", "filter", "crop", "resize", "grade"],
        "video":     ["video", "mp4", "animate", "clip", "transcode", "frame", "cut", "concat"],
        "audio":     ["audio", "mp3", "wav", "music", "sound", "voice", "mix", "pitch", "bpm"],
        "lightroom": ["raw", "cr2", "nef", "develop", "lightroom", "batch export", "preset"],
        "vector":    ["svg", "vector", "illustrator", "shape", "icon", "logo", "draw"],
        "layout":    ["pdf", "layout", "print", "document", "page", "indesign", "report"],
        "tracking":  ["face", "pose", "body", "landmark", "detect", "track", "mediapipe"],
        "browser":   ["asset", "browse", "rename", "thumbnail", "portfolio", "folder"],
    }
    for engine, keywords in keyword_map.items():
        if any(kw in low for kw in keywords) and caps.get(engine):
            suggestions.append(engine)
    return suggestions or ["photoshop"]  # default to image editing
