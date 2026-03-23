"""
ThreeDCell — 3D asset generation & auto-rigging brain cell.

Integrates:
  - Tripo 3.0 API (tripoai.com) — image/text → 3D + auto-rigging for VTuber assets
  - TRELLIS.2 (Microsoft, ICLR 2026) — 4B param image → 3D with PBR materials, locally hosted
  - InstantMesh — fast multi-view → mesh (single GPU, ~20s)
  - TripoSR — single image → mesh (Stability AI, <1s)
  - Shap-E (OpenAI) — text/image → 3D

Activated when cortex detects 3D modeling, rigging, or VTuber asset intent.
"""

import logging
import requests as _req

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

EVE_URL       = "http://127.0.0.1:8870"
TRELLIS_URL   = "http://127.0.0.1:7960"   # TRELLIS local server (when installed)
TRIPOSR_URL   = "http://127.0.0.1:7961"   # TripoSR local gradio

_3D_KEYWORDS = {
    # 3D modeling
    "3d", "mesh", "model", "object", "sculpt", "geometry", "polygon", "vertices",
    "wireframe", "topology", "remesh", "retopology", "normal map", "uv unwrap",
    # Materials
    "pbr", "material", "texture", "diffuse", "roughness", "metallic", "albedo",
    "specular", "emission", "bump map", "displacement",
    # Export
    "glb", "gltf", "fbx", "obj", "stl", "ply", "3mf", "dae", "blend",
    # Rigging
    "rig", "rigging", "skeleton", "bone", "armature", "weight paint", "skinning",
    "auto-rig", "control rig", "inverse kinematics", "ik", "blend shape", "morph",
    # Rendering
    "render", "blender", "cinema 4d", "maya", "3ds max", "zbrush", "substance",
    "octane", "cycles", "eevee", "arnold", "redshift",
    # VTuber specific
    "moc3", "live2d cubism", "vtube 3d", "vroid", "vtuber model", "avatar model",
    "character model", "puppet", "3d avatar",
    # Specific tools
    "trellis", "triposr", "instantmesh", "tripo", "shap-e", "one-2-3-45",
    "point-e", "meshy", "luma genie",
}


def _detect_3d_intent(message: str) -> bool:
    low = message.lower()
    return any(kw in low for kw in _3D_KEYWORDS)


class ThreeDCell(BaseCell):
    name        = "threed"
    description = "3D asset generation — Tripo 3.0, TRELLIS.2, InstantMesh, TripoSR"
    color       = "#8b5cf6"
    lazy        = True
    position    = (4, 1)

    def __init__(self):
        super().__init__()
        self._tripo_key: str = ""
        self._fal_key: str   = ""
        self._backends: dict = {}

    async def boot(self) -> None:
        import os
        self._tripo_key = os.environ.get("TRIPO_API_KEY", "")
        self._fal_key   = os.environ.get("FAL_KEY", "")
        self._backends  = self._probe_backends()
        if not any(self._backends.values()):
            self._status = CellStatus.DEGRADED
            logger.warning("[ThreeDCell] No 3D backends available")
        else:
            online = [k for k, v in self._backends.items() if v]
            logger.info("[ThreeDCell] Online backends: %s", online)

    def _probe_backends(self) -> dict:
        return {
            "trellis_local": self._ping(f"{TRELLIS_URL}/"),
            "triposr_local": self._ping(f"{TRIPOSR_URL}/"),
            "tripo_api":     bool(self._tripo_key),
            "fal_api":       bool(self._fal_key),
        }

    def _ping(self, url: str) -> bool:
        try:
            return _req.get(url, timeout=2).status_code < 400
        except Exception:
            return False

    async def process(self, ctx: CellContext) -> dict:
        is_3d = _detect_3d_intent(ctx.message)
        return {
            "3d_intent_detected": is_3d,
            "backends": self._backends,
            "tripo_api_available": bool(self._tripo_key),
            "fal_3d_available": bool(self._fal_key),
            "trellis_local": self._backends.get("trellis_local", False),
            "triposr_local": self._backends.get("triposr_local", False),
            "suggested_workflow": self._suggest_workflow(ctx.message) if is_3d else None,
        }

    def _suggest_workflow(self, message: str) -> str:
        low = message.lower()
        if any(x in low for x in ["vtube", "vtuber", "moc3", "live2d", "rig"]):
            return "tripo_with_autorig"   # Tripo 3.0 API — includes auto-rigging
        if any(x in low for x in ["pbr", "material", "realistic", "game asset"]):
            return "trellis_pbr"          # TRELLIS.2 — best PBR material quality
        if any(x in low for x in ["fast", "quick", "instant", "sketch"]):
            return "triposr_fast"         # TripoSR — <1s single image
        return "instantmesh"             # Default: InstantMesh balanced

    def health(self) -> dict:
        return {
            "backends":    self._probe_backends(),
            "tripo_key":   bool(self._tripo_key),
            "fal_key":     bool(self._fal_key),
        }
