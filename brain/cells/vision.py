"""
VisionCell — IRIS visual awareness + Native Creative Engine face/pose tracking.

Provides visual context to the Cortex by pulling the current IRIS
observation (face + screen). Now also runs mediapipe face/pose analysis
on live IRIS webcam frames for expression-aware context.
"""

import base64
import logging
import tempfile
import time
from pathlib import Path

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

# How often to run the heavier mediapipe tracking pass (seconds)
_TRACK_INTERVAL = 10.0
_last_track_time: float = 0.0
_last_face_data: dict = {}


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Simple word-overlap Jaccard similarity. No embedding model needed."""
    import re
    words_a = set(re.findall(r"\b[a-z]{3,}\b", text_a.lower()))
    words_b = set(re.findall(r"\b[a-z]{3,}\b", text_b.lower()))
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union        = words_a | words_b
    return len(intersection) / len(union)


class VisionCell(BaseCell):
    name        = "vision"
    description = "IRIS Vision — visual awareness + real-time face/pose tracking + predictive coding"
    color       = "#8b5cf6"
    lazy        = False
    position    = (3, 1)

    # Predictive coding threshold: below this = stable scene → brief delta only
    _SURPRISE_STABLE = 0.15

    def __init__(self):
        super().__init__()
        self._iris = None
        self._engine = None
        # Predictive coding state
        self._last_prediction: Optional[str] = None   # what we predicted next scene would be
        self._last_caption:    Optional[str] = None   # what we actually saw
        self._surprise_delta:  float         = 0.0    # current surprise score
        self._stable_count:    int           = 0      # consecutive stable frames
        self._total_frames:    int           = 0
        self._skipped_frames:  int           = 0

    async def boot(self) -> None:
        try:
            from iris import iris as _iris_instance
            self._iris = _iris_instance
            logger.info("[VisionCell] IRIS module attached")
        except Exception as exc:
            self._status = CellStatus.DEGRADED
            logger.warning("[VisionCell] IRIS not available: %s", exc)
        # Attach native creative engine for tracking
        try:
            from eve_creative_engine import get_engine
            self._engine = get_engine()
            logger.info("[VisionCell] TrackingEngine attached — mediapipe face/pose active")
        except Exception as exc:
            logger.warning("[VisionCell] TrackingEngine unavailable: %s", exc)

    async def process(self, ctx: CellContext) -> str:
        """Returns the current IRIS context string + live face expression data.
        Uses predictive coding to skip stable frames and focus on surprising changes."""
        global _last_track_time, _last_face_data

        self._total_frames += 1

        iris_context = ""
        if self._iris:
            try:
                iris_context = self._iris.get_context() or ""
            except Exception as exc:
                logger.debug("[VisionCell] get_context failed: %s", exc)

        # ── Predictive Coding ──────────────────────────────────────────────────
        if iris_context:
            iris_context = self._apply_predictive_coding(iris_context)

        ctx.iris_context = iris_context

        # Run mediapipe on live webcam frame (throttled)
        now = time.time()
        if self._engine and self._iris and (now - _last_track_time) > _TRACK_INTERVAL:
            face_data = _run_tracking_on_iris_frame(self._iris, self._engine)
            if face_data:
                _last_face_data = face_data
                _last_track_time = now
                # Inject expression context into conversation
                expr = face_data.get("expression", {})
                mouth = expr.get("mouth_open", 0)
                eye   = expr.get("eye_open_left", 0)
                if mouth > 0.05:
                    iris_context += " [Forge appears to be speaking or reacting]"
                if eye < 0.01:
                    iris_context += " [Forge's eyes appear closed or narrowed]"

        return iris_context

    def _apply_predictive_coding(self, actual_caption: str) -> str:
        """
        Predictive coding: compare actual caption with prediction.
        If surprise < threshold: scene is stable → return brief delta.
        If surprise > threshold: return full caption.
        Generate prediction for NEXT frame (zero-order hold: same as current).
        """
        # Compute surprise delta
        if self._last_prediction is not None:
            surprise = 1.0 - _jaccard_similarity(self._last_prediction, actual_caption)
        else:
            surprise = 1.0   # First frame is always surprising

        self._surprise_delta = surprise
        self._last_caption   = actual_caption

        # Update prediction for next frame (zero-order hold)
        self._last_prediction = actual_caption

        if surprise < self._SURPRISE_STABLE and self._last_prediction:
            # Scene is stable — return brief delta instead of full caption
            self._skipped_frames += 1
            self._stable_count   += 1
            return f"[IRIS stable scene — no significant change since last frame]"
        else:
            self._stable_count = 0
            return actual_caption

    def health(self) -> dict:
        base = {
            "iris_attached":   False,
            "tracking_engine": False,
            "last_face_data":  {},
            "predictive_coding": {
                "surprise_delta":  round(self._surprise_delta, 3),
                "stable_count":    self._stable_count,
                "total_frames":    self._total_frames,
                "skipped_frames":  self._skipped_frames,
                "skip_rate":       round(self._skipped_frames / max(self._total_frames, 1), 3),
            },
        }
        if not self._iris:
            return base
        try:
            from iris import iris as _i
            base.update({
                "iris_attached":      True,
                "face_active":        bool(_i.face_frame_b64),
                "screen_active":      bool(_i.screen_frame_b64),
                "tracking_engine":    self._engine is not None,
                "last_face_data":     _last_face_data,
            })
        except Exception:
            pass
        return base


def _run_tracking_on_iris_frame(iris_instance, engine) -> dict:
    """
    Decode the IRIS webcam face frame (base64 PNG) → write to temp file
    → run mediapipe TrackingEngine → return face landmark + expression data.
    """
    try:
        b64 = getattr(iris_instance, "face_frame_b64", None)
        if not b64:
            return {}
        img_bytes = base64.b64decode(b64)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tf.write(img_bytes)
            tmp_path = tf.name
        result = engine.tracking.analyze_face(tmp_path)
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass
        return result if result.get("faces", 0) > 0 else {}
    except Exception as e:
        logger.debug("[VisionCell] Tracking pass failed: %s", e)
        return {}


def get_latest_face_data() -> dict:
    """Public accessor — other systems (ANIMA, EmotionCell) can pull live face data."""
    return _last_face_data
