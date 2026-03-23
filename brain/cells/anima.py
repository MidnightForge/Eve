"""
AnimaCell — Avatar & VTuber expression coordinator.

Receives emotion signals from EmotionCell and drives:
  - The canvas avatar in index.html (via /ws/anima frames)
  - VTube Studio parameters (via the VTS bridge)
  - Live2D Lotes Eve expressions

Now also consumes VisionCell face data (get_latest_face_data) to:
  - Derive Eve's empathetic expression from Forge's live facial state
  - Keep the avatar alive with idle frames between speech
  - Blend real face energy into mouth animation even when silent

Expression priority (highest wins):
  1. Active TTS speech (anima_engine.synthesize drives mouth)
  2. Forge face-derived empathetic expression (VisionCell data)
  3. EmotionCell current emotion
  4. Fallback: neutral
"""

import asyncio
import logging
import requests as _req

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

EVE_URL = "http://127.0.0.1:8870"

_IDLE_FRAME_INTERVAL = 0.5   # seconds between idle frames


def _derive_expression_from_face(face_data: dict, emotion: str) -> str:
    """
    Map Forge's live facial metrics to Eve's empathetic expression.
    Falls back to EmotionCell emotion if no face data available.
    """
    if not face_data:
        return emotion or "neutral"

    expr       = face_data.get("expression", {})
    mouth_open = expr.get("mouth_open", 0.0)
    eye_left   = expr.get("eye_open_left",  1.0)
    eye_right  = expr.get("eye_open_right", 1.0)
    avg_eye    = (eye_left + eye_right) / 2.0

    if mouth_open > 0.35:
        return "happy"       # Forge laughing / wide smile -> Eve happy
    if mouth_open > 0.08:
        return "attentive"   # Forge speaking -> Eve listening
    if avg_eye < 0.05:
        return "tender"      # Forge eyes closed -> Eve tender / soft
    if avg_eye < 0.40:
        return "curious"     # Forge squinting / thinking -> Eve curious
    return emotion or "neutral"


class AnimaCell(BaseCell):
    name        = "anima"
    description = "ANIMA Engine — avatar expressions & VTuber drive"
    color       = "#f59e0b"
    lazy        = True
    position    = (1, 0)

    def __init__(self):
        super().__init__()
        self._last_expression = "neutral"
        self._face_expression = "neutral"
        self._idle_task       = None

    async def boot(self) -> None:
        self._idle_task = asyncio.create_task(self._idle_broadcaster())
        logger.info("[AnimaCell] Idle broadcaster started")

    async def process(self, ctx: CellContext) -> str:
        """Pull face data from VisionCell, derive Eve expression, push to avatar."""
        emotion = ctx.emotion_state or "neutral"

        try:
            from brain.cells.vision import get_latest_face_data
            derived = _derive_expression_from_face(get_latest_face_data(), emotion)
        except Exception:
            derived = emotion

        self._face_expression = derived

        if derived == self._last_expression:
            return derived

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: _req.post(
                    f"{EVE_URL}/avatar/set-expression",
                    json={"expression": derived},
                    timeout=1.0,
                )
            )
            self._last_expression = derived
            logger.debug("[AnimaCell] Expression -> %s (face-derived, base=%s)", derived, emotion)
        except Exception:
            pass

        return derived

    async def _idle_broadcaster(self):
        """
        Send idle AnimaFrames every 0.5s so the avatar stays alive between speech.
        Mouth energy comes from Forge's real face (subtle, scaled down).
        """
        await asyncio.sleep(3.0)
        while True:
            try:
                await asyncio.sleep(_IDLE_FRAME_INTERVAL)

                expr = self._face_expression or "neutral"
                mouth_idle = 0.0
                try:
                    from brain.cells.vision import get_latest_face_data
                    fd = get_latest_face_data()
                    if fd:
                        mouth_idle = min(
                            fd.get("expression", {}).get("mouth_open", 0.0) * 0.3,
                            0.15,
                        )
                except Exception:
                    pass

                idle_frame = {
                    "t": 0.0, "rms": mouth_idle, "mouth": mouth_idle,
                    "phoneme": "", "expression": expr, "speaking": False,
                }
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda f=idle_frame: _req.post(
                        f"{EVE_URL}/anima/idle-frame",
                        json=f, timeout=0.5,
                    )
                )
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def get_active_expression(self) -> str:
        """TTS pipeline reads this to pick base expression for speech synthesis."""
        return self._face_expression or self._last_expression or "neutral"

    def health(self) -> dict:
        return {
            "last_expression":  self._last_expression,
            "face_expression":  self._face_expression,
            "idle_broadcaster": self._idle_task is not None and not self._idle_task.done(),
        }
