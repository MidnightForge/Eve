"""
ToolsCell — system tool access via the Tool Server at :8769.

Provides file read/write, code execution, and download capabilities.
Activated when the Cortex detects tool-use intent.
"""

import logging
import requests as _req

from brain.base_cell import BaseCell, CellContext, CellStatus

logger = logging.getLogger(__name__)

TOOL_URL = "http://127.0.0.1:8769"


class ToolsCell(BaseCell):
    name        = "tools"
    description = "Tool Server — filesystem, execute, download"
    color       = "#0284c7"
    lazy        = True
    position    = (1, 2)

    async def boot(self) -> None:
        try:
            r = _req.get(f"{TOOL_URL}/health", timeout=2)
            if r.status_code < 400:
                logger.info("[ToolsCell] Tool server online")
            else:
                self._status = CellStatus.DEGRADED
        except Exception as exc:
            self._status = CellStatus.DEGRADED
            logger.warning("[ToolsCell] Tool server unreachable: %s", exc)

    async def process(self, ctx: CellContext):
        return {"tool_server": TOOL_URL, "available": self._status == CellStatus.ACTIVE}

    def health(self) -> dict:
        try:
            r = _req.get(f"{TOOL_URL}/health", timeout=1)
            return {"tool_server_online": r.status_code < 400}
        except Exception:
            return {"tool_server_online": False}
