"""
WebCell — real-time internet access for Eve.

Activated when Cortex detects web-search, price-lookup, or
URL-fetch intent. Uses httpx for async non-blocking fetches.
"""

import asyncio
import logging
import re

import httpx

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


class WebCell(BaseCell):
    name        = "web"
    description = "Web Access — real-time search & URL fetch"
    color       = "#0f766e"
    lazy        = True
    position    = (0, 2)

    async def process(self, ctx: CellContext) -> dict:
        """Returns web context if the message contains URLs or search cues."""
        urls = re.findall(r'https?://\S+', ctx.message)
        if not urls:
            return {"fetched": False}

        results = {}
        async with httpx.AsyncClient(timeout=8.0, headers=_HEADERS, follow_redirects=True) as client:
            for url in urls[:2]:
                try:
                    r = await client.get(url)
                    # Strip HTML tags for a clean text snippet
                    text = re.sub(r'<[^>]+>', ' ', r.text)
                    text = re.sub(r'\s+', ' ', text).strip()[:2000]
                    results[url] = text
                except Exception as exc:
                    results[url] = f"[fetch failed: {exc}]"
        return {"fetched": bool(results), "pages": results}

    def health(self) -> dict:
        return {"internet_access": True}
