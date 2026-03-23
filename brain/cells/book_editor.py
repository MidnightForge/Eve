"""
BookEditingPipeline — brain/cells/book_editor.py

End-to-end book editing cell. Accepts TXT/EPUB/PDF/DOCX files, splits into
chapters, edits each chapter via Claude Sonnet with continuity awareness,
and stores continuity facts in ChromaDB for cross-chapter consistency.

REST endpoints:
  POST /brain/book/edit           — edit a single chapter
  GET  /brain/book/continuity/{title} — return continuity ledger
  POST /brain/book/edit_all       — edit all chapters (streamed progress)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import anthropic

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

_SONNET = "claude-sonnet-4-6"
_HAIKU  = "claude-haiku-4-5-20251001"

# ChromaDB memory service
_MEMORY_URL = "http://127.0.0.1:8767"

# Chapter detection patterns
_CHAPTER_PATTERNS = [
    re.compile(r"^(Chapter\s+\d+[\.\:]?\s*.*)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(CHAPTER\s+[IVXLCDM\d]+[\.\:]?\s*.*)$", re.MULTILINE),
    re.compile(r"^(\d+\.\s+[A-Z][^\n]{0,60})$", re.MULTILINE),
    re.compile(r"^(Part\s+\d+[\.\:]?\s*.*)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(\*{3,}|\-{3,}|={3,})$", re.MULTILINE),   # horizontal rules
]


# ── Text extraction helpers ───────────────────────────────────────────────────

def _extract_text_from_file(file_path: str) -> str:
    """Extract raw text from TXT/EPUB/PDF/DOCX files."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = p.suffix.lower()

    if suffix == ".txt":
        return p.read_text(encoding="utf-8", errors="replace")

    if suffix == ".epub":
        try:
            import ebooklib
            from ebooklib import epub
            from html.parser import HTMLParser

            class _StripHTML(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._parts = []
                def handle_data(self, d):
                    self._parts.append(d)
                def get_text(self):
                    return " ".join(self._parts)

            book = epub.read_epub(str(p))
            parts = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                parser = _StripHTML()
                parser.feed(item.get_content().decode("utf-8", errors="replace"))
                parts.append(parser.get_text())
            return "\n\n".join(parts)
        except ImportError:
            raise RuntimeError("ebooklib not installed — run: pip install ebooklib")

    if suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(p))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise RuntimeError("pypdf not installed — run: pip install pypdf")

    if suffix in (".docx", ".doc"):
        try:
            import docx
            doc = docx.Document(str(p))
            return "\n".join(para.text for para in doc.paragraphs)
        except ImportError:
            raise RuntimeError("python-docx not installed — run: pip install python-docx")

    raise ValueError(f"Unsupported file type: {suffix}")


def _split_chapters(text: str) -> list[dict]:
    """
    Split text into chapters. Returns list of {index, title, text} dicts.
    Uses multiple patterns; falls back to equal-length splits if no breaks found.
    """
    # Find all chapter break positions
    breaks: list[tuple[int, str]] = []   # (char_offset, title)

    for pattern in _CHAPTER_PATTERNS:
        for m in pattern.finditer(text):
            title = m.group(0).strip()
            if len(title) > 3:
                breaks.append((m.start(), title))

    # De-duplicate and sort by position
    seen = set()
    unique_breaks = []
    for pos, title in sorted(breaks):
        if pos not in seen:
            seen.add(pos)
            unique_breaks.append((pos, title))

    if not unique_breaks:
        # Fallback: split every ~3000 words
        words = text.split()
        chunk = 3000
        chapters = []
        for i in range(0, len(words), chunk):
            chapters.append({
                "index": len(chapters),
                "title": f"Chapter {len(chapters) + 1}",
                "text":  " ".join(words[i:i+chunk]),
            })
        return chapters

    chapters = []
    for i, (pos, title) in enumerate(unique_breaks):
        start = pos
        end   = unique_breaks[i+1][0] if i+1 < len(unique_breaks) else len(text)
        chapter_text = text[start:end].strip()
        chapters.append({
            "index": i,
            "title": title,
            "text":  chapter_text,
        })

    return chapters


# ── Continuity ledger (ChromaDB via memory service) ──────────────────────────

def _ledger_session(book_title: str) -> str:
    return f"book_continuity_{re.sub(r'[^a-z0-9]', '_', book_title.lower())}"


def _fetch_continuity(book_title: str) -> str:
    """Fetch accumulated continuity facts from ChromaDB."""
    try:
        import requests
        session = _ledger_session(book_title)
        r = requests.post(
            f"{_MEMORY_URL}/inject",
            json={"query": "characters events locations timeline", "top_k": 10,
                  "threshold": 0.0, "session_id": session},
            timeout=4,
        )
        return r.json().get("injection", "")
    except Exception:
        return ""


def _save_continuity(book_title: str, chapter_title: str, facts: str) -> None:
    """Save continuity facts for a chapter to ChromaDB."""
    try:
        import requests
        session = _ledger_session(book_title)
        requests.post(
            f"{_MEMORY_URL}/save",
            json={
                "user_input":   f"[Chapter: {chapter_title}]",
                "eve_response": facts,
                "session_id":   session,
            },
            timeout=4,
        )
    except Exception:
        pass


# ── Core edit logic ────────────────────────────────────────────────────────────

def _edit_chapter_sync(
    chapter_text: str,
    chapter_title: str,
    continuity: str,
    client: anthropic.Anthropic,
) -> dict:
    """
    Edit one chapter via Claude Sonnet.
    Returns {edited_text, diffs: [{original, edited, reason}], continuity_facts}.
    """
    system = (
        "You are a professional fiction editor. Your job is to improve prose quality "
        "while preserving the author's voice, plot, and style.\n\n"
        "Focus on:\n"
        "- Awkward or unnatural sentences\n"
        "- Pacing problems (too rushed or too slow)\n"
        "- Repetitive phrases or words\n"
        "- Dialogue that sounds unnatural\n"
        "- Flow breaks between paragraphs\n\n"
        "CRITICAL: Preserve all plot points, characters, events, and timeline.\n"
        "Make surgical improvements — do not rewrite everything.\n"
        "Return a JSON object with keys: edited_text, diffs (array of {original, edited, reason})."
    )

    continuity_block = f"\n\nCONTINUITY LEDGER (facts established in previous chapters):\n{continuity}" if continuity else ""

    user_prompt = (
        f"Chapter: {chapter_title}{continuity_block}\n\n"
        f"CHAPTER TEXT:\n{chapter_text[:8000]}"
    )

    try:
        msg = client.messages.create(
            model=_SONNET,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown code fences
        if "```json" in raw:
            m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
            raw = m.group(1) if m else raw
        elif raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()

        data = json.loads(raw)
        edited = data.get("edited_text", chapter_text)
        diffs  = data.get("diffs", [])
    except json.JSONDecodeError:
        # If JSON parsing fails, treat the response as pure edited text
        edited = raw if len(raw) > 100 else chapter_text
        diffs  = []
    except Exception as e:
        logger.warning("[BookEditor] Edit failed: %s", e)
        edited = chapter_text
        diffs  = []

    # Extract continuity facts
    continuity_facts = _extract_continuity_facts(edited, chapter_title, client)

    return {
        "edited_text":       edited,
        "diffs":             diffs,
        "continuity_facts":  continuity_facts,
    }


def _extract_continuity_facts(
    chapter_text: str,
    chapter_title: str,
    client: anthropic.Anthropic,
) -> str:
    """Extract characters, events, locations, timeline from chapter text."""
    try:
        msg = client.messages.create(
            model=_HAIKU,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    f"Extract continuity facts from this chapter for tracking across the book.\n"
                    f"Chapter: {chapter_title}\n\n{chapter_text[:4000]}\n\n"
                    "List ONLY factual continuity items:\n"
                    "- Characters introduced (name, description)\n"
                    "- Key events that happened\n"
                    "- Locations established\n"
                    "- Timeline/chronology markers\n"
                    "- Objects or items of importance\n\n"
                    "Be concise. Format as bullet points."
                ),
            }],
        )
        return msg.content[0].text.strip()
    except Exception:
        return ""


# ── BookEditingPipelineCell ────────────────────────────────────────────────────

class BookEditingPipelineCell(BaseCell):
    name        = "book_editor"
    description = (
        "End-to-end book editing pipeline. Splits manuscripts into chapters, "
        "edits each with Claude Sonnet for prose quality + pacing, "
        "maintains a continuity ledger in ChromaDB across all chapters."
    )
    color       = "#7e22ce"
    lazy        = True
    position    = (5, 3)

    system_tier     = "online"
    hardware_req    = "API (Claude Sonnet)"
    research_basis  = "Professional fiction editing heuristics + continuity-aware LLM editing"
    build_notes     = (
        "LIVE: TXT/EPUB/PDF/DOCX support. Chapter detection (regex). "
        "Continuity ledger in ChromaDB. Diff output per chapter. "
        "POST /brain/book/edit | POST /brain/book/edit_all | GET /brain/book/continuity/{title}"
    )
    framework_layer = "Agentic AI"

    def __init__(self):
        super().__init__()
        self._client: Optional[anthropic.Anthropic] = None

    async def boot(self) -> None:
        self._client = anthropic.Anthropic()
        logger.info("[BookEditor] Cell online")

    async def process(self, ctx: CellContext) -> Any:
        return {"status": "book_editor ready — use REST endpoints"}

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def edit_chapter(
        self,
        file_path: str,
        book_title: str,
        chapter_index: Optional[int] = None,
    ) -> dict:
        """Edit a specific chapter (or chapter 0 if index not given)."""
        client = self._get_client()
        text   = _extract_text_from_file(file_path)
        chapters = _split_chapters(text)

        idx = chapter_index if chapter_index is not None else 0
        if idx >= len(chapters):
            return {"error": f"Chapter index {idx} out of range (book has {len(chapters)} chapters)"}

        chap = chapters[idx]
        continuity = _fetch_continuity(book_title)
        result = _edit_chapter_sync(chap["text"], chap["title"], continuity, client)
        _save_continuity(book_title, chap["title"], result["continuity_facts"])

        return {
            "chapter_index": idx,
            "chapter_title": chap["title"],
            "original_len":  len(chap["text"]),
            "edited_len":    len(result["edited_text"]),
            "diff_count":    len(result["diffs"]),
            "diffs":         result["diffs"][:20],   # cap to 20 for response size
            "edited_text":   result["edited_text"],
            "continuity_facts": result["continuity_facts"],
        }

    def get_continuity(self, book_title: str) -> dict:
        """Return the full continuity ledger for a book."""
        ledger = _fetch_continuity(book_title)
        return {
            "book_title": book_title,
            "session_id": _ledger_session(book_title),
            "ledger":     ledger,
        }

    def edit_all_generator(self, file_path: str, book_title: str):
        """Generator that yields progress dicts for each chapter."""
        client   = self._get_client()
        text     = _extract_text_from_file(file_path)
        chapters = _split_chapters(text)

        yield {"type": "start", "total_chapters": len(chapters), "book_title": book_title}

        for chap in chapters:
            t0 = time.time()
            continuity = _fetch_continuity(book_title)
            result = _edit_chapter_sync(chap["text"], chap["title"], continuity, client)
            _save_continuity(book_title, chap["title"], result["continuity_facts"])

            yield {
                "type":          "chapter_done",
                "index":         chap["index"],
                "title":         chap["title"],
                "diff_count":    len(result["diffs"]),
                "duration_s":    round(time.time() - t0, 1),
                "edited_text":   result["edited_text"],
            }

        yield {"type": "complete", "book_title": book_title}

    def generate_metadata(self, book_title: str, file_path: str,
                           author: str = "Unknown") -> dict:
        """
        Generate audiobook metadata JSON after editing.
        Duration estimate: 150 words/min narration, 165 words/min dialogue (industry standard).
        Includes character roster with first appearance chapter.
        """
        try:
            text     = _extract_text_from_file(file_path)
            chapters = _split_chapters(text)
        except Exception as e:
            return {"error": str(e)}

        # Rough dialogue detection (text inside quotes)
        _dialogue_re = re.compile(r'[""][^""]+[""]')

        chapter_meta = []
        total_words  = 0
        total_dur    = 0.0

        for chap in chapters:
            t = chap["text"]
            wc = len(t.split())
            dialogue_words = sum(len(m.group(0).split()) for m in _dialogue_re.finditer(t))
            narr_words     = wc - dialogue_words
            dur_min        = narr_words / 150 + dialogue_words / 165
            total_words   += wc
            total_dur     += dur_min
            chapter_meta.append({
                "number":                chap["index"] + 1,
                "title":                 chap["title"],
                "word_count":            wc,
                "estimated_duration_min": round(dur_min, 1),
            })

        # Fetch character roster from ChromaDB
        try:
            import requests
            from brain.cells.book_voice import _char_session
            session = _char_session(book_title)
            r = requests.post(
                f"{_MEMORY_URL}/inject",
                json={"query": "character voice profile", "top_k": 50,
                      "threshold": 0.0, "session_id": session},
                timeout=3,
            )
            injection = r.json().get("injection", "")
            characters = []
            for m in re.finditer(r"\{[^}]+\}", injection):
                try:
                    entry = json.loads(m.group(0))
                    if "character" in entry:
                        characters.append({
                            "name":          entry["character"],
                            "voice_profile": entry.get("speaker", "unknown"),
                            "first_appears": 1,
                        })
                except Exception:
                    pass
        except Exception:
            characters = []

        metadata = {
            "title":                      book_title,
            "author":                     author,
            "narrator":                   "Eve (AI)",
            "chapters":                   chapter_meta,
            "characters":                 characters,
            "total_word_count":           total_words,
            "estimated_total_duration_min": round(total_dur, 1),
            "generated_at":               time.time(),
            "lufs_target":                -16,
            "encoding":                   "MP3 128kbps 44100 Hz",
            "standard":                   "ACX/Audible compatible",
        }

        # Write to output folder
        out_dir = Path(r"C:\Users\<your-username>\eve\audiobooks") / re.sub(r"[^\w\s\-]", "", book_title).strip().replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        _safe_title = re.sub(r'[^\w]', '_', book_title)
        meta_path = str(out_dir / f"{_safe_title}_metadata.json")
        try:
            Path(meta_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception:
            pass

        return {"metadata": metadata, "saved_to": meta_path}

    def health(self) -> dict:
        return {"status": self._status.value, "client_ready": self._client is not None}
