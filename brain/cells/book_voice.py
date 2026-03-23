"""
BookVoiceCell — brain/cells/book_voice.py

Assigns unique Kokoro voice profiles to characters, generates TTS per
dialogue segment, and concatenates WAVs into full chapter audio.

Character → voice mapping is stored in ChromaDB for cross-chapter persistence.
Calls the voice sidecar at http://127.0.0.1:8766/tts for actual audio synthesis.

REST endpoints:
  POST /brain/book/voice          — generate TTS for a text snippet
  POST /brain/book/voice_chapter  — generate full chapter audio
  GET  /brain/book/characters/{title} — list character voice profiles
"""

from __future__ import annotations

import io
import json
import logging
import re
import struct
import time
import wave
from pathlib import Path
from typing import Any, Optional

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

_VOICE_URL   = "http://127.0.0.1:8766"
_MEMORY_URL  = "http://127.0.0.1:8767"

# ── Voice profile pool ────────────────────────────────────────────────────────
# Each profile: (speaker_id, speed, pitch_semitones)
_NARRATOR_PROFILE = {"speaker": "af_heart",    "speed": 1.0,  "pitch": 0}

_VOICE_POOL = [
    # Males
    {"speaker": "am_michael", "speed": 0.95, "pitch": -2,  "gender": "male",   "archetype": "default_male"},
    {"speaker": "am_adam",    "speed": 0.90, "pitch": -3,  "gender": "male",   "archetype": "villain"},
    {"speaker": "am_michael", "speed": 1.0,  "pitch": 0,   "gender": "male",   "archetype": "neutral_male"},
    # Females
    {"speaker": "af_sky",     "speed": 1.0,  "pitch": 1,   "gender": "female", "archetype": "default_female"},
    {"speaker": "af_sarah",   "speed": 1.05, "pitch": 2,   "gender": "female", "archetype": "bright_female"},
    {"speaker": "af_sky",     "speed": 1.1,  "pitch": 3,   "gender": "child",  "archetype": "child"},
    {"speaker": "af_heart",   "speed": 0.95, "pitch": -1,  "gender": "female", "archetype": "older_female"},
]

_ATTRIBUTION_PATTERNS = re.compile(
    r'[""]([^""]+)[""]\s*,?\s*(said|whispered|shouted|replied|asked|muttered|cried|yelled|called|hissed|growled|snapped|breathed|laughed|sighed|began|continued|added|answered)\s+(\w+)',
    re.IGNORECASE,
)

_DIALOGUE_SPLIT = re.compile(r'([""][^""]*[""])', re.DOTALL)


# ── Session key helpers ───────────────────────────────────────────────────────

def _char_session(book_title: str) -> str:
    return f"book_characters_{re.sub(r'[^a-z0-9]', '_', book_title.lower())}"


# ── SSML Preprocessing ────────────────────────────────────────────────────────
# Emotion-to-speed mapping based on dialogue tags in text
_EMOTION_TAGS = {
    "shouted":   {"speed_delta": +0.08, "label": "shout"},
    "yelled":    {"speed_delta": +0.08, "label": "shout"},
    "cried":     {"speed_delta": +0.06, "label": "emotional"},
    "whispered": {"speed_delta": -0.10, "label": "whisper"},
    "murmured":  {"speed_delta": -0.08, "label": "whisper"},
    "hissed":    {"speed_delta": -0.05, "label": "hiss"},
    "laughed":   {"speed_delta": +0.05, "label": "laugh"},
    "sighed":    {"speed_delta": -0.07, "label": "sigh"},
    "growled":   {"speed_delta": -0.05, "label": "growl"},
    "sobbed":    {"speed_delta": -0.08, "label": "emotional"},
    "screamed":  {"speed_delta": +0.10, "label": "shout"},
}

# Sentence splitter (handles Mr., Dr., Mrs., etc.)
_SENTENCE_ABBREV = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Inc|Ltd|Corp|St|Ave|Blvd|Dept|approx|est|fig|vol|no)\.",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\u201c\u2018])')

# Emphasis detection: words in ALL CAPS (3+ chars) or *italics* markers
_CAPS_EMPHASIS = re.compile(r'\b([A-Z]{3,})\b')
_ITALIC_EMPHASIS = re.compile(r'\*([^*]+)\*')


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences, respecting common abbreviations.
    This is the ElevenLabs/Tortoise approach: sentence-level TTS gives better prosody
    than paragraph-level because intonation resets at natural boundaries.
    """
    # Temporarily replace abbreviation periods to protect them
    protected = _SENTENCE_ABBREV.sub(lambda m: m.group(0).replace(".", "\x01"), text)
    sentences = _SENTENCE_SPLIT.split(protected)
    # Restore periods
    return [s.replace("\x01", ".").strip() for s in sentences if s.strip()]


def _detect_emotion_speed(text: str, base_speed: float) -> float:
    """
    Detect emotional cues in surrounding text and adjust TTS speed.
    Example: "he whispered" → speed -0.10 for that dialogue segment.
    """
    text_lower = text.lower()
    for tag, params in _EMOTION_TAGS.items():
        if tag in text_lower:
            adjusted = base_speed + params["speed_delta"]
            return max(0.7, min(1.3, adjusted))  # clamp to reasonable range
    return base_speed


def _has_emphasis(text: str) -> bool:
    """Check if text contains ALL CAPS or *italic* emphasis markers."""
    return bool(_CAPS_EMPHASIS.search(text) or _ITALIC_EMPHASIS.search(text))


def _is_dialogue(text: str) -> bool:
    """Check if text segment is primarily dialogue."""
    return text.startswith('"') or text.startswith('\u201c') or text.startswith("'")


# ── TTS call ─────────────────────────────────────────────────────────────────

def _tts(text: str, speaker: str, speed: float = 1.0) -> Optional[bytes]:
    """Call voice sidecar TTS endpoint. Returns WAV bytes or None."""
    # Cap text length per TTS call — sentence-level chunking prevents long inputs
    text = text[:600].strip()
    if not text:
        return None
    try:
        import requests
        r = requests.post(
            f"{_VOICE_URL}/tts",
            json={"text": text, "speaker": speaker, "speed": round(speed, 3)},
            timeout=30,
        )
        if r.status_code == 200:
            return r.content
    except Exception as e:
        logger.warning("[BookVoice] TTS call failed: %s", e)
    return None


# ── WAV concatenation ─────────────────────────────────────────────────────────

def _concat_wavs(wav_chunks: list[bytes]) -> bytes:
    """Concatenate multiple WAV byte strings into one WAV file."""
    if not wav_chunks:
        return b""
    if len(wav_chunks) == 1:
        return wav_chunks[0]

    # Read parameters from first chunk
    with wave.open(io.BytesIO(wav_chunks[0])) as w:
        params = w.getparams()
        all_frames = [w.readframes(w.getnframes())]

    for chunk in wav_chunks[1:]:
        try:
            with wave.open(io.BytesIO(chunk)) as w:
                all_frames.append(w.readframes(w.getnframes()))
        except Exception:
            continue

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        for frames in all_frames:
            out.writeframes(frames)
    return buf.getvalue()


def _silence_wav(duration_s: float, sample_rate: int = 22050) -> bytes:
    """Generate a WAV of pure silence."""
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


# ── Character management ───────────────────────────────────────────────────────

class CharacterVoiceMap:
    """Manages character → voice profile assignments, persists in ChromaDB."""

    def __init__(self, book_title: str):
        self.book_title = book_title
        self._map: dict[str, dict] = {}
        self._pool_idx = 0
        self._load()

    def _load(self) -> None:
        """Load existing character map from ChromaDB."""
        try:
            import requests
            session = _char_session(self.book_title)
            r = requests.post(
                f"{_MEMORY_URL}/inject",
                json={"query": "character voice profile", "top_k": 20,
                      "threshold": 0.0, "session_id": session},
                timeout=3,
            )
            injection = r.json().get("injection", "")
            # Parse stored JSON blocks
            for m in re.finditer(r"\{[^}]+\}", injection):
                try:
                    entry = json.loads(m.group(0))
                    if "character" in entry and "speaker" in entry:
                        self._map[entry["character"].lower()] = entry
                except Exception:
                    pass
        except Exception:
            pass

    def _save_character(self, name: str, profile: dict) -> None:
        try:
            import requests
            session = _char_session(self.book_title)
            requests.post(
                f"{_MEMORY_URL}/save",
                json={
                    "user_input":   f"[Character] {name}",
                    "eve_response": json.dumps(profile),
                    "session_id":   session,
                },
                timeout=3,
            )
        except Exception:
            pass

    def assign(self, name: str, gender_hint: str = "") -> dict:
        """Get or create a voice profile for a character."""
        key = name.lower()
        if key in self._map:
            return self._map[key]

        # Pick next profile from pool
        if gender_hint == "child":
            profile = _VOICE_POOL[5].copy()
        else:
            profile = _VOICE_POOL[self._pool_idx % len(_VOICE_POOL)].copy()
            self._pool_idx += 1

        profile["character"] = name
        self._map[key] = profile
        self._save_character(name, profile)
        logger.info("[BookVoice] Assigned '%s' → speaker=%s", name, profile["speaker"])
        return profile

    def get_all(self) -> list[dict]:
        return list(self._map.values())


# ── Chapter segmentation ──────────────────────────────────────────────────────

def _segment_chapter(text: str) -> list[dict]:
    """
    Split chapter into {speaker, text} segments.
    Identifies narrator vs. character dialogue.
    """
    segments = []
    pos = 0
    attribution_map: dict[str, str] = {}

    # First pass: collect all attributions in the chapter
    for m in _ATTRIBUTION_PATTERNS.finditer(text):
        character = m.group(3).strip()
        attribution_map[character.lower()] = character

    # Second pass: split by dialogue vs narrative
    parts = _DIALOGUE_SPLIT.split(text)
    last_speaker = "Narrator"

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith('"') or part.startswith('\u201c'):
            # Dialogue segment — look for attribution nearby
            # Find the text around this segment
            idx = text.find(part, pos)
            surrounding = text[idx:idx+len(part)+100] if idx >= 0 else ""
            attr_match = _ATTRIBUTION_PATTERNS.search(surrounding)
            if attr_match:
                last_speaker = attr_match.group(3).strip()
            segments.append({"speaker": last_speaker, "text": part, "is_dialogue": True})
        else:
            # Narrative segment
            # Check if contains attribution that sets next speaker
            attr_match = _ATTRIBUTION_PATTERNS.search(part)
            if attr_match:
                last_speaker = attr_match.group(3).strip()
            segments.append({"speaker": "Narrator", "text": part, "is_dialogue": False})

        pos += len(part)

    return [s for s in segments if len(s["text"].strip()) > 5]


# ── BookVoiceCell ─────────────────────────────────────────────────────────────

class BookVoiceCell(BaseCell):
    name        = "book_voice"
    description = (
        "Assigns unique Kokoro voice profiles to book characters, "
        "generates TTS per dialogue segment via voice sidecar, "
        "concatenates chapter audio with proper silence gaps."
    )
    color       = "#0e7490"
    lazy        = True
    position    = (5, 4)

    system_tier     = "online"
    hardware_req    = "Voice sidecar (port 8766) + Kokoro TTS"
    research_basis  = "Audiobook narration conventions + Kokoro multi-speaker TTS"
    build_notes     = (
        "LIVE: character detection from dialogue attribution patterns. "
        "Voice pool: 7 profiles (male/female/child/villain/narrator). "
        "ChromaDB persistence for character→voice across chapters. "
        "POST /brain/book/voice | POST /brain/book/voice_chapter | GET /brain/book/characters/{title}"
    )
    framework_layer = "Agentic AI"

    def __init__(self):
        super().__init__()
        self._char_maps: dict[str, CharacterVoiceMap] = {}

    async def boot(self) -> None:
        logger.info("[BookVoice] Cell online")

    async def process(self, ctx: CellContext) -> Any:
        return {"status": "book_voice ready — use REST endpoints"}

    def _get_char_map(self, book_title: str) -> CharacterVoiceMap:
        if book_title not in self._char_maps:
            self._char_maps[book_title] = CharacterVoiceMap(book_title)
        return self._char_maps[book_title]

    def speak(self, text: str, book_title: str, character: Optional[str] = None) -> Optional[bytes]:
        """Generate TTS for a text chunk. Returns WAV bytes or None."""
        char_map = self._get_char_map(book_title)

        if character and character.lower() != "narrator":
            profile = char_map.assign(character)
        else:
            profile = _NARRATOR_PROFILE

        return _tts(text, profile["speaker"], profile.get("speed", 1.0))

    def generate_chapter_audio(self, chapter_text: str, book_title: str,
                                is_first_chapter: bool = False) -> dict:
        """
        Generate full audio for a chapter using sentence-level TTS chunking.

        Key technique (from ElevenLabs/Tortoise research):
        - Split into sentences first — better prosody than paragraph-level TTS
        - Apply SSML-inspired speed adjustments per segment type
        - Narration: 0.95 speed (slightly slower for clarity)
        - Dialogue: 1.02 speed (natural conversation pace)
        - Emotional cues ("whispered", "shouted") adjust speed dynamically
        - Sentence gap: 0.3s | Paragraph gap: 0.8s | Chapter opening: 3.0s

        Returns {wav_bytes (bytes), characters_used (list), duration_estimate_s (float)}.
        """
        char_map = self._get_char_map(book_title)
        segments = _segment_chapter(chapter_text)

        wav_chunks  = []
        chars_used  = set()

        # Silence constants (research-based industry standards)
        sent_silence  = _silence_wav(0.3)   # between sentences
        para_silence  = _silence_wav(0.8)   # between paragraphs
        chap_silence  = _silence_wav(3.0)   # chapter opening pause

        # Chapter opening pause (3s before first sentence — gives listener time to settle)
        if is_first_chapter:
            wav_chunks.append(chap_silence)

        for seg_idx, seg in enumerate(segments):
            speaker_name = seg["speaker"]
            text         = seg["text"].strip()
            is_dialogue  = seg.get("is_dialogue", False)

            if not text:
                continue

            if speaker_name == "Narrator":
                profile = _NARRATOR_PROFILE.copy()
                # Narration: slightly slower (0.95) for clarity — industry standard
                base_speed = 0.95
            else:
                chars_used.add(speaker_name)
                profile = char_map.assign(speaker_name).copy()
                # Dialogue: slightly faster (1.02) for natural conversation feel
                base_speed = profile.get("speed", 1.0) * 1.02

            # SSML-inspired: detect emphasis, adjust speed
            if _has_emphasis(text):
                base_speed *= 0.93  # emphasised text reads slightly slower

            # Sentence-level chunking: split this segment into sentences
            sentences = _split_sentences(text)
            if not sentences:
                sentences = [text]

            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Per-sentence emotion detection
                sentence_speed = _detect_emotion_speed(sentence, base_speed)

                wav = _tts(sentence, profile["speaker"], sentence_speed)
                if wav:
                    wav_chunks.append(wav)
                    # 0.3s between sentences, 0.8s at paragraph/segment boundary
                    if sent_idx < len(sentences) - 1:
                        wav_chunks.append(sent_silence)
                    else:
                        wav_chunks.append(para_silence)

        combined = _concat_wavs(wav_chunks)

        # Duration estimate: 150 words/min narration, 165 words/min dialogue (audiobook standards)
        words = chapter_text.split()
        dialogue_word_count = sum(
            len(s["text"].split()) for s in segments if s.get("is_dialogue")
        )
        narration_word_count = len(words) - dialogue_word_count
        duration_est = (narration_word_count / 150 + dialogue_word_count / 165) * 60

        return {
            "wav_bytes":           combined,
            "characters_used":     list(chars_used),
            "segments_processed":  len(segments),
            "sentences_processed": sum(len(_split_sentences(s["text"])) for s in segments),
            "duration_estimate_s": round(duration_est, 1),
        }

    def get_characters(self, book_title: str) -> list[dict]:
        return self._get_char_map(book_title).get_all()

    def health(self) -> dict:
        ok = False
        try:
            import requests
            r = requests.get(f"{_VOICE_URL}/health", timeout=2)
            ok = r.status_code == 200
        except Exception:
            pass
        return {"status": self._status.value, "voice_sidecar": ok}
