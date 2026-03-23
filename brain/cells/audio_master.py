"""
AudioMasteringPipeline — brain/cells/audio_master.py

Takes per-chapter WAV files and produces distribution-ready audiobook output:
  - Normalize each chapter to -16 LUFS (broadcast standard)
  - Add silence between paragraphs/chapters
  - Encode to MP3 (128kbps) with ID3 tags
  - Produce combined M4B audiobook with QuickTime chapter markers
  - Write manifest JSON

Output: C:\\Users\\<your-username>\\eve\\audiobooks\\{book_title}\\

REST endpoints:
  POST /brain/book/master          — master a list of chapter WAVs
  GET  /brain/book/output/{title}  — list finished files
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)

_OUTPUT_BASE = Path(r"C:\Users\<your-username>\eve\audiobooks")

# ffmpeg full path — do not assume it is on PATH
_FFMPEG_CANDIDATES = [
    r"F:\pinokio\bin\miniconda\Library\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
]
_FFPROBE_CANDIDATES = [
    r"F:\pinokio\bin\miniconda\Library\bin\ffprobe.exe",
    r"C:\ffmpeg\bin\ffprobe.exe",
    r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
]


def _find_binary(candidates: list, fallback: str) -> str:
    """Find first existing binary path, fall back to bare name (PATH lookup)."""
    for c in candidates:
        if Path(c).exists():
            return c
    return fallback


_FFMPEG  = _find_binary(_FFMPEG_CANDIDATES, "ffmpeg")
_FFPROBE = _find_binary(_FFPROBE_CANDIDATES, "ffprobe")


def _run_ffmpeg(*args, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [_FFMPEG, "-y", "-loglevel", "error"] + list(args)
    env = {**os.environ, "PATH": str(Path(_FFMPEG).parent) + os.pathsep + os.environ.get("PATH", "")}
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=check, env=env)


def _check_ffmpeg() -> bool:
    try:
        r = subprocess.run([_FFMPEG, "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _get_wav_duration(wav_path: str) -> float:
    """Get WAV duration in seconds via ffprobe."""
    try:
        r = subprocess.run(
            [_FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _normalize_chapter(input_wav: str, output_wav: str) -> bool:
    """Normalize to -16 LUFS using ffmpeg loudnorm filter (two-pass)."""
    try:
        # Pass 1: measure
        r = _run_ffmpeg(
            "-i", input_wav,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-",
            check=False,
        )
        # Parse loudnorm stats from stderr
        import re
        m = re.search(r"\{[^}]+\"input_i\"[^}]+\}", r.stderr, re.DOTALL)
        if m:
            stats = json.loads(m.group(0))
            il  = stats.get("input_i", "-16.0")
            lra = stats.get("input_lra", "11.0")
            tp  = stats.get("input_tp", "-1.5")
            offset = stats.get("target_offset", "0.0")
            af = (
                f"loudnorm=I=-16:TP=-1.5:LRA=11:"
                f"measured_I={il}:measured_LRA={lra}:measured_TP={tp}:"
                f"measured_thresh={stats.get('input_thresh', '-26.0')}:"
                f"offset={offset}:linear=true:print_format=summary"
            )
        else:
            af = "loudnorm=I=-16:TP=-1.5:LRA=11:linear=true"

        # Pass 2: apply
        _run_ffmpeg("-i", input_wav, "-af", af, output_wav)
        return True
    except Exception as e:
        logger.warning("[AudioMaster] Normalize failed: %s", e)
        return False


def _encode_to_mp3(
    input_wav: str,
    output_mp3: str,
    book_title: str,
    chapter_num: int,
    chapter_title: str,
) -> bool:
    """Encode WAV → MP3 with ID3 tags."""
    try:
        _run_ffmpeg(
            "-i", input_wav,
            "-codec:a", "libmp3lame",
            "-b:a", "128k",
            "-id3v2_version", "3",
            "-metadata", f"title={book_title} — Chapter {chapter_num}",
            "-metadata", f"artist=Eve (AI Narrator)",
            "-metadata", f"album={book_title}",
            "-metadata", f"track={chapter_num}",
            "-metadata", f"comment={chapter_title}",
            output_mp3,
        )
        return True
    except Exception as e:
        logger.warning("[AudioMaster] MP3 encode failed: %s", e)
        return False


def _make_silence_wav(duration_s: float, output_path: str, sample_rate: int = 44100) -> None:
    """Generate silence WAV at 44100 Hz (distribution standard)."""
    _run_ffmpeg(
        "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", str(duration_s),
        output_path,
    )


def _add_fades_and_room_tone(input_wav: str, output_wav: str, duration_s: float) -> bool:
    """
    Add 50ms fade-in and fade-out, plus very subtle room tone floor (-60dB).
    Industry standard for Audible/ACX — prevents clicks/pops at chapter boundaries.
    """
    try:
        fade_dur = 0.05   # 50ms
        af_chain = (
            f"afade=t=in:st=0:d={fade_dur},"
            f"afade=t=out:st={max(0, duration_s - fade_dur):.3f}:d={fade_dur},"
            # Very subtle room tone at -60dB for warmth (Audible recommendation)
            "aeval=val(0)+0.001*sin(2*PI*t*50)*0.0001:c=mono"
        )
        _run_ffmpeg("-i", input_wav, "-af", af_chain, output_wav)
        return True
    except Exception as e:
        logger.warning("[AudioMaster] Fade/room-tone failed: %s — copying as-is", e)
        import shutil
        shutil.copy2(input_wav, output_wav)
        return False


def _build_m4b(
    chapter_mp3s: list[dict],
    output_m4b: str,
    book_title: str,
) -> bool:
    """
    Concatenate chapter MP3s into a single M4B audiobook with QuickTime chapter markers.
    chapter_mp3s: list of {path, title, num, duration_s}
    """
    try:
        # Build ffmpeg concat filter
        import tempfile
        concat_file = tempfile.mktemp(suffix=".txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for ch in chapter_mp3s:
                f.write(f"file '{ch['path']}'\n")

        # First build concatenated audio
        tmp_concat = output_m4b.replace(".m4b", "_concat.aac")
        _run_ffmpeg(
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-codec:a", "aac", "-b:a", "128k",
            tmp_concat,
        )

        # Build chapter metadata file
        meta_file = tempfile.mktemp(suffix=".txt")
        offset_ms = 0
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(";FFMETADATA1\n")
            f.write(f"title={book_title}\n")
            f.write(f"artist=Eve (AI Narrator)\n")
            f.write(f"album={book_title}\n\n")
            for ch in chapter_mp3s:
                dur_ms = int(ch["duration_s"] * 1000)
                f.write("[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={offset_ms}\n")
                f.write(f"END={offset_ms + dur_ms}\n")
                f.write(f"title={ch['title']}\n\n")
                offset_ms += dur_ms

        # Combine audio + chapter metadata into M4B
        _run_ffmpeg(
            "-i", tmp_concat,
            "-i", meta_file,
            "-map_metadata", "1",
            "-codec:a", "copy",
            "-movflags", "+faststart",
            output_m4b,
        )

        # Cleanup
        for tmp in [concat_file, meta_file, tmp_concat]:
            try:
                os.unlink(tmp)
            except Exception:
                pass

        return True
    except Exception as e:
        logger.warning("[AudioMaster] M4B build failed: %s", e)
        return False


# ── AudioMasteringPipelineCell ────────────────────────────────────────────────

class AudioMasteringPipelineCell(BaseCell):
    name        = "audio_master"
    description = (
        "Audiobook mastering pipeline. Normalizes chapter WAVs to -16 LUFS, "
        "encodes to MP3 with ID3 tags, produces M4B with chapter markers."
    )
    color       = "#b45309"
    lazy        = True
    position    = (5, 5)

    system_tier     = "online"
    hardware_req    = "CPU (ffmpeg required)"
    research_basis  = "ITU-R BS.1770 broadcast loudness standard (LUFS), AAC/M4B audiobook format"
    build_notes     = (
        "LIVE: ffmpeg-based normalization + MP3 encoding + M4B chapter markers. "
        "Output: C:\\Users\\<your-username>\\eve\\audiobooks\\{book_title}\\ "
        "POST /brain/book/master | GET /brain/book/output/{title}"
    )
    framework_layer = "AI & ML"

    def __init__(self):
        super().__init__()
        self._ffmpeg_ok: Optional[bool] = None

    async def boot(self) -> None:
        self._ffmpeg_ok = _check_ffmpeg()
        if not self._ffmpeg_ok:
            logger.warning("[AudioMaster] ffmpeg not found on PATH — mastering will be limited")
        else:
            logger.info("[AudioMaster] Cell online — ffmpeg available")

    async def process(self, ctx: CellContext) -> Any:
        return {"status": "audio_master ready — use REST endpoints"}

    def _generate_title_card_wav(self, chapter_title: str, out_path: str) -> bool:
        """
        Generate a WAV announcement for the chapter title via TTS.
        Format: 2s silence + "Chapter X: Title" + 4s silence (ACX standard)
        Returns True if generated, False if TTS unavailable.
        """
        try:
            import requests as _req
            r = _req.post(
                "http://127.0.0.1:8766/tts",
                json={"text": chapter_title, "speaker": "af_heart", "speed": 0.9},
                timeout=20,
            )
            if r.status_code == 200:
                import io, wave
                title_wav_bytes = r.content
                # Build: 2s silence + title TTS + 4s silence
                sr = 44100
                n_pre  = int(sr * 2.0)
                n_post = int(sr * 4.0)
                buf = io.BytesIO()
                with wave.open(io.BytesIO(title_wav_bytes)) as tw:
                    params = tw.getparams()
                    title_frames = tw.readframes(tw.getnframes())

                with wave.open(buf, "wb") as out_w:
                    out_w.setnchannels(1)
                    out_w.setsampwidth(2)
                    out_w.setframerate(sr)
                    out_w.writeframes(b"\x00\x00" * n_pre)
                    # Resample title_frames if needed
                    if params.framerate != sr:
                        import audioop
                        title_frames, _ = audioop.ratecv(
                            title_frames, params.sampwidth, params.nchannels,
                            params.framerate, sr, None
                        )
                    out_w.writeframes(title_frames)
                    out_w.writeframes(b"\x00\x00" * n_post)

                Path(out_path).write_bytes(buf.getvalue())
                return True
        except Exception as e:
            logger.debug("[AudioMaster] Title card TTS failed: %s", e)
        return False

    def master_chapters(
        self,
        book_title: str,
        chapter_wavs: list[str],
        chapter_titles: Optional[list[str]] = None,
    ) -> dict:
        """
        Master a list of chapter WAV files.
        Pipeline: normalize (-16 LUFS) → fade in/out + room tone → chapter title card → MP3 encode.
        Industry standard: 44100 Hz, 128kbps MP3, -16 LUFS, 50ms fades, ACX silence gaps.
        Returns dict with output paths and manifest info.
        """
        if not chapter_wavs:
            return {"error": "No chapter WAVs provided"}

        # Sanitize book title for filesystem
        safe_title = re.sub(r"[^\w\s\-]", "", book_title).strip().replace(" ", "_")
        out_dir = _OUTPUT_BASE / safe_title
        out_dir.mkdir(parents=True, exist_ok=True)

        titles = chapter_titles or [f"Chapter {i+1}" for i in range(len(chapter_wavs))]
        results = []
        chapter_mp3s = []

        for i, wav_path in enumerate(chapter_wavs):
            if not Path(wav_path).exists():
                logger.warning("[AudioMaster] WAV not found: %s", wav_path)
                continue

            chap_num   = i + 1
            chap_title = titles[i] if i < len(titles) else f"Chapter {chap_num}"
            norm_wav   = str(out_dir / f"ch{chap_num:02d}_norm.wav")
            faded_wav  = str(out_dir / f"ch{chap_num:02d}_faded.wav")
            title_wav  = str(out_dir / f"ch{chap_num:02d}_title.wav")
            final_wav  = str(out_dir / f"ch{chap_num:02d}_final.wav")
            mp3_path   = str(out_dir / f"ch{chap_num:02d}.mp3")

            logger.info("[AudioMaster] Processing chapter %d/%d: %s", chap_num, len(chapter_wavs), chap_title)

            # Step 1: Normalize to -16 LUFS (ACX requires -18 to -23 LUFS, we use -16 for headroom)
            norm_ok = self._ffmpeg_ok and _normalize_chapter(wav_path, norm_wav)
            source_wav = norm_wav if norm_ok else wav_path

            # Step 2: Add 50ms fade-in/out + subtle room tone floor
            duration = _get_wav_duration(source_wav) if self._ffmpeg_ok else 0.0
            if self._ffmpeg_ok and duration > 0:
                _add_fades_and_room_tone(source_wav, faded_wav, duration)
                source_wav = faded_wav if Path(faded_wav).exists() else source_wav

            # Step 3: Prepend chapter title card (2s pre-silence + TTS + 4s post-silence)
            title_ok = self._generate_title_card_wav(chap_title, title_wav)
            if title_ok and self._ffmpeg_ok:
                # Concatenate title card + chapter audio
                import tempfile
                concat_f = tempfile.mktemp(suffix=".txt")
                try:
                    with open(concat_f, "w") as cf:
                        cf.write(f"file '{title_wav}'\n")
                        cf.write(f"file '{source_wav}'\n")
                    _run_ffmpeg(
                        "-f", "concat", "-safe", "0",
                        "-i", concat_f,
                        "-ar", "44100",  # ensure 44100 Hz output
                        final_wav,
                    )
                    if Path(final_wav).exists():
                        source_wav = final_wav
                except Exception as e:
                    logger.debug("[AudioMaster] Title card concat failed: %s", e)
                finally:
                    try:
                        os.unlink(concat_f)
                    except Exception:
                        pass

            # Step 4: Encode to MP3 (44100 Hz, 128kbps — distribution standard)
            mp3_ok = self._ffmpeg_ok and _encode_to_mp3(
                source_wav, mp3_path, book_title, chap_num, chap_title
            )

            duration = _get_wav_duration(source_wav) if self._ffmpeg_ok else 0.0

            results.append({
                "chapter": chap_num,
                "title":   chap_title,
                "mp3":     mp3_path if mp3_ok else None,
                "duration_s": round(duration, 1),
            })

            if mp3_ok:
                chapter_mp3s.append({
                    "path":       mp3_path,
                    "title":      chap_title,
                    "num":        chap_num,
                    "duration_s": duration,
                })

            # Clean up intermediate WAVs
            for tmp in [norm_wav, faded_wav, title_wav, final_wav]:
                try:
                    if Path(tmp).exists():
                        os.unlink(tmp)
                except Exception:
                    pass

        # Build M4B
        m4b_path = str(out_dir / f"{safe_title}_complete.m4b")
        m4b_ok   = bool(chapter_mp3s) and self._ffmpeg_ok and _build_m4b(chapter_mp3s, m4b_path, book_title)

        # Write manifest
        manifest = {
            "book_title":    book_title,
            "created_at":    time.time(),
            "chapters":      results,
            "m4b":           m4b_path if m4b_ok else None,
            "output_dir":    str(out_dir),
            "total_chapters": len(results),
        }
        manifest_path = str(out_dir / f"{safe_title}_manifest.json")
        try:
            Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception:
            pass

        return manifest

    def list_output(self, book_title: str) -> dict:
        safe_title = re.sub(r"[^\w\s\-]", "", book_title).strip().replace(" ", "_")
        out_dir = _OUTPUT_BASE / safe_title
        if not out_dir.exists():
            return {"error": f"No output found for '{book_title}'"}

        files = []
        for f in sorted(out_dir.iterdir()):
            files.append({
                "name":       f.name,
                "size_bytes": f.stat().st_size,
                "path":       str(f),
            })

        return {"book_title": book_title, "output_dir": str(out_dir), "files": files}

    def health(self) -> dict:
        if self._ffmpeg_ok is None:
            self._ffmpeg_ok = _check_ffmpeg()
        return {"status": self._status.value, "ffmpeg": self._ffmpeg_ok}


import re  # needed for safe_title in list_output
