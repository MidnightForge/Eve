# Eve Creative Studio - Backend Routes
import json, time, random, uuid, requests, subprocess, os, threading
import urllib3; urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Native creative engine — Adobe replacement suite
try:
    from eve_creative_engine import get_engine as _get_creative
    _CREATIVE_OK = True
except Exception:
    _CREATIVE_OK = False

router = APIRouter(prefix="/studio", tags=["creative_studio"])
OUTPUT_DIR = Path("F:/output")
FNE_INPUT  = Path("F:/input")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FNE_INPUT.mkdir(parents=True, exist_ok=True)

# Flux reads descriptive visual tags, NOT instructions — keep this concise and tag-based.
# Applied to every generation regardless of subject.
FPS = 16.2

# ── NSFW video prompt additions ───────────────────────────────────────────────
# These are appended to the positive prompt when content_mode == "nsfw"
NSFW_VIDEO_POSITIVE = (
    "fully nude, bare skin, unclothed, explicit, uncensored adult content, "
    "intimate scene, sensual movement, anatomically correct, high detail"
)
NSFW_VIDEO_NEGATIVE = (
    "clothed, dressed, covered, censored, blur, mosaic, pixelated, watermark, "
    "children, minors, underage"
)

# ── Video resolution presets (WAN2.2 optimized) ───────────────────────────────
VIDEO_RESOLUTIONS = {
    "landscape": (832, 480),
    "portrait":  (480, 832),
    "square":    (512, 512),
}

# ── Frame interpolation state ─────────────────────────────────────────────────
# prompt_id -> {"status": "running"|"done", "filename": str}
_interp_jobs: dict = {}

STYLE_PREFIXES = {
    "cinematic":      "Cinematic 4K shot, dramatic lighting, film grain, shallow DOF, ",
    "anime":          "Anime style, vibrant colors, expressive, detailed illustration, uncensored, ",
    "photorealistic": "Ultra-photorealistic, 8K, DSLR, natural lighting, sharp focus, ",
    "painterly":      "Oil painting style, textured brushstrokes, impressionist, vivid, ",
    "neon":           "Neon cyberpunk aesthetic, glowing lights, rain-slicked streets, dark, ",
    "horror":         "Dark horror aesthetic, unsettling atmosphere, dramatic shadows, ",
    "documentary":    "Documentary style, natural lighting, candid, realistic, ",
    "abstract":       "Abstract art, geometric shapes, bold colors, artistic, ",
    "cyberpunk":      "Cyberpunk aesthetic, neon lights, futuristic, high tech, ",
}
MOOD_SUFFIXES = {
    "neutral":    "",
    "dark":       ", moody dark atmosphere, deep shadows",
    "bright":     ", bright airy atmosphere, golden hour light",
    "mysterious": ", mysterious foggy atmosphere, enigmatic",
    "epic":       ", epic dramatic scale, awe-inspiring grandeur",
    "dread":      ", dread-filled, oppressive, slow-burn horror",
    "vibrant":    ", vibrant saturated colors, high energy",
    "melancholy": ", melancholy blue tones, somber, reflective",
    "peaceful":   ", peaceful, serene, calming atmosphere",
    "intense":    ", intense energy, dynamic composition",
}
MOTION_TAGS = {
    "smooth":  "smooth camera movement, ",
    "dynamic": "dynamic fast camera, handheld energy, ",
    "slow":    "slow motion, graceful movement, ",
    "static":  "static locked camera, still background, ",
    "orbit":   "orbital camera movement around subject, ",
}

# Track which prompt_ids were submitted to FNE
_fne_jobs: set = set()

# ── Async image job registry ──────────────────────────────────────────────────
# prompt_id -> {"status": "pending"|"running"|"done"|"error", ...}
_image_jobs: dict = {}


class SubmitRequest(BaseModel):
    mode: str = "image"
    content_mode: str = "safe"          # "safe" | "nsfw"
    positive_prompt: str
    negative_prompt: str = ""
    style: str = "cinematic"
    mood: str = "neutral"
    motion: str = "smooth"
    width: int = 1024
    height: int = 1024
    duration_seconds: int = 4
    image_filename: str = ""
    image_filenames: List[str] = []     # multiple references
    steps: int = 20                     # sampling steps (5=fast, 20=quality, 30=best)
    video_resolution: str = "landscape" # landscape | portrait | square
    interpolate: bool = True            # run ffmpeg frame interpolation to 30fps
    # ── Native Creative Engine post-processing ──────────────────────────────
    post_grade: bool = True             # auto color-grade output to match mood
    post_sharpen: bool = False          # apply sharpening pass
    post_upscale: bool = False          # Pillow Lanczos upscale 2x (fast, no VRAM)
    face_analyze: bool = False          # run mediapipe face tracking on result


def _resolve_refs(req: SubmitRequest) -> list[str]:
    """Return unified list of reference filenames from either field."""
    refs = list(req.image_filenames)
    if req.image_filename and req.image_filename not in refs:
        refs.insert(0, req.image_filename)
    return refs


def _build_fne_video_workflow(req: SubmitRequest) -> tuple:
    """Build a WAN2.2 FNE workflow for video generation."""
    motion_tag  = MOTION_TAGS.get(req.motion, "")
    style_pre   = STYLE_PREFIXES.get(req.style, "")
    mood_suf    = MOOD_SUFFIXES.get(req.mood, "")
    refs = _resolve_refs(req)
    is_nsfw = req.content_mode == "nsfw"

    # Fetch Eve's learned feedback context for this style/mood/prompt
    fb = _fetch_feedback_context(req.positive_prompt, req.style, req.mood, "video")

    extra_note = ""
    if len(refs) > 1:
        extras = ", ".join(r.split("/")[-1].split("\\")[-1] for r in refs[1:])
        extra_note = f", also referencing visual elements from: {extras}"

    # NSFW suffix injected into positive prompt (equivalent of image path)
    nsfw_pos = f", {NSFW_VIDEO_POSITIVE}" if is_nsfw else ""
    full_prompt = f"{motion_tag}{style_pre}{req.positive_prompt}{mood_suf}{nsfw_pos}{extra_note}"

    # Reinforce positive patterns Eve learned from thumbs-up ratings
    if fb["positive_hints"]:
        full_prompt = f"{full_prompt}, {', '.join(fb['positive_hints'])}"

    # Build negative prompt incorporating Eve's learned avoidances + NSFW guard
    neg_parts = [req.negative_prompt] if req.negative_prompt.strip() else []
    if is_nsfw:
        neg_parts.append(NSFW_VIDEO_NEGATIVE)
    if fb["negative_hints"]:
        neg_parts += fb["negative_hints"]
    combined_negative = ", ".join(neg_parts)

    # Resolution
    vid_w, vid_h = VIDEO_RESOLUTIONS.get(req.video_resolution, (832, 480))

    # Steps: respect explicit request, cap at 50 to avoid OOM on long videos
    steps_val = max(1, min(req.steps, 50))

    # LoRA + guidance strategy:
    # The lightx2v cfg_step_distill LoRA bakes in CFG guidance internally.
    # When using it: guidance_scale must be 1.0 (double-CFG causes artifacts).
    # At ≤8 steps: use LoRA for quality-at-speed.
    # At >8 steps: disable LoRA, use proper CFG for full quality.
    use_lora = steps_val <= 8
    if use_lora:
        lora_scale = 1.0
        guidance   = 1.0           # LoRA handles guidance internally
        steps_val  = 4             # distill LoRA is optimized for exactly 4 steps
    else:
        lora_scale = 0.0           # disable LoRA (delta=0, no weight modification)
        guidance   = 6.0 if is_nsfw else 5.5

    num_frames_raw = max(1, int(req.duration_seconds * FPS))
    # WAN requires 4k+1 frames
    num_frames = max(1, ((num_frames_raw - 1 + 2) // 4) * 4 + 1)
    seed   = random.randint(0, 2**32 - 1)
    prefix = f"EveStudio_{uuid.uuid4().hex[:8]}"

    image_filename = refs[0] if refs else None
    # T2V when no reference image, I2V otherwise — selects correct distill LoRA
    lora_mode = "t2v" if not image_filename else "i2v"

    wf = {
        "1": {"class_type": "WanVideoVAELoader",   "inputs": {"device": "cuda", "dtype": "bf16"}},
        "2": {"class_type": "WanVideoTextEncode",  "inputs": {
            "text": full_prompt, "negative_text": combined_negative,
            "max_length": 512, "device": "cuda", "dtype": "bf16"}},
        "3": {"class_type": "WanVideoModelLoader", "inputs": {"dtype": "fp16"}},
        "6": {"class_type": "WanVideoSampler",     "inputs": {
            "model": ["3", 0], "positive_embeds": ["2", 0], "negative_embeds": ["2", 1],
            "num_frames": num_frames, "width": vid_w, "height": vid_h,
            "steps": steps_val, "guidance_scale": guidance,
            "seed": seed, "lora_scale": lora_scale,
            "lora_mode": lora_mode}},
        "7": {"class_type": "WanVideoVAEDecode",   "inputs": {"vae": ["1", 0], "latent": ["6", 0]}},
        "8": {"class_type": "VHS_VideoCombine",    "inputs": {
            "images": ["7", 0], "frame_rate": FPS, "loop_count": 0,
            "filename_prefix": prefix, "format": "video/h264-mp4",
            "pix_fmt": "yuv420p", "crf": 19, "save_output": True, "pingpong": False}},
    }
    if image_filename:
        wf["4"] = {"class_type": "LoadImage",                  "inputs": {"image": image_filename}}
        wf["5"] = {"class_type": "WanVideoImageToVideoEncode", "inputs": {"vae": ["1", 0], "image": ["4", 0]}}
        wf["6"]["inputs"]["image_latent"] = ["5", 0]

    print(f"[Studio] Video: {vid_w}×{vid_h}, {steps_val} steps, cfg={guidance:.1f}, "
          f"lora={'ON (' + lora_mode + ')' if use_lora else 'OFF'}, NSFW={is_nsfw}, frames={num_frames}")
    return wf, full_prompt


def _run_rife_interp(prompt_id: str, src_path: Path) -> None:
    """
    Background thread: interpolate video from ~16fps to 30fps using ffmpeg minterpolate.
    Updates _interp_jobs[prompt_id] when done.
    Falls back to original file on any error.
    """
    dst_path = src_path.with_stem(src_path.stem + "_30fps")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src_path),
            "-vf", "minterpolate=fps=30:mi_mode=blend",
            "-c:v", "libx264", "-crf", "17", "-preset", "fast",
            str(dst_path),
        ]
        r = subprocess.run(cmd, timeout=600, capture_output=True)
        if r.returncode == 0 and dst_path.exists() and dst_path.stat().st_size > 1024:
            _interp_jobs[prompt_id] = {"status": "done", "filename": dst_path.name}
            print(f"[RIFE] Interpolated -> {dst_path.name}")
        else:
            stderr = r.stderr.decode(errors="replace")[:300]
            print(f"[RIFE] ffmpeg failed (rc={r.returncode}): {stderr}")
            _interp_jobs[prompt_id] = {"status": "done", "filename": src_path.name}
    except Exception as e:
        print(f"[RIFE] Error: {e}")
        _interp_jobs[prompt_id] = {"status": "done", "filename": src_path.name}



def _studio_vram_free_gb() -> float:
    """Return free GPU VRAM in GB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=5, text=True
        ).strip()
        return float(out.split()[0]) / 1024.0
    except Exception:
        return 99.0


def _vllm_is_running() -> bool:
    """True if vLLM is reachable on port 8099."""
    try:
        requests.get("http://127.0.0.1:8099/health", timeout=2)
        return True
    except Exception:
        return False


def _studio_kill_vllm() -> bool:
    """
    Kill vLLM in WSL2 to free ~19 GB VRAM for image/video generation.
    Fast-switch: no-op if vLLM is already down AND VRAM is already free.
    Polls actual VRAM until at least 15 GB is free (max 120s).
    """
    vram_free = _studio_vram_free_gb()
    if not _vllm_is_running() and vram_free >= 15.0:
        return True   # already free, nothing to do

    # Kill the vLLM process
    try:
        subprocess.run(
            ["wsl", "-d", "Ubuntu", "bash", "-c",
             "pkill -9 -f 'vllm.entrypoints' 2>/dev/null; pkill -9 -f 'run_vllm' 2>/dev/null; true"],
            timeout=10, capture_output=True
        )
    except Exception:
        pass

    # Wait up to 120s for CUDA to actually release VRAM (not just the process to exit)
    print("[Studio] Waiting for VRAM to free (need >=15 GB free)...")
    for _ in range(120):
        time.sleep(1)
        free_gb = _studio_vram_free_gb()
        if free_gb >= 15.0:
            print(f"[Studio] VRAM free: {free_gb:.1f} GB — ready for Flux")
            return True
    print(f"[Studio] WARNING: VRAM still only {_studio_vram_free_gb():.1f} GB free after 120s")
    return True


def _studio_restart_vllm() -> None:
    """
    Restart vLLM in WSL2 as a background non-blocking process.
    Fast-switch: no-op if vLLM is already up.
    """
    if _vllm_is_running():
        return   # already running
    try:
        subprocess.Popen(
            ["wsl", "-d", "Ubuntu", "bash", "-c",
             "nohup /home/<your-username>/run_vllm.sh > ~/vllm_out.log 2>&1 &"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


# ── Mood → color grade mapping ────────────────────────────────────────────────
_MOOD_GRADE = {
    "dark":       {"brightness": 0.88, "contrast": 1.25, "saturation": 0.80},
    "bright":     {"brightness": 1.12, "contrast": 1.05, "saturation": 1.20},
    "mysterious": {"brightness": 0.92, "contrast": 1.15, "saturation": 0.70},
    "epic":       {"brightness": 0.95, "contrast": 1.30, "saturation": 1.10},
    "dread":      {"brightness": 0.80, "contrast": 1.35, "saturation": 0.55},
    "vibrant":    {"brightness": 1.05, "contrast": 1.10, "saturation": 1.50},
    "melancholy": {"brightness": 0.90, "contrast": 1.00, "saturation": 0.60},
    "peaceful":   {"brightness": 1.08, "contrast": 0.95, "saturation": 1.05},
    "intense":    {"brightness": 0.97, "contrast": 1.40, "saturation": 1.20},
}

# ── Style → Lightroom-style preset mapping ────────────────────────────────────
_STYLE_PRESET = {
    "cinematic":      "cinematic",
    "painterly":      "vivid",
    "horror":         "matte",
    "anime":          "vivid",
    "photorealistic": "portrait",
    "documentary":    "portrait",
    "abstract":       "vivid",
}


def _post_process_image(src: Path, req: "SubmitRequest") -> Path:
    """
    Run native creative engine post-processing on a generated image.
    Returns the (possibly new) output path.
    """
    if not _CREATIVE_OK:
        return src

    engine = _get_creative()
    out = src  # will update as we chain ops

    try:
        # 1. Lightroom-style preset for the style (fast Pillow pass)
        preset = _STYLE_PRESET.get(req.style)
        if req.post_grade and preset:
            graded = out.with_stem(out.stem + "_g")
            result = engine.lightroom.apply_preset(str(out), str(graded), preset)
            if result.get("status") == "ok":
                out = graded

        # 2. Fine-tune per-mood colour grade on top
        if req.post_grade and req.mood in _MOOD_GRADE:
            g = _MOOD_GRADE[req.mood]
            graded2 = out.with_stem(out.stem + "_m")
            result = engine.ps.color_grade(
                str(out), str(graded2),
                brightness=g["brightness"],
                contrast=g["contrast"],
                saturation=g["saturation"],
            )
            if result.get("status") == "ok":
                out = graded2

        # 3. Optional sharpening pass
        if req.post_sharpen:
            sharp = out.with_stem(out.stem + "_s")
            result = engine.ps.apply_filter(str(out), str(sharp), "sharpen", 1.0)
            if result.get("status") == "ok":
                out = sharp

        # 4. Optional fast 2x upscale via Lanczos (no VRAM needed)
        if req.post_upscale:
            from PIL import Image as _PILImg
            img = _PILImg.open(str(out))
            up = out.with_stem(out.stem + "_2x")
            img_up = img.resize((img.width * 2, img.height * 2), _PILImg.LANCZOS)
            img_up.save(str(up))
            out = up

    except Exception as e:
        print(f"[Studio] Post-process warning: {e}")

    return out


def _face_analyze_image(path: Path) -> dict:
    """Run mediapipe face analysis on the output — returns expression data or {}."""
    if not _CREATIVE_OK:
        return {}
    try:
        engine = _get_creative()
        return engine.tracking.analyze_face(str(path))
    except Exception:
        return {}


def _run_image_job(prompt_id: str, req: "SubmitRequest"):
    """Background thread: free VRAM, generate image, copy to output dir, store result."""
    import shutil, threading
    try:
        _image_jobs[prompt_id] = {"status": "running"}

        # Free VRAM from vLLM before Flux loads
        _studio_kill_vllm()

        # Build full prompt with style/mood/feedback
        style_pre = STYLE_PREFIXES.get(req.style, "")
        mood_suf  = MOOD_SUFFIXES.get(req.mood, "")
        fb = _fetch_feedback_context(req.positive_prompt, req.style, req.mood, "image")
        nsfw_suf = ", nude, naked, fully unclothed, bare skin, explicit" if req.content_mode == "nsfw" else ""
        full_prompt = f"{style_pre}{req.positive_prompt}{mood_suf}{nsfw_suf}"
        if fb["positive_hints"]:
            full_prompt = f"{full_prompt}, {', '.join(fb['positive_hints'])}"
        neg_parts = [req.negative_prompt] if req.negative_prompt.strip() else []
        if fb["negative_hints"]:
            neg_parts += fb["negative_hints"]
        combined_neg = ", ".join(neg_parts)

        from image_engine import generate as _img_generate
        result = _img_generate(
            prompt          = full_prompt,
            negative_prompt = combined_neg,
            width           = req.width,
            height          = req.height,
        )
        if "error" in result:
            _image_jobs[prompt_id] = {"status": "error", "detail": result["error"]}
            return

        # Copy generated file to OUTPUT_DIR so /studio/output/{filename} can serve it
        src = Path(result["path"])
        dst = OUTPUT_DIR / src.name
        if src != dst:
            shutil.copy2(str(src), str(dst))

        # ── Native creative engine post-processing ───────────────────────────
        processed = _post_process_image(dst, req)
        if processed != dst:
            # Move processed version to OUTPUT_DIR with its own name
            final_dst = OUTPUT_DIR / processed.name
            if processed != final_dst:
                shutil.copy2(str(processed), str(final_dst))
            out_name = final_dst.name
            print(f"[Studio] Post-processed: {dst.name} -> {out_name}")
        else:
            out_name = dst.name

        # Optional face analysis (non-blocking, stored as metadata)
        face_data = _face_analyze_image(OUTPUT_DIR / out_name) if req.face_analyze else {}

        _image_jobs[prompt_id] = {
            "status":   "done",
            "type":     "image",
            "filename": out_name,
            "original": src.name,
            "seed":     result.get("seed"),
            "prompt":   full_prompt,
            "post_processed": processed != dst,
            "face": face_data if face_data.get("faces", 0) > 0 else None,
        }
    except Exception as e:
        _image_jobs[prompt_id] = {"status": "error", "detail": str(e)}
    finally:
        # Unload Flux pipeline to free VRAM/RAM so vLLM can restart cleanly
        # and so the next image job reloads with fp8 quantization
        try:
            import image_engine as _ie
            _ie.unload()
        except Exception:
            pass
        # Restart vLLM in background so chat is ready again after ~60s
        _studio_restart_vllm()


def _submit_fne(workflow: dict) -> str:
    """Submit workflow to the FNE (Forge Node Engine) queue."""
    from forge_engine import get_job_queue
    client_id = str(uuid.uuid4())
    return get_job_queue().submit(workflow, client_id)


# ── Non-blocking submit (returns prompt_id immediately) ──────────────────────
@router.post("/submit")
def studio_submit(req: SubmitRequest):
    try:
        if req.mode == "video":
            wf, prompt_used = _build_fne_video_workflow(req)
            prompt_id = _submit_fne(wf)
            _fne_jobs.add(prompt_id)
            return {"prompt_id": prompt_id, "prompt_used": prompt_used, "mode": req.mode}
        else:
            # Image generation — async to avoid blocking HTTP (Flux load takes 2-5 min)
            import threading as _threading
            prompt_id = str(uuid.uuid4())
            _image_jobs[prompt_id] = {"status": "pending"}
            _threading.Thread(
                target=_run_image_job, args=(prompt_id, req), daemon=True
            ).start()
            return {"prompt_id": prompt_id, "mode": req.mode}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Poll job status ───────────────────────────────────────────────────────────
@router.get("/job/{prompt_id}")
def studio_job_status(prompt_id: str):
    return _fne_job_status(prompt_id)


def _fne_job_status(prompt_id: str) -> dict:
    # Image jobs are tracked in _image_jobs (not FNE)
    if prompt_id in _image_jobs:
        job = _image_jobs[prompt_id]
        if job["status"] == "done":
            return {"status": "done", "type": "image",
                    "filename": job["filename"], "seed": job.get("seed")}
        if job["status"] == "error":
            return {"status": "error", "detail": job.get("detail", "unknown")}
        return {"status": "running"}   # pending or running

    try:
        from forge_engine import get_job_queue
        q = get_job_queue()
        hist = q.get_history(prompt_id)
        entry = hist.get(prompt_id)
        if not entry:
            qs = q.get_queue_status()
            running_ids = [j[1] for j in qs.get("queue_running", [])]
            pending_ids = [j[1] for j in qs.get("queue_pending", [])]
            if prompt_id in running_ids:
                return {"status": "running"}
            if prompt_id in pending_ids:
                return {"status": "pending"}
            return {"status": "unknown"}

        status = entry.get("status", {})
        if not status.get("completed"):
            return {"status": "running"}
        if status.get("status_str") == "error":
            messages = status.get("messages") or [["error", "unknown"]]
            err = messages[0][1] if messages else "unknown"
            return {"status": "error", "detail": str(err)}

        for node_out in entry.get("outputs", {}).values():
            for item in node_out.get("gifs", []):
                fname = item.get("filename", "") if isinstance(item, dict) else str(item)
                if fname:
                    # Check if we should interpolate (only for FNE video jobs)
                    if prompt_id in _fne_jobs:
                        interp = _interp_jobs.get(prompt_id)
                        if interp is None:
                            # First time done — launch interpolation thread
                            src = OUTPUT_DIR / fname
                            if src.exists():
                                _interp_jobs[prompt_id] = {"status": "running", "filename": fname}
                                threading.Thread(
                                    target=_run_rife_interp, args=(prompt_id, src), daemon=True
                                ).start()
                                return {"status": "processing", "detail": "Smoothing to 30fps…"}
                        elif interp["status"] == "running":
                            return {"status": "processing", "detail": "Smoothing to 30fps…"}
                        else:
                            return {"status": "done", "type": "video",
                                    "filename": interp["filename"]}
                    return {"status": "done", "type": "video", "filename": fname}
        return {"status": "done", "type": "unknown"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ── Queue status (FNE only) ────────────────────────────────────────────────────
@router.get("/queue")
def studio_queue():
    running, pending = 0, 0
    try:
        from forge_engine import get_job_queue
        fq = get_job_queue().get_queue_status()
        running += len(fq.get("queue_running", []))
        pending += len(fq.get("queue_pending", []))
    except Exception:
        pass
    return {"running": running, "pending": pending}


# ── Upload reference image/video ─────────────────────────────────────────────
@router.post("/upload")
def studio_upload(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        # Save to FNE input dir for video reference and image_engine references
        fne_dest = FNE_INPUT / file.filename
        fne_dest.write_bytes(content)
        return {"name": file.filename, "subfolder": "", "type": "input"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Context image search ──────────────────────────────────────────────────────
CONTEXT_CACHE = FNE_INPUT / "context_cache"
CONTEXT_CACHE.mkdir(parents=True, exist_ok=True)

class ImageSearchRequest(BaseModel):
    query: str
    count: int = 6

@router.post("/image-search")
def studio_image_search(req: ImageSearchRequest):
    """
    Search the web for images matching query, download and cache them locally,
    then return filenames that can be used as references in generation.
    """
    import hashlib
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise HTTPException(500, "duckduckgo_search not installed")

    results = []
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.images(
                req.query,
                max_results=req.count * 3,   # fetch extra, many will 404
                safesearch="off",
            ))
    except Exception as e:
        raise HTTPException(502, f"Search failed: {e}")

    import hashlib, shutil
    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://duckduckgo.com/",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    }

    def _try_download(url: str, fname: str) -> bool:
        """Try to download url → CONTEXT_CACHE/fname. Returns True on success."""
        cached = CONTEXT_CACHE / fname
        if cached.exists():
            return True
        try:
            r = requests.get(url, timeout=10, headers=_HEADERS, verify=False)
            ct = r.headers.get("content-type", "")
            if r.status_code != 200 or "image" not in ct:
                return False
            data = r.content
            if len(data) < 1024:
                return False
            cached.write_bytes(data)
            # Mirror into FNE input dir for reference pipeline
            dest = FNE_INPUT / fname
            if not dest.exists():
                shutil.copy2(str(cached), str(dest))
            return True
        except Exception:
            return False

    for img in hits:
        if len(results) >= req.count:
            break
        img_url = img.get("image", "")
        thumb   = img.get("thumbnail", "")
        title   = img.get("title", "")
        if not img_url and not thumb:
            continue

        # Build cache filename from the primary URL (or thumb if no main)
        key_url = img_url or thumb
        ext = key_url.split("?")[0].rsplit(".", 1)[-1].lower()
        if ext not in ("jpg", "jpeg", "png", "webp", "gif"):
            ext = "jpg"
        fname = "ctx_" + hashlib.md5(key_url.encode()).hexdigest()[:16] + "." + ext

        # Strategy 1: try the full-resolution image URL
        ok = _try_download(img_url, fname) if img_url else False
        # Strategy 2: fall back to DDG/Bing thumbnail CDN (always accessible)
        if not ok and thumb:
            thumb_fname = "ctx_" + hashlib.md5(thumb.encode()).hexdigest()[:16] + ".jpg"
            ok = _try_download(thumb, thumb_fname)
            if ok:
                fname = thumb_fname
        if not ok:
            continue

        results.append({
            "filename": fname,
            "title":    title[:80],
            "thumb":    thumb or img_url,
        })

    if not results:
        raise HTTPException(404, "No images found — try rephrasing your search")
    return {"results": results}


@router.get("/context-image/{filename}")
def studio_context_image(filename: str):
    """Serve a cached context search image as a thumbnail."""
    p = CONTEXT_CACHE / filename
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(str(p))


# ── List recent outputs ───────────────────────────────────────────────────────
@router.get("/outputs")
def studio_outputs():
    files = []
    for f in sorted(OUTPUT_DIR.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
        files.append({"name": f.name, "size": f.stat().st_size, "type": "video" if f.suffix == ".mp4" else "image"})
    return {"files": files}


# ── Serve output file ─────────────────────────────────────────────────────────
@router.get("/output/{filename}")
def studio_get_output(filename: str):
    p = OUTPUT_DIR / filename
    if not p.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(p))


# ── Rating system ─────────────────────────────────────────────────────────────
RATINGS_FILE = OUTPUT_DIR / "ratings.jsonl"
FACTORY_FEEDBACK = Path("C:/Users/<your-username>/eve/self-improving-llm-factory/human_feedback_visual.jsonl")
MEMORY_URL        = "http://localhost:8767/save"
MEMORY_SEARCH_URL = "http://127.0.0.1:8767/search"


def _fetch_feedback_context(prompt: str, style: str, mood: str, media_type: str) -> dict:
    """
    Query Eve's memory for past visual feedback relevant to this generation.
    Returns:
      positive_hints  – list of terms to reinforce (from thumbs-up memories)
      negative_hints  – list of terms to avoid   (from thumbs-down memories)
    """
    positive_hints, negative_hints = [], []
    try:
        query = (
            f"visual_feedback {media_type} generation prompt: {prompt} "
            f"style: {style} mood: {mood}"
        )
        resp = requests.post(
            MEMORY_SEARCH_URL,
            json={"query": query, "top_k": 8},
            timeout=3,
        )
        if resp.status_code != 200:
            return {"positive_hints": [], "negative_hints": []}

        results = resp.json().get("results", [])
        for r in results:
            eve_text = r.get("eve_response", "").lower()
            user_text = r.get("user_input", "").lower()
            # Only process memories tagged as visual_feedback session
            if "visual_feedback" not in r.get("session_id", "") and "visual_feedback" not in user_text:
                continue
            if "thumbs-up" in user_text or "rated it thumbs-up" in user_text or "lean into" in eve_text:
                # Positive memory — extract what worked
                if style and style in user_text:
                    positive_hints.append(f"{style} style")
                if mood and mood in user_text:
                    positive_hints.append(f"{mood} mood")
                positive_hints.append("high quality")
            elif "thumbs-down" in user_text or "rated it thumbs-down" in user_text or "avoid" in eve_text:
                # Negative memory — extract what to avoid
                # Pull issue terms from the user message (after "Issues noted:")
                for marker in ["issues noted:", "issues:", "problems:"]:
                    if marker in user_text:
                        issue_part = user_text.split(marker, 1)[1].strip()
                        # Extract comma-separated terms
                        terms = [t.strip() for t in issue_part.split(",") if 2 < len(t.strip()) < 60]
                        negative_hints.extend(terms[:4])
                # Also pull from eve's response
                for marker in ["avoid", "problems:"]:
                    if marker in eve_text:
                        after = eve_text.split(marker, 1)[1]
                        snippet = after[:120].split(".")[0]
                        if snippet.strip():
                            negative_hints.append(snippet.strip())
    except Exception:
        pass  # memory service offline — generate without context

    # Deduplicate
    positive_hints = list(dict.fromkeys(h for h in positive_hints if h))[:4]
    negative_hints = list(dict.fromkeys(h for h in negative_hints if h))[:6]
    return {"positive_hints": positive_hints, "negative_hints": negative_hints}


class RateRequest(BaseModel):
    filename: str
    type: str = "image"       # "image" or "video"
    rating: int               # 1 = thumbs up, 0 = thumbs down
    comment: str = ""
    prompt: str = ""
    style: str = ""
    mood: str = ""


@router.post("/rate")
def studio_rate(req: RateRequest):
    ts = datetime.now(timezone.utc).isoformat()
    entry = {
        "filename": req.filename,
        "type": req.type,
        "rating": req.rating,
        "comment": req.comment,
        "prompt": req.prompt,
        "style": req.style,
        "mood": req.mood,
        "timestamp": ts,
    }

    # Persist to ratings log
    with open(RATINGS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    # Write to factory human feedback file (all ratings)
    try:
        FACTORY_FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
        with open(FACTORY_FEEDBACK, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

    # Push feedback into Eve's memory so she learns from both positive and negative results
    try:
        media = 'an image' if req.type == 'image' else 'a video'
        ctx = f'prompt: "{req.prompt}"' + (f', style={req.style}' if req.style else '') + (f', mood={req.mood}' if req.mood else '')
        if req.rating == 1:
            user_msg = (
                f"I generated {media} ({ctx}) and rated it thumbs-up — great result."
            )
            eve_ack = (
                f"Excellent. I'm logging what worked for \"{req.prompt[:80]}\" "
                f"({req.type}, {req.style or 'default'} style, {req.mood or 'default'} mood). "
                f"I'll lean into these choices for future {req.type} generations."
            )
        else:
            comment_part = f" Issues noted: {req.comment}" if req.comment.strip() else ""
            user_msg = (
                f"I generated {media} ({ctx}) and rated it thumbs-down.{comment_part}"
            )
            eve_ack = (
                f"Understood, Forge. The prompt \"{req.prompt[:80]}\" had problems"
                + (f": {req.comment}" if req.comment.strip() else "")
                + f". I'll avoid these issues in future {req.type} generations — "
                f"adjusting prompt strategy, composition, and quality guidance."
            )
        requests.post(
            MEMORY_URL,
            json={"user_input": user_msg, "eve_response": eve_ack, "session_id": "visual_feedback"},
            timeout=5,
        )
    except Exception:
        pass  # memory service offline — rating still saved locally

    label = "👍 approved" if req.rating == 1 else f"👎 flagged — {req.comment[:60]}"
    return {"status": "saved", "label": label}


# ═══════════════════════════════════════════════════════════════════════════════
#  NATIVE CREATIVE ENGINE — Studio Integration Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class PostProcessRequest(BaseModel):
    filename: str                        # existing output file to reprocess
    grade_preset: str = "cinematic"     # vivid | matte | cinematic | bw | portrait | landscape
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    sharpen: bool = False
    upscale_2x: bool = False
    filter: Optional[str] = None        # blur | sharpen | edge | emboss | grayscale | invert


@router.post("/post-process")
def studio_post_process(req: PostProcessRequest):
    """
    Apply native creative engine post-processing to any existing studio output.
    Returns the new filename in OUTPUT_DIR.
    """
    if not _CREATIVE_OK:
        raise HTTPException(503, "Creative engine not available")
    src = OUTPUT_DIR / req.filename
    if not src.exists():
        raise HTTPException(404, f"File not found: {req.filename}")

    engine = _get_creative()
    stem = src.stem + "_pp"
    out = OUTPUT_DIR / (stem + src.suffix)
    chain = str(src)

    try:
        # Lightroom preset
        tmp1 = str(OUTPUT_DIR / (stem + "_1.png"))
        r = engine.lightroom.apply_preset(chain, tmp1, req.grade_preset)
        if r.get("status") == "ok":
            chain = tmp1

        # Fine color grade
        tmp2 = str(OUTPUT_DIR / (stem + "_2.png"))
        r = engine.ps.color_grade(chain, tmp2,
            brightness=req.brightness, contrast=req.contrast, saturation=req.saturation)
        if r.get("status") == "ok":
            chain = tmp2

        # Optional filter
        if req.filter:
            tmp3 = str(OUTPUT_DIR / (stem + "_3.png"))
            r = engine.ps.apply_filter(chain, tmp3, req.filter)
            if r.get("status") == "ok":
                chain = tmp3

        # Optional 2x upscale
        if req.upscale_2x:
            from PIL import Image as _PI
            img = _PI.open(chain)
            img_up = img.resize((img.width * 2, img.height * 2), _PI.LANCZOS)
            tmp4 = str(OUTPUT_DIR / (stem + "_4x.png"))
            img_up.save(tmp4)
            chain = tmp4

        # Copy final result
        import shutil
        shutil.copy2(chain, str(out))
        # Clean up temps
        for t in [stem + "_1.png", stem + "_2.png", stem + "_3.png", stem + "_4x.png"]:
            p = OUTPUT_DIR / t
            if p.exists() and str(p) != str(out):
                try:
                    p.unlink()
                except Exception:
                    pass

        return {"status": "ok", "filename": out.name,
                "original": req.filename}
    except Exception as e:
        raise HTTPException(500, str(e))


class BatchExportRequest(BaseModel):
    filenames: List[str]                 # output filenames to export
    format: str = "png"                  # png | jpg | webp
    quality: int = 92
    grade_preset: Optional[str] = None  # apply preset to all
    resize_w: Optional[int] = None
    resize_h: Optional[int] = None


@router.post("/batch-export")
def studio_batch_export(req: BatchExportRequest):
    """
    Batch export studio outputs — convert format, resize, grade all at once.
    Results go to OUTPUT_DIR/batch_export_{timestamp}/
    """
    if not _CREATIVE_OK:
        raise HTTPException(503, "Creative engine not available")
    engine = _get_creative()
    ts = int(time.time())
    export_dir = OUTPUT_DIR / f"batch_export_{ts}"
    export_dir.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []
    for fname in req.filenames:
        src = OUTPUT_DIR / fname
        if not src.exists():
            errors.append({"file": fname, "error": "not found"})
            continue
        try:
            from PIL import Image as _PI
            img = _PI.open(str(src)).convert("RGB")
            if req.resize_w and req.resize_h:
                img = img.resize((req.resize_w, req.resize_h), _PI.LANCZOS)
            out_name = Path(fname).stem + f"_export.{req.format}"
            out_path = export_dir / out_name
            save_kw = {"quality": req.quality} if req.format in ("jpg", "jpeg", "webp") else {}
            img.save(str(out_path), **save_kw)
            # Optional grade
            if req.grade_preset:
                graded = str(out_path.with_stem(out_path.stem + "_g"))
                engine.lightroom.apply_preset(str(out_path), graded, req.grade_preset)
                out_name = Path(graded).name
            results.append(out_name)
        except Exception as e:
            errors.append({"file": fname, "error": str(e)})

    return {
        "status": "ok",
        "export_dir": str(export_dir),
        "exported": len(results),
        "errors": errors,
        "files": results,
    }


class PDFPortfolioRequest(BaseModel):
    filenames: List[str]                 # images from OUTPUT_DIR to include
    title: str = "Lotus Forge Portfolio"
    pagesize: str = "letter"            # letter | a4


@router.post("/pdf-portfolio")
def studio_pdf_portfolio(req: PDFPortfolioRequest):
    """
    Package generated images into a PDF portfolio using reportlab.
    Perfect for presenting or archiving Forge's generated art.
    """
    if not _CREATIVE_OK:
        raise HTTPException(503, "Creative engine not available")
    engine = _get_creative()
    paths = []
    for fname in req.filenames:
        p = OUTPUT_DIR / fname
        if p.exists():
            paths.append(str(p))
    if not paths:
        raise HTTPException(404, "No valid files found")
    out_name = f"portfolio_{int(time.time())}.pdf"
    out_path = str(OUTPUT_DIR / out_name)
    result = engine.layout.images_to_pdf(paths, out_path, req.pagesize)
    if result.get("status") != "ok":
        raise HTTPException(500, result.get("error", "PDF generation failed"))
    return {"status": "ok", "filename": out_name, "pages": len(paths)}


class AudioVideoRequest(BaseModel):
    video_filename: str                  # mp4 in OUTPUT_DIR
    audio_path: str                      # full path to audio file
    output_filename: Optional[str] = None
    audio_volume: float = 1.0


@router.post("/add-audio")
def studio_add_audio(req: AudioVideoRequest):
    """
    Mix audio into a generated video using native moviepy pipeline.
    No VRAM required — pure CPU operation.
    """
    if not _CREATIVE_OK:
        raise HTTPException(503, "Creative engine not available")
    video_path = OUTPUT_DIR / req.video_filename
    if not video_path.exists():
        raise HTTPException(404, f"Video not found: {req.video_filename}")
    if not Path(req.audio_path).exists():
        raise HTTPException(404, f"Audio not found: {req.audio_path}")

    engine = _get_creative()
    out_name = req.output_filename or (video_path.stem + "_audio.mp4")
    out_path = str(OUTPUT_DIR / out_name)
    result = engine.video.add_audio(str(video_path), req.audio_path, out_path,
                                     req.audio_volume)
    if result.get("status") != "ok":
        raise HTTPException(500, result.get("error", "Audio mix failed"))
    return {"status": "ok", "filename": out_name}


@router.get("/face-analyze/{filename}")
def studio_face_analyze(filename: str):
    """
    Run mediapipe face analysis on any studio output image.
    Returns 478 landmarks + mouth/eye expression metrics.
    Great for VTube rigging and character animation.
    """
    if not _CREATIVE_OK:
        raise HTTPException(503, "Creative engine not available")
    p = OUTPUT_DIR / filename
    if not p.exists():
        raise HTTPException(404, "File not found")
    engine = _get_creative()
    result = engine.tracking.analyze_face(str(p))
    return result


@router.get("/video-info/{filename}")
def studio_video_info(filename: str):
    """Get video metadata (duration, codec, fps, resolution) via ffprobe."""
    if not _CREATIVE_OK:
        raise HTTPException(503, "Creative engine not available")
    p = OUTPUT_DIR / filename
    if not p.exists():
        raise HTTPException(404, "File not found")
    engine = _get_creative()
    return engine.video.get_info(str(p))


class AudioAnalyzeRequest(BaseModel):
    audio_path: str  # full path to audio file


@router.post("/audio-analyze")
def studio_audio_analyze(req: AudioAnalyzeRequest):
    """Analyze audio file — returns BPM, estimated key, duration, RMS level."""
    if not _CREATIVE_OK:
        raise HTTPException(503, "Creative engine not available")
    if not Path(req.audio_path).exists():
        raise HTTPException(404, "Audio file not found")
    engine = _get_creative()
    return engine.audio.analyze(req.audio_path)


@router.get("/creative-status")
def studio_creative_status():
    """Return status of the native creative engine integration."""
    if not _CREATIVE_OK:
        return {"available": False, "reason": "eve_creative_engine import failed"}
    caps = _get_creative().capabilities()
    return {
        "available": True,
        "modules": {
            mod: all(v for k, v in info.items() if k != "features")
            for mod, info in caps.items()
        },
        "post_processing": {
            "auto_grade": "Pillow + rawpy color grading on every image",
            "mood_lut": f"{len(_MOOD_GRADE)} mood presets",
            "style_lut": f"{len(_STYLE_PRESET)} style presets",
        },
    }
