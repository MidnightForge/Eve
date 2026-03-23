"""
image_engine.py
Native Flux Dev fp8 txt2img — no ComfyUI dependency.
Face-swap for Eve portraits via InsightFace inswapper (no ReActor needed).

Model paths (local, no HF download required at runtime):
  Transformer : F:/models/diffusion_models/flux1-dev-fp8.safetensors
  VAE         : F:/models/vae/ae.safetensors
  CLIP-L      : F:/models/text_encoders/clip_l.safetensors
  T5-XXL fp8  : F:/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors
  Face swap   : F:/models/insightface/buffalo_1  +  F:/models/reactor/inswapper_128.onnx
"""

import gc
import os
import random
import threading
import traceback
import uuid
from pathlib import Path

# ── Triton/torchao compatibility stub (triton 3.x removed AttrsDescriptor) ────
# torch._inductor imports AttrsDescriptor from triton — stub it before any
# transformers/diffusers import triggers the chain.
try:
    from dataclasses import dataclass as _dc
    @_dc
    class _AttrsDescriptor:
        divisible_by_16: tuple = ()
        equal_to_1: tuple = ()
    import triton.compiler.compiler as _tcc
    if not hasattr(_tcc, 'AttrsDescriptor'):
        _tcc.AttrsDescriptor = _AttrsDescriptor
    try:
        import triton.backends.compiler as _tbc
        if not hasattr(_tbc, 'AttrsDescriptor'):
            _tbc.AttrsDescriptor = _AttrsDescriptor
    except Exception:
        pass
except Exception:
    pass

import torch
from PIL import Image

# ── Model paths ────────────────────────────────────────────────────────────
_FLUX_CKPT   = Path(r"F:/models/diffusion_models/flux1-dev-fp8.safetensors")
_VAE_CKPT    = Path(r"F:/models/vae/ae.safetensors")
_CLIP_CKPT   = Path(r"F:/models/text_encoders/clip_l.safetensors")
_T5_CKPT     = Path(r"F:/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors")
_CFG_DIR     = Path(r"F:/models/flux-dev-config")   # local config + tokenizer files
_INSIGHT_DIR = Path(r"F:/models/insightface")
_INSWAP_ONNX = Path(r"F:/models/reactor/inswapper_128.onnx")
_EVE_FACE    = Path(r"C:/Users/<your-username>/eve/static/avatar.png")

# Output dir (same location ComfyUI used, so Eve's UI keeps working)
_OUT_DIR = Path(r"C:/Users/<your-username>/eve/static/generated")
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Singleton pipeline ──────────────────────────────────────────────────────
_pipe  = None
_lock  = threading.Lock()
_dtype = torch.bfloat16   # bf16 for safe compute; fp8 weights upcast automatically


def _load():
    global _pipe
    import os
    from diffusers import (
        FluxPipeline, FluxTransformer2DModel, AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
    )
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    print("[ImageEngine] Loading Flux Dev pipeline from local files…")

    # Block all HF network calls — everything is local
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        scheduler    = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(_CFG_DIR / "scheduler"))
        tokenizer    = CLIPTokenizer.from_pretrained(str(_CFG_DIR / "tokenizer"))
        tokenizer_2  = T5TokenizerFast.from_pretrained(str(_CFG_DIR / "tokenizer_2"))

        from safetensors.torch import load_file as _sf_load
        from transformers import CLIPTextConfig, T5Config

        # Instantiate from config only (no weight search), then load local weights
        clip_cfg     = CLIPTextConfig.from_pretrained(str(_CFG_DIR / "text_encoder"))
        text_encoder = CLIPTextModel(clip_cfg).to(_dtype)
        clip_sd      = _sf_load(str(_CLIP_CKPT))
        # Keys already have "text_model." prefix matching CLIPTextModel's state dict — no stripping needed
        m_clip, u_clip = text_encoder.load_state_dict(clip_sd, strict=False)
        print(f"[ImageEngine] CLIP load: {len(m_clip)} missing, {len(u_clip)} unexpected")
        if m_clip:
            print(f"[ImageEngine] CLIP missing keys (first 5): {m_clip[:5]}")

        t5_cfg         = T5Config.from_pretrained(str(_CFG_DIR / "text_encoder_2"))
        text_encoder_2 = T5EncoderModel(t5_cfg).to(_dtype)
        t5_sd          = _sf_load(str(_T5_CKPT))
        m_t5, u_t5 = text_encoder_2.load_state_dict(t5_sd, strict=False)
        print(f"[ImageEngine] T5 load: {len(m_t5)} missing, {len(u_t5)} unexpected")

        # Load transformer directly from safetensors in native fp8 dtype (~12 GB RAM).
        # Avoids the 24 GB bf16 spike from from_single_file(torch_dtype=bfloat16).
        # Uses diffusers' own key-conversion utility + meta device for zero-copy loading.
        import torch.nn as _nn
        from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers

        print("[ImageEngine] Loading transformer state dict (fp8, ~12 GB)…")
        _raw_sd = _sf_load(str(_FLUX_CKPT))
        # Extract ComfyUI transformer keys and strip prefix
        _XPFX = "model.diffusion_model."
        _xfmr_sd_comfy = {k[len(_XPFX):]: v for k, v in _raw_sd.items() if k.startswith(_XPFX)}
        del _raw_sd

        # Convert ComfyUI key names → diffusers key names (no dtype change)
        _xfmr_sd = convert_flux_transformer_checkpoint_to_diffusers(_xfmr_sd_comfy)
        del _xfmr_sd_comfy

        # Create model with meta tensors (zero RAM), then assign real fp8 weights
        with torch.device("meta"):
            transformer = FluxTransformer2DModel.from_config(str(_CFG_DIR / "transformer"))
        missing, unexpected = transformer.load_state_dict(_xfmr_sd, strict=False, assign=True)
        del _xfmr_sd
        print(f"[ImageEngine] Transformer load: {len(missing)} missing, {len(unexpected)} unexpected")

        # Cast all non-linear-weight fp8 params (norms, scales, biases) to bf16.
        # These are small tensors — their fp8 dtype would corrupt hidden-state dtypes.
        _linear_weight_names = {
            n + ".weight"
            for n, m in transformer.named_modules()
            if isinstance(m, _nn.Linear)
        }
        for _pname, _p in transformer.named_parameters():
            if _p.dtype == torch.float8_e4m3fn and _pname not in _linear_weight_names:
                _p.data = _p.data.to(torch.bfloat16)

        # Register per-layer hooks: fp8 Linear weights → bf16 just for each matmul
        def _fp8_pre(mod, args):
            if hasattr(mod, 'weight') and mod.weight.dtype == torch.float8_e4m3fn:
                mod._w8 = mod.weight
                mod.weight = _nn.Parameter(mod.weight.to(torch.bfloat16), requires_grad=False)
        def _fp8_post(mod, inp, out):
            if hasattr(mod, '_w8'):
                mod.weight = mod._w8
                del mod._w8
        _fp8_count = 0
        for _m in transformer.modules():
            if isinstance(_m, _nn.Linear) and _m.weight.dtype == torch.float8_e4m3fn:
                _m.register_forward_pre_hook(_fp8_pre)
                _m.register_forward_hook(_fp8_post)
                _fp8_count += 1
        print(f"[ImageEngine] Transformer ready: {_fp8_count} fp8 linear weights, hooks registered")

        vae = AutoencoderKL.from_single_file(
            str(_VAE_CKPT), torch_dtype=_dtype,
            config=str(_CFG_DIR / "vae"),
        )

        _pipe = FluxPipeline(
            scheduler=scheduler,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=transformer,
            vae=vae,
        )
        print("[ImageEngine] Pipeline assembled from local components")

        # ── Fuse uncensored LoRAs (GPU-accelerated, fp8 preserved) ──────────────
        # CPU fp8 ↔ bf16 casts have no hardware support and take 30+ min.
        # GPU handles fp8 natively — 504 layers fuse in seconds.
        _gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        def _fuse_lora_diffusers(xfmr, sd, scale, label):
            """Fuse a diffusers-format LoRA (lora_A/lora_B keys) into transformer weights."""
            pairs: dict = {}
            for k, v in sd.items():
                if ".lora_A.weight" in k:
                    mk = k.replace("transformer.", "").replace(".lora_A.weight", "")
                    pairs.setdefault(mk, {})["A"] = v
                elif ".lora_B.weight" in k:
                    mk = k.replace("transformer.", "").replace(".lora_B.weight", "")
                    pairs.setdefault(mk, {})["B"] = v
            fused = 0
            for mk, ab in pairs.items():
                if "A" not in ab or "B" not in ab:
                    continue
                mod = xfmr
                try:
                    for p in mk.split("."):
                        mod = getattr(mod, p)
                except AttributeError:
                    continue
                if not hasattr(mod, "weight") or mod.weight is None:
                    continue
                orig_dtype = mod.weight.data.dtype
                w  = mod.weight.data.to(device=_gpu, dtype=torch.bfloat16)
                lA = ab["A"].to(device=_gpu, dtype=torch.bfloat16)
                lB = ab["B"].to(device=_gpu, dtype=torch.bfloat16)
                mod.weight.data = (w + scale * (lB @ lA)).to(dtype=orig_dtype, device="cpu")
                del w, lA, lB
                fused += 1
            torch.cuda.empty_cache()
            print(f"[ImageEngine] {label}: fused {fused} layers @ scale {scale}")
            return fused

        def _fuse_lora_kohya(xfmr, sd, scale, label):
            """Fuse a Kohya-format Flux LoRA (lora_unet_* / lora_down / lora_up keys)."""
            # Build key map: kohya underscore path -> (down, up, alpha)
            alphas: dict = {}
            downs:  dict = {}
            ups:    dict = {}
            for k, v in sd.items():
                if not k.startswith("lora_unet_"):
                    continue  # skip te1/te2 text-encoder keys
                base = k.split(".")[0]  # e.g. "lora_unet_double_blocks_0_img_attn_proj"
                if k.endswith(".alpha"):
                    alphas[base] = float(v)
                elif ".lora_down.weight" in k:
                    downs[base] = v
                elif ".lora_up.weight" in k:
                    ups[base] = v
            # Kohya key -> diffusers module path for Flux
            # lora_unet_double_blocks_{i}_img_attn_proj -> double_blocks.{i}.attn.to_out.0
            # lora_unet_double_blocks_{i}_img_attn_qkv  -> double_blocks.{i}.attn.to_qkv  (if exists)
            # lora_unet_double_blocks_{i}_img_mlp_0     -> double_blocks.{i}.ff.net.0.proj
            # lora_unet_double_blocks_{i}_img_mlp_2     -> double_blocks.{i}.ff.net.2
            # lora_unet_double_blocks_{i}_txt_attn_proj -> double_blocks.{i}.attn.to_add_out
            # lora_unet_double_blocks_{i}_txt_mlp_0     -> double_blocks.{i}.ff_context.net.0.proj
            # lora_unet_double_blocks_{i}_txt_mlp_2     -> double_blocks.{i}.ff_context.net.2
            # lora_unet_single_blocks_{i}_linear1       -> single_transformer_blocks.{i}.attn.to_q (approx)
            # lora_unet_single_blocks_{i}_linear2       -> single_transformer_blocks.{i}.proj_out
            def _kohya_to_diffusers(key: str):
                # strip prefix, split on underscore runs
                k = key[len("lora_unet_"):]
                if k.startswith("double_blocks_"):
                    parts = k.split("_")
                    idx = parts[2]
                    rest = "_".join(parts[3:])
                    mapping = {
                        "img_attn_proj": f"double_blocks.{idx}.attn.to_out.0",
                        "img_attn_qkv":  f"double_blocks.{idx}.attn.to_qkv",
                        "img_mlp_0":     f"double_blocks.{idx}.ff.net.0.proj",
                        "img_mlp_2":     f"double_blocks.{idx}.ff.net.2",
                        "txt_attn_proj": f"double_blocks.{idx}.attn.to_add_out",
                        "txt_attn_qkv":  f"double_blocks.{idx}.attn.add_q_proj",
                        "txt_mlp_0":     f"double_blocks.{idx}.ff_context.net.0.proj",
                        "txt_mlp_2":     f"double_blocks.{idx}.ff_context.net.2",
                    }
                    return mapping.get(rest)
                elif k.startswith("single_blocks_"):
                    parts = k.split("_")
                    idx = parts[2]
                    rest = "_".join(parts[3:])
                    mapping = {
                        "linear1": f"single_transformer_blocks.{idx}.attn.to_q",
                        "linear2": f"single_transformer_blocks.{idx}.proj_out",
                    }
                    return mapping.get(rest)
                return None

            fused = 0
            skipped = 0
            for base in downs:
                if base not in ups:
                    continue
                mod_path = _kohya_to_diffusers(base)
                if mod_path is None:
                    continue
                mod = xfmr
                try:
                    for p in mod_path.split("."):
                        mod = getattr(mod, p)
                except AttributeError:
                    continue
                if not hasattr(mod, "weight") or mod.weight is None:
                    continue
                alpha = alphas.get(base, float(downs[base].shape[0]))
                rank  = downs[base].shape[0]
                effective_scale = scale * (alpha / rank)
                orig_dtype = mod.weight.data.dtype
                w    = mod.weight.data.to(device=_gpu, dtype=torch.bfloat16)
                down = downs[base].to(device=_gpu, dtype=torch.bfloat16)
                up   = ups[base].to(device=_gpu, dtype=torch.bfloat16)
                delta = effective_scale * (up @ down)
                if delta.shape != w.shape:
                    del w, down, up, delta
                    skipped += 1
                    continue
                mod.weight.data = (w + delta).to(dtype=orig_dtype, device="cpu")
                del w, down, up, delta
                fused += 1
            torch.cuda.empty_cache()
            print(f"[ImageEngine] {label}: fused {fused}, skipped {skipped} @ scale {scale}")
            return fused

        # Stack all three LoRAs
        _LORAS = [
            (Path(r"F:/models/loras/flux_uncensored_v2.safetensors"),    "diffusers", 1.2),
            (Path(r"F:/models/loras/flux-lora-uncensored.safetensors"),  "kohya",     1.0),
            (Path(r"F:/models/loras/NSFW-flux-lora.safetensors"),        "kohya",     1.0),
        ]
        for _lp, _fmt, _sc in _LORAS:
            if not _lp.exists():
                print(f"[ImageEngine] LoRA not found, skipping: {_lp.name}")
                continue
            try:
                _lsd = _sf_load(str(_lp))
                if _fmt == "diffusers":
                    _fuse_lora_diffusers(transformer, _lsd, _sc, _lp.name)
                else:
                    _fuse_lora_kohya(transformer, _lsd, _sc, _lp.name)
                del _lsd
            except Exception as _le:
                print(f"[ImageEngine] LoRA fuse failed ({_lp.name}): {_le}")

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"[ImageEngine] Failed to load Flux pipeline: {e}") from e

    # CPU-offload: each model is moved to GPU only during its forward pass.
    # With fp8 transformer (~12 GB) + bf16 T5 (~9 GB), sequential offload keeps
    # peak VRAM to max(12, 9) ≈ 12 GB instead of loading all 21 GB simultaneously.
    _pipe.enable_model_cpu_offload()

    # Guard: accelerate's cpu-offload hook casts incoming tensors to match the
    # transformer's parameter dtype (fp8).  This hook fires AFTER accelerate's
    # hook (registration order) and casts every floating-point tensor back to bf16
    # so the forward pass sees bf16 activations entering fp8-weight linear layers.
    import torch.nn as _nn2
    def _xfmr_fp8_input_guard(mod, args, kwargs):
        def _to_bf16(x):
            return (x.to(torch.bfloat16)
                    if isinstance(x, torch.Tensor) and x.is_floating_point()
                       and x.dtype != torch.bfloat16
                    else x)
        return (tuple(_to_bf16(a) for a in args),
                {k: _to_bf16(v) for k, v in kwargs.items()})

    _pipe.transformer.register_forward_pre_hook(_xfmr_fp8_input_guard, with_kwargs=True)
    print("[ImageEngine] Pipeline ready (model cpu-offload, peak VRAM ~12GB)")


def _ensure_loaded():
    global _pipe
    if _pipe is None:
        _load()


def unload():
    """Free all VRAM used by the Flux pipeline."""
    global _pipe
    with _lock:
        if _pipe is not None:
            del _pipe
            _pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            print("[ImageEngine] Pipeline unloaded")


# ── Face swap (InsightFace inswapper) ──────────────────────────────────────

_face_app  = None
_inswapper = None
_fs_lock   = threading.Lock()

def _load_face_tools():
    global _face_app, _inswapper
    if _face_app is not None:
        return True
    try:
        import insightface
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(
            name="buffalo_1",
            root=str(_INSIGHT_DIR),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))

        import onnxruntime as ort
        _inswapper = insightface.model_zoo.get_model(
            str(_INSWAP_ONNX),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print("[ImageEngine] InsightFace + inswapper ready")
        return True
    except Exception as e:
        print(f"[ImageEngine] Face-swap tools unavailable: {e}")
        return False


def _swap_face(generated: Image.Image, source_face_img: Image.Image) -> Image.Image:
    """Swap Eve's face from source_face_img into the generated image."""
    import numpy as np
    import cv2

    src_bgr = cv2.cvtColor(np.array(source_face_img), cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2BGR)

    src_faces = _face_app.get(src_bgr)
    dst_faces = _face_app.get(dst_bgr)

    if not src_faces or not dst_faces:
        print("[ImageEngine] Face detection failed — returning unswapped image")
        return generated

    src_face = src_faces[0]
    result   = dst_bgr.copy()
    for dst_face in dst_faces:
        result = _inswapper.get(result, dst_face, src_face, paste_back=True)

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


# ── Public API ──────────────────────────────────────────────────────────────

def generate(
    prompt:          str,
    negative_prompt: str  = "",
    width:           int  = 1024,
    height:          int  = 1024,
    steps:           int  = 20,
    guidance:        float = 3.5,
    seed:            int  = None,
    face_swap:       bool = False,
) -> dict:
    """
    Generate an image with Flux Dev.
    Returns {"ok": True, "image_urls": [...], "path": str}
    or      {"error": str}
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    with _lock:
        try:
            _ensure_loaded()
            generator = torch.Generator(device="cpu").manual_seed(seed)

            result_img = _pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Flux generation failed: {e}"}

    # Face swap
    if face_swap and _EVE_FACE.exists():
        with _fs_lock:
            if _load_face_tools():
                try:
                    eve_face_img = Image.open(_EVE_FACE).convert("RGB")
                    result_img   = _swap_face(result_img, eve_face_img)
                except Exception as e:
                    print(f"[ImageEngine] Face-swap failed (continuing): {e}")

    # Save
    fname    = f"eve_imagine_{uuid.uuid4().hex[:8]}.png"
    out_path = _OUT_DIR / fname
    result_img.save(out_path, "PNG")

    url = f"/static/generated/{fname}"
    return {"ok": True, "image_urls": [url], "path": str(out_path), "seed": seed}
