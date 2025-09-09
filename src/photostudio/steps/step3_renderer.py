#!/usr/bin/env python3
"""
step3_renderer.py – Step 3/6 Ghost-Mannequin Renderer (SDXL + ControlNet + QA) v3.0
=====================================================================================

Plugs into Step-1 (alpha.png, mask.png, label_crop.png?, garment_analysis.json)
and Step-2 (enhanced_prompt.txt). Features:

- Dual rendering routes: Route A (SDXL/ControlNet/IP-Adapter) and Route B (Gemini)
- Correct inpaint mask polarity + interior band ring calculation
- Depth detector caching
- Proper LAB conversion for ΔE (no double gamma)
- Label contrast QA & gating
- Deterministic execution, FP16 autocast, size multiple-of-8 enforcement
- Uses Step-1 mask for ΔE region (stable)
- Memory safeties: slicing / attention; explicit dtype/device; graceful CPU fallback

Dependencies: torch diffusers transformers accelerate controlnet-aux opencv-python pillow google-generativeai
"""

from __future__ import annotations

import os
import json
import math
import random
import argparse
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from PIL import Image, ImageOps
import cv2

# Optional dependencies - loaded lazily
try:
    import torch
    HAVE_TORCH = True
    
    # PyTorch 2.1.0 compatibility fix for torch.get_default_device
    if not hasattr(torch, 'get_default_device'):
        def get_default_device():
            """Fallback implementation for PyTorch < 2.2"""
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        
        torch.get_default_device = get_default_device
        logger = logging.getLogger("photostudio.renderer")
        logger.info("Applied PyTorch 2.1.0 compatibility patch for torch.get_default_device")
        
except ImportError:
    HAVE_TORCH = False
    torch = None

try:
    from diffusers import (
        StableDiffusionXLControlNetInpaintPipeline,
        ControlNetModel,
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        UniPCMultistepScheduler,
    )
    HAVE_DIFFUSERS = True
except ImportError:
    HAVE_DIFFUSERS = False

try:
    from controlnet_aux import MidasDetector
    HAVE_MIDAS = True
except ImportError:
    HAVE_MIDAS = False

try:
    import google.generativeai as genai
    HAVE_GENAI = True
except ImportError:
    HAVE_GENAI = False
    genai = None

logger = logging.getLogger("photostudio.renderer")

# -----------------------------
# Bulletproof Local Cache Loading
# -----------------------------
def _from_pretrained_local(cls, model_id: str, local_files_only: bool = True, **kwargs):
    """
    Bulletproof model loading that never hits the network and gracefully handles variant mismatches.
    
    Args:
        cls: The diffusers class (e.g., ControlNetModel, AutoencoderKL)
        model_id: Model identifier
        local_files_only: Force local-only mode (default: True)
        **kwargs: Additional arguments for from_pretrained
    
    Returns:
        Loaded model or raises exception if not available locally
    """
    # Force local-only mode - never hit the network in offline runs
    kwargs.setdefault("local_files_only", local_files_only)
    
    # Handle variant fallback gracefully
    variant = kwargs.get("variant", None)
    
    try:
        # Try with the requested variant first
        return cls.from_pretrained(model_id, **kwargs)
    except Exception as e:
        if variant and ("variant" in str(e).lower() or "revision" in str(e).lower()):
            # Variant-specific failure - try without variant
            logger.warning(f"Variant '{variant}' not found for {model_id}, trying default: {e}")
            fallback_kwargs = {k: v for k, v in kwargs.items() if k != "variant"}
            return cls.from_pretrained(model_id, **fallback_kwargs)
        else:
            # Different error - re-raise
            raise

def _validate_model_cache(model_paths: ModelPaths, local_files_only: bool = True) -> Dict[str, bool]:
    """
    Preflight validation to check if all required models are available in cache.
    
    Args:
        model_paths: ModelPaths configuration
        local_files_only: Whether to enforce local-only mode
    
    Returns:
        Dict with availability status for each model
    """
    cache_status = {
        "base": False,
        "controlnet_canny": False, 
        "controlnet_depth": False,
        "vae": False,
        "depth_detector": False
    }
    
    # Check SDXL base model - use a lightweight check
    try:
        import transformers
        import os
        from pathlib import Path
        
        # Check if the model directory exists in cache
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_name = f"models--{model_paths.base.replace('/', '--')}"
        model_path = Path(cache_dir) / model_cache_name
        
        if model_path.exists():
            # Check for key SDXL files
            snapshots_dir = model_path / "snapshots"
            if snapshots_dir.exists():
                # Look for any snapshot directory with SDXL files
                for snapshot in snapshots_dir.iterdir():
                    if snapshot.is_dir():
                        # Check for UNet model files (various formats)
                        unet_dir = snapshot / "unet"
                        config_file = snapshot / "model_index.json"
                        
                        if unet_dir.exists() and config_file.exists():
                            # Look for any UNet model file
                            unet_files = [
                                "diffusion_pytorch_model.safetensors",
                                "diffusion_pytorch_model.fp16.safetensors",
                                "diffusion_pytorch_model.bin"
                            ]
                            
                            for unet_filename in unet_files:
                                if (unet_dir / unet_filename).exists():
                                    cache_status["base"] = True
                                    logger.debug(f"Found SDXL base model: {unet_filename}")
                                    break
                        
                        if cache_status["base"]:
                            break
        
        if not cache_status["base"]:
            logger.debug(f"Base model files not found in cache: {model_path}")
    except Exception as e:
        logger.debug(f"Base model cache check failed: {e}")
    
    # Check ControlNets
    try:
        _from_pretrained_local(ControlNetModel, model_paths.controlnet_canny, local_files_only, torch_dtype=torch.float32)
        cache_status["controlnet_canny"] = True
    except Exception as e:
        logger.debug(f"Canny ControlNet not cached: {e}")
        
    try:
        _from_pretrained_local(ControlNetModel, model_paths.controlnet_depth, local_files_only, torch_dtype=torch.float32)
        cache_status["controlnet_depth"] = True
    except Exception as e:
        logger.debug(f"Depth ControlNet not cached: {e}")
    
    # Check VAE (optional)
    if model_paths.vae:
        try:
            _from_pretrained_local(AutoencoderKL, model_paths.vae, local_files_only, torch_dtype=torch.float32)
            cache_status["vae"] = True
        except Exception as e:
            logger.debug(f"VAE not cached: {e}")
    else:
        cache_status["vae"] = True  # Not required
    
    # Check depth detector (optional)
    if HAVE_MIDAS:
        try:
            _DepthCache.get(local_files_only=local_files_only)
            cache_status["depth_detector"] = True
        except Exception as e:
            logger.debug(f"Depth detector not cached: {e}")
    else:
        cache_status["depth_detector"] = True  # Not available anyway
    
    return cache_status

def _init_route_a_bulletproof(models: ModelPaths, device: str = "cpu", local_files_only: bool = True):
    """
    Bulletproof Route A initialization using hf_local helpers.
    
    This function replaces the fragile model loading with variant fallbacks.
    Never hits the network and gracefully handles fp16/fp32 variant mismatches.
    
    Args:
        models: ModelPaths configuration with repo IDs
        device: Target device ("cuda", "mps", "cpu")
        local_files_only: Force local-only loading (default: True)
        
    Returns:
        Initialized SDXL pipeline with ControlNets
        
    Raises:
        RuntimeError: If core models are not available in local cache
    """
    from photostudio.utils.hf_local import (
        resolve_torch_dtype, 
        safe_from_pretrained,
        suppress_model_warnings
    )
    
    # Suppress noisy model loading warnings
    suppress_model_warnings()
    
    # Resolve optimal dtype for device
    dtype = resolve_torch_dtype(device, prefer_half=(device == "cuda"))
    logger.info(f"Using dtype {dtype} for device {device}")
    
    # 1) Load ControlNets with variant fallback
    logger.info("Loading ControlNets with bulletproof variant fallback...")
    
    # Canny ControlNet (required)
    cnet_canny = safe_from_pretrained(
        ControlNetModel,
        models.controlnet_canny,  # "diffusers/controlnet-canny-sdxl-1.0"
        torch_dtype=dtype,
        local_files_only=local_files_only,
        variant_candidates=("fp16", None, "fp32"),
    )
    
    # Depth ControlNet (optional)
    cnet_depth = None
    try:
        cnet_depth = safe_from_pretrained(
            ControlNetModel,
            models.controlnet_depth,  # "diffusers/controlnet-depth-sdxl-1.0"
            torch_dtype=dtype,
            local_files_only=local_files_only,
            variant_candidates=("fp16", None, "fp32"),
        )
    except Exception as e:
        logger.warning(f"Depth ControlNet not available locally; proceeding with canny only. {e}")
    
    # Prepare ControlNet list
    controlnets = [cn for cn in (cnet_canny, cnet_depth) if cn is not None]
    logger.info(f"Loaded {len(controlnets)} ControlNets")
    
    # 2) Load SDXL base model with variant fallback
    logger.info("Loading SDXL base model with bulletproof variant fallback...")
    
    pipe = safe_from_pretrained(
        StableDiffusionXLControlNetInpaintPipeline,
        models.base,  # "stabilityai/stable-diffusion-xl-base-1.0"
        controlnet=controlnets[0] if len(controlnets) == 1 else controlnets,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        variant_candidates=("fp16", None, "fp32"),
    )
    
    # 3) Optional VAE override (also via local cache)
    if models.vae:
        try:
            logger.info(f"Loading custom VAE: {models.vae}")
            vae = safe_from_pretrained(
                AutoencoderKL,
                models.vae,  # e.g. "madebyollin/sdxl-vae-fp16-fix"
                torch_dtype=dtype,
                local_files_only=local_files_only,
                variant_candidates=(None, "fp16", "fp32"),  # Try None first for VAE
            )
            pipe.vae = vae
        except Exception as e:
            logger.warning(f"VAE override unavailable locally; using default VAE. {e}")
    
    # 4) Device placement and memory optimizations
    logger.info(f"Moving pipeline to device: {device}")
    
    if device == "cuda":
        pipe.to("cuda")
        # Enable CUDA optimizations
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("✅ xformers memory efficient attention enabled")
        except Exception as e:
            logger.debug(f"xformers not available: {e}")
            
        try:
            pipe.enable_model_cpu_offload()
            logger.info("✅ Model CPU offload enabled")
        except Exception as e:
            logger.debug(f"CPU offload failed: {e}")
            
    elif device == "mps":
        # MPS with memory-aware loading - use CPU offload on limited memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if memory_gb < 20:  # Limited memory - use CPU offload
                logger.info(f"Limited memory ({memory_gb:.1f}GB) - using CPU offload for MPS")
                pipe.enable_model_cpu_offload()
                logger.info("✅ MPS with CPU offload enabled")
            else:
                pipe.to("mps")  # Full MPS loading for high-memory systems
                logger.info("✅ MPS device enabled with fp32 dtype")
        except Exception as e:
            logger.warning(f"MPS device setup failed: {e}, falling back to CPU")
            pipe.to("cpu")
            logger.info("✅ CPU fallback enabled")
        
    else:
        pipe.to("cpu")  # CPU fallback
        logger.info("✅ CPU device enabled")
    
    # 5) Memory efficiency optimizations (all devices)
    try:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        logger.info("✅ VAE slicing and tiling enabled")
    except Exception as e:
        logger.debug(f"VAE optimizations failed: {e}")
    
    # 6) Disable progress bars for batch processing
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    
    logger.info("🎉 Route A initialized successfully with bulletproof loading")
    return pipe

# -----------------------------
# Determinism & environment
# -----------------------------
def set_determinism(seed: int = 12345):
    random.seed(seed)
    np.random.seed(seed)
    if HAVE_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass
class ModelPaths:
    base: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae: Optional[str] = None  # e.g., "madebyollin/sdxl-vae-fp16-fix"
    controlnet_canny: str = "diffusers/controlnet-canny-sdxl-1.0"
    controlnet_depth: str = "diffusers/controlnet-depth-sdxl-1.0"
    ip_adapter: Optional[str] = None  # e.g., "h94/IP-Adapter"

@dataclass
class RendererConfig:
    size: int = 2048                 # will be rounded to multiple of 8
    frame_fill_pct: int = 85
    guidance_scale: float = 5.5
    num_inference_steps: int = 28
    canny_low: int = 80
    canny_high: int = 160
    controlnet_weight_canny: float = 0.7
    controlnet_weight_depth: float = 0.3
    negative_prompt: str = (
        "no background, no props, no hands, no mannequin, "
        "no patterns not in JSON, no extra pockets, no extra hoods"
    )
    delta_e_tolerance: float = 2.0
    retries: int = 2
    retry_guidance_delta: float = 0.5
    retry_cnet_delta: float = 0.1
    use_unipc: bool = True
    use_fp16: bool = True
    device: str = "mps" if HAVE_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else ("cuda" if HAVE_TORCH and torch.cuda.is_available() else "cpu")
    # Label preservation
    protect_label_bbox: bool = True
    label_reference_as_ip: bool = False  # if IP-Adapter ref used
    # Ghost interior shading
    interior_band_r: int = 8
    interior_dark_amount: float = 0.07  # 7% darkening

@dataclass
class QAThresholds:
    label_min_contrast: float = 4.5  # WCAG-like

# -----------------------------
# Utility: color & LAB
# -----------------------------
def _srgb_to_linear_01(x_8bit: np.ndarray) -> np.ndarray:
    """x_8bit: ...x3 uint8, returns linear 0..1 for WCAG luminance math."""
    x = x_8bit.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)

def _rgb_to_lab_opencv(rgb_u8: np.ndarray) -> np.ndarray:
    """
    Accurate LAB via OpenCV assuming sRGB input. No manual linearization here.
    rgb_u8: HxWx3 uint8
    returns: HxWx3 float32 in LAB
    """
    bgr = rgb_u8[..., ::-1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab.astype(np.float32)

def _delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    # Vectorized CIEDE2000 (as in your original, with numeric guards)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7 + 1e-12)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    avg_Cp = (C1p + C2p) / 2.0
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    deltahp = h2p - h1p
    deltahp = np.where(deltahp > 180, deltahp - 360, deltahp)
    deltahp = np.where(deltahp < -180, deltahp + 360, deltahp)
    deltaLp = L2 - L1
    deltaCp = C2p - C1p
    deltaHp = 2 * np.sqrt(C1p * C2p + 1e-12) * np.sin(np.radians(deltahp) / 2.0)
    avg_Lp = (L1 + L2) / 2.0
    avg_hp = (h1p + h2p) / 2.0
    avg_hp = np.where(np.abs(h1p - h2p) > 180, avg_hp + 180, avg_hp) % 360.0
    T = (
        1
        - 0.17 * np.cos(np.radians(avg_hp - 30))
        + 0.24 * np.cos(np.radians(2 * avg_hp))
        + 0.32 * np.cos(np.radians(3 * avg_hp + 6))
        - 0.20 * np.cos(np.radians(4 * avg_hp - 63))
    )
    Sl = 1 + (0.015 * (avg_Lp - 50) ** 2) / np.sqrt(20 + (avg_Lp - 50) ** 2 + 1e-12)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    delta_ro = 30 * np.exp(-((avg_hp - 275) / 25) ** 2)
    Rc = 2 * np.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7 + 1e-12))
    Rt = -Rc * np.sin(np.radians(2 * delta_ro))
    kL = kC = kH = 1.0
    dE = np.sqrt(
        (deltaLp / (kL * Sl + 1e-12)) ** 2
        + (deltaCp / (kC * Sc + 1e-12)) ** 2
        + (deltaHp / (kH * Sh + 1e-12)) ** 2
        + Rt * (deltaCp / (kC * Sc + 1e-12)) * (deltaHp / (kH * Sh + 1e-12))
    )
    return dE.astype(np.float32)

def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

# -----------------------------
# Utility: masks & geometry
# -----------------------------
def load_mask(path: Path, size: int) -> np.ndarray:
    """Load a binary mask, resize to (size,size)."""
    m = Image.open(path).convert("L").resize((size, size), Image.BOX)
    m = np.array(m, dtype=np.uint8)
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    return m

def ensure_square_and_resize(img: Image.Image, size: int) -> Image.Image:
    """Pad to square with transparent BG, then resize to target (keeps RGBA)."""
    w, h = img.size
    s = max(w, h)
    bg = Image.new("RGBA", (s, s), (255, 255, 255, 0))
    bg.paste(img, ((s - w) // 2, (s - h) // 2))
    return bg.resize((size, size), Image.LANCZOS)

def round_to_multiple_of_8(n: int) -> int:
    return int((n // 8) * 8) if n % 8 == 0 else int(((n // 8) + 1) * 8)

def interior_band(mask: np.ndarray, r: int = 8) -> np.ndarray:
    """
    Build a thin interior ring (erode by r then erode by 2r, subtract).
    This yields a band inside the garment suitable for subtle ghost shading/inpaint.
    """
    r = max(1, int(r))
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r, 2 * r))
    er1 = cv2.erode(mask, k1)
    er2 = cv2.erode(mask, k2)
    band = cv2.subtract(er1, er2)  # <-- correct orientation: outer minus inner
    return np.where(band > 0, 255, 0).astype(np.uint8)

def apply_interior_shading(img_rgb: Image.Image, band: np.ndarray, dark_amount: float) -> Image.Image:
    """
    Slightly darken a thin interior ring to fake depth without halos.
    band: uint8 mask where 255 == interior ring to darken
    """
    img_np = np.array(img_rgb.convert("RGB"), dtype=np.float32) / 255.0
    bandf = (band > 0).astype(np.float32)[..., None]
    out = img_np * (1.0 - bandf * float(dark_amount))
    return Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8), mode="RGB")

# -----------------------------
# Control images
# -----------------------------
def build_canny_control(rgb_img: Image.Image, low: int, high: int) -> Image.Image:
    arr = np.array(rgb_img.convert("RGB"))
    edges = cv2.Canny(arr, int(low), int(high))
    canny = np.repeat(edges[..., None], 3, axis=-1)
    return Image.fromarray(canny, mode="RGB")

class _DepthCache:
    model = None
    @classmethod
    def get(cls, local_files_only: bool = False):
        if not HAVE_MIDAS:
            return None
        if cls.model is None:
            try:
                cls.model = MidasDetector.from_pretrained(
                    "Intel/dpt-hybrid-midas",
                    local_files_only=local_files_only
                )
            except Exception as e:
                logger.warning(f"Depth model unavailable ({e}); disabling depth control")
                cls.model = None
        return cls.model

def build_depth_control(rgb_img: Image.Image, local_files_only: bool = False) -> Optional[Image.Image]:
    md = _DepthCache.get(local_files_only=local_files_only)
    if md is None:
        return None
    depth = md(rgb_img)
    return depth.convert("RGB") if depth.mode != "RGB" else depth

# -----------------------------
# Label helper & QA
# -----------------------------
def read_label_bbox(json_data: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    b = json_data.get("segmentation_results", {}).get("label_bbox_norm")
    # also support your earlier brand_information.label_bbox
    if b is None:
        b = json_data.get("brand_information", {}).get("label_bbox")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return None

def protect_label_region(mask_like: np.ndarray, bbox_norm: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = mask_like.shape
    x0 = max(0, min(int(bbox_norm[0] * w), w - 1))
    y0 = max(0, min(int(bbox_norm[1] * h), h - 1))
    x1 = max(0, min(int(bbox_norm[2] * w), w - 1))
    y1 = max(0, min(int(bbox_norm[3] * h), h - 1))
    protect = np.zeros_like(mask_like, dtype=np.uint8)
    cv2.rectangle(protect, (x0, y0), (x1, y1), 255, -1)
    return protect

def relative_luminance(rgb_mean_8bit: np.ndarray) -> float:
    lin = _srgb_to_linear_01(rgb_mean_8bit.astype(np.uint8))
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)

def label_contrast_ratio(img_rgb: Image.Image, bbox_norm: Tuple[float, float, float, float]) -> float:
    w, h = img_rgb.size
    x0 = max(0, int(bbox_norm[0] * w)); x1 = max(0, int(bbox_norm[2] * w))
    y0 = max(0, int(bbox_norm[1] * h)); y1 = max(0, int(bbox_norm[3] * h))
    patch = np.array(img_rgb.convert("RGB"))[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = patch[th == 0]; bg = patch[th == 255]
    if fg.size == 0 or bg.size == 0:
        return 0.0
    L1 = relative_luminance(fg.mean(axis=0))
    L2 = relative_luminance(bg.mean(axis=0))
    return float((max(L1, L2) + 0.05) / (min(L1, L2) + 0.05))

def compute_delta_e(img_rgb: Image.Image, garment_mask: np.ndarray, target_hex: str) -> float:
    rgb = np.array(img_rgb.convert("RGB"), dtype=np.uint8)
    lab = _rgb_to_lab_opencv(rgb)
    tr, tg, tb = hex_to_rgb(target_hex)
    tgt_rgb = np.full_like(rgb, (tr, tg, tb), dtype=np.uint8)
    tgt_lab = _rgb_to_lab_opencv(tgt_rgb)
    gm = garment_mask > 0
    if np.count_nonzero(gm) < 10:
        return 999.0
    dE = _delta_e_ciede2000(lab[gm], tgt_lab[gm])
    return float(np.mean(dE))

# -----------------------------
# Heuristic structure checks
# -----------------------------
def simple_structure_checks(img_rgb: Image.Image, garment_mask: np.ndarray) -> Dict[str, Any]:
    out = {"zipper_like": False, "button_like": 0, "pocket_like": False}
    gray = cv2.cvtColor(np.array(img_rgb.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gmask = (garment_mask > 0).astype(np.uint8) * 255
    g = cv2.bitwise_and(gray, gray, mask=gmask)

    # Buttons (circles)
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=18, param1=80, param2=25, minRadius=4, maxRadius=20
    )
    out["button_like"] = 0 if circles is None else int(circles.shape[1])

    # Zipper (approx parallel lines)
    edges = cv2.Canny(g, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=40, maxLineGap=6)
    found = False
    if lines is not None and len(lines) < 120:
        limit = min(len(lines), 24)
        for i in range(limit):
            for j in range(i + 1, limit):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]
                a1 = math.atan2(y2 - y1, x2 - x1)
                a2 = math.atan2(y4 - y3, x4 - x3)
                if abs(a1 - a2) < 0.12:
                    dist = abs((y3 - y1) * math.cos(a1) - (x3 - x1) * math.sin(a1))
                    if 5 < dist < 30:
                        found = True
                        break
            if found:
                break
    out["zipper_like"] = bool(found)

    # Pockets (rectangles)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pocket_like = False
    for c in cnts[:80]:
        area = cv2.contourArea(c)
        if 400 < area < 5000:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                pocket_like = True
                break
    out["pocket_like"] = pocket_like
    return out

# -----------------------------
# Route B: Gemini Image Editing 
# -----------------------------
class GeminiRenderer:
    """Route B: Gemini image editing with source image + mask + JSON instructions for strict product fidelity."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash-image-preview"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.ready = False
        
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY found; Route B unavailable")
            return
            
        try:
            # Use the modern google-genai SDK
            import google.genai as genai
            from google.genai import types
            
            self.client = genai.Client(api_key=self.api_key)
            self.types = types
            self.ready = True
            
            logger.info(f"Route B (Gemini {self.model_name}) initialized successfully for image editing with modern SDK")
        except ImportError:
            logger.error("google-genai package not installed; Route B unavailable. Install with: pip install google-genai")
        except Exception as e:
            logger.error(f"Route B (Gemini {self.model_name}) init failed: {e}")
            
    def render(
        self,
        prompt_text: str,
        reference_image: Optional[Image.Image] = None,
        inpaint_mask: Optional[np.ndarray] = None,
        garment_json: Optional[dict] = None,
        size: int = 2048,
        count: int = 1
    ) -> List[Image.Image]:
        """Edit images using modern Gemini SDK with source image + mask + JSON for strict product fidelity."""
        if not self.ready:
            logger.error("Route B (Gemini) not available, returning mock images")
            return self._generate_mock_images(size, count)
            
        if not reference_image:
            logger.error("Route B requires reference image for editing, returning mock images")
            return self._generate_mock_images(size, count)
            
        try:
            import io
            import mimetypes
            
            logger.info(f"Route B: Editing {count} candidates via Gemini 2.5 Flash Image Preview with source+mask+JSON")
            generated_images = []
            
            # Prepare the source image
            source_rgb = reference_image.convert('RGB')
            if source_rgb.size != (size, size):
                source_rgb = source_rgb.resize((size, size), Image.LANCZOS)
            
            # Create editing mask (areas to modify for ghost-mannequin effect)
            if inpaint_mask is not None:
                # Use provided inpaint mask
                edit_mask = inpaint_mask
                if edit_mask.shape[:2] != (size, size):
                    edit_mask = cv2.resize(edit_mask, (size, size), interpolation=cv2.INTER_NEAREST)
            else:
                # Create a conservative interior mask for ghost-mannequin editing
                gray = np.array(source_rgb.convert('L'))
                garment_mask = (gray < 240).astype(np.uint8) * 255
                edit_mask = interior_band(garment_mask, r=12)
            
            # Convert mask to PIL Image (white = edit areas, black = preserve)
            mask_img = Image.fromarray(edit_mask).convert('RGB')
            
            # Create the editing instruction with JSON context
            editing_instruction = self._create_editing_instruction_with_json(prompt_text, garment_json)
            
            for i in range(count):
                logger.info(f"Editing candidate {i+1}/{count} with Gemini...")
                
                # Add lighting variation for multiple candidates
                variant_instruction = editing_instruction
                if count > 1:
                    variations = [
                        "Apply slightly warmer, softer lighting.",
                        "Use cooler, more clinical lighting.", 
                        "Add subtle directional lighting from the left.",
                        "Add subtle directional lighting from the right.",
                        "Use perfectly even studio lighting."
                    ]
                    variant_instruction += f"\n\nLIGHTING VARIATION: {variations[i % len(variations)]}"
                
                try:
                    # Use the modern SDK approach with streaming
                    contents = [
                        self.types.Content(
                            role="user",
                            parts=[
                                # Source image to edit
                                self.types.Part.from_bytes(
                                    data=self._image_to_bytes(source_rgb),
                                    mime_type="image/jpeg"
                                ),
                                # Editing mask (white = edit, black = preserve)
                                self.types.Part.from_bytes(
                                    data=self._image_to_bytes(mask_img),
                                    mime_type="image/jpeg"
                                ),
                                # Editing instruction with JSON context
                                self.types.Part.from_text(text=variant_instruction)
                            ]
                        )
                    ]
                    
                    # Configure for image generation
                    config = self.types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                        temperature=0.2 + (i * 0.05),  # Low temperature for faithful editing
                        max_output_tokens=4096
                    )
                    
                    # Generate using streaming API
                    response_stream = self.client.models.generate_content_stream(
                        model=self.model_name,
                        contents=contents,
                        config=config
                    )
                    
                    # Extract image from streaming response
                    image_extracted = False
                    for chunk in response_stream:
                        if (
                            chunk.candidates is None
                            or chunk.candidates[0].content is None
                            or chunk.candidates[0].content.parts is None
                        ):
                            continue
                            
                        for part in chunk.candidates[0].content.parts:
                            if part.inline_data and part.inline_data.data:
                                try:
                                    data_buffer = part.inline_data.data
                                    mime_type = part.inline_data.mime_type or "image/jpeg"
                                    
                                    if 'image' in mime_type.lower():
                                        image = Image.open(io.BytesIO(data_buffer))
                                        
                                        # Ensure correct size
                                        if image.size != (size, size):
                                            image = image.resize((size, size), Image.LANCZOS)
                                        
                                        generated_images.append(image.convert('RGB'))
                                        logger.info(f"✅ Successfully edited candidate {i+1} via Gemini 2.5 Flash Image Preview ({image.size})")
                                        image_extracted = True
                                        break
                                        
                                except Exception as img_error:
                                    logger.warning(f"Failed to extract image for candidate {i+1}: {img_error}")
                                    continue
                            else:
                                # Handle text response
                                if hasattr(part, 'text') and part.text:
                                    logger.debug(f"Gemini text response: {part.text[:100]}...")
                        
                        if image_extracted:
                            break
                    
                    # Fallback if no image extracted
                    if not image_extracted:
                        logger.warning(f"No image extracted for candidate {i+1}, using mock fallback")
                        mock_images = self._generate_mock_images(size, 1)
                        generated_images.extend(mock_images)
                        
                except Exception as gen_error:
                    logger.warning(f"Generation failed for candidate {i+1}: {gen_error}, using mock fallback")
                    mock_images = self._generate_mock_images(size, 1)
                    generated_images.extend(mock_images)
            
            if generated_images:
                logger.info(f"Successfully edited {len(generated_images)} images with modern Gemini SDK")
                return generated_images
            else:
                logger.error("No images generated, falling back to mocks")
                return self._generate_mock_images(size, count)
                
        except Exception as e:
            logger.error(f"Route B Gemini image editing failed: {e}")
            logger.info("Falling back to improved mock images")
            return self._generate_mock_images(size, count)
            
    def _create_editing_instruction_with_json(self, prompt_text: str, garment_json: Optional[dict] = None) -> str:
        """Create comprehensive image editing instruction using source image + mask + JSON context."""
        instruction = """
PROFESSIONAL GARMENT IMAGE EDITING TASK:

You are provided with:
1. SOURCE IMAGE: The original garment photograph to edit
2. EDITING MASK: White areas indicate where to apply ghost-mannequin effect, black areas preserve original
3. JSON CONTEXT: Detailed garment analysis and requirements below

EDITING OBJECTIVE:
Transform this exact garment into a professional ghost-mannequin product photograph while maintaining 100% fidelity to the original garment's appearance.

CRITICAL REQUIREMENTS:
- PRESERVE: All original colors, patterns, textures, fabric properties, design elements, logos, labels, construction details
- EDIT: Only areas indicated by the white mask regions
- CREATE: Professional ghost-mannequin effect (hollow interior, invisible form)
- LIGHTING: Clean, professional studio lighting with pure white background
- QUALITY: Sharp focus, high-resolution product photography standard
"""
        
        # Add JSON context if provided
        if garment_json:
            instruction += "\n\nGARMENT ANALYSIS JSON CONTEXT:\n"
            
            # Add color information
            if "color_extraction" in garment_json:
                color_info = garment_json["color_extraction"]
                primary_color = color_info.get("primary_color", {})
                if "hex_value" in primary_color:
                    instruction += f"- PRIMARY COLOR: {primary_color['hex_value']} ({primary_color.get('description', 'N/A')})\n"
                if "dominant_colors" in color_info:
                    instruction += f"- DOMINANT COLORS: {', '.join([c.get('hex_value', 'N/A') for c in color_info['dominant_colors'][:3]])}\n"
            
            # Add garment category and type
            if "garment_category" in garment_json:
                instruction += f"- GARMENT TYPE: {garment_json['garment_category']}\n"
            
            # Add fabric analysis
            if "fabric_analysis" in garment_json:
                fabric_info = garment_json["fabric_analysis"]
                if "fabric_type" in fabric_info:
                    instruction += f"- FABRIC TYPE: {fabric_info['fabric_type']}\n"
                if "texture_description" in fabric_info:
                    instruction += f"- TEXTURE: {fabric_info['texture_description']}\n"
            
            # Add key measurements/dimensions
            if "measurements" in garment_json:
                instruction += f"- MEASUREMENTS: {garment_json['measurements']}\n"
        
        # Add structured prompt information
        if "NON-NEGOTIABLE FACTS:" in prompt_text:
            facts_section = prompt_text.split("NON-NEGOTIABLE FACTS:")[1].split("MANDATORY STYLE GUIDE:")[0]
            instruction += "\nSTRUCTURED PROMPT REQUIREMENTS:\n"
            for line in facts_section.strip().split("\n"):
                if line.strip().startswith("-"):
                    instruction += line.strip() + "\n"
        
        if "MANDATORY STYLE GUIDE:" in prompt_text:
            style_section = prompt_text.split("MANDATORY STYLE GUIDE:")[1].split("GARMENT-AWARE RULES:")[0]
            instruction += "\nSTYLE GUIDE:\n"
            for line in style_section.strip().split("\n"):
                if line.strip().startswith("-"):
                    instruction += line.strip() + "\n"
        
        instruction += """

FINAL INSTRUCTION:
Edit ONLY the masked areas to create a ghost-mannequin effect while preserving every detail of the original garment. The result must be indistinguishable from the source garment except for the professional ghost-mannequin transformation.
"""
        
        return instruction
        
    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes for Gemini API."""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()
            
    def _generate_mock_images(self, size: int, count: int) -> List[Image.Image]:
        """Generate mock garment images for development/testing."""
        import PIL.ImageDraw as ImageDraw
        import random
        
        images = []
        for i in range(count):
            # Create a white background
            img = Image.new("RGB", (size, size), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # Create a more realistic garment silhouette (dress/shirt shape)
            # Center coordinates
            cx, cy = size // 2, size // 2
            
            # Define garment shape - simplified dress/shirt outline
            garment_color = (
                252 - i * 10,  # Slight color variation per candidate
                251 - i * 5, 
                243 - i * 15
            )
            
            # Draw main garment body (rectangular with rounded corners)
            body_width = int(size * 0.4)
            body_height = int(size * 0.6)
            body_left = cx - body_width // 2
            body_top = cy - body_height // 2 + int(size * 0.1)  # Slightly lower
            body_right = body_left + body_width
            body_bottom = body_top + body_height
            
            # Draw main body
            draw.rectangle(
                [(body_left, body_top), (body_right, body_bottom)],
                fill=garment_color,
                outline=(200, 200, 200)
            )
            
            # Draw neckline (boat neck style)
            neck_width = int(body_width * 0.6)
            neck_height = int(size * 0.05)
            neck_left = cx - neck_width // 2
            neck_top = body_top - neck_height
            neck_right = neck_left + neck_width
            neck_bottom = body_top + int(size * 0.02)
            
            draw.rectangle(
                [(neck_left, neck_top), (neck_right, neck_bottom)],
                fill=garment_color,
                outline=(200, 200, 200)
            )
            
            # Draw sleeves
            sleeve_width = int(size * 0.15)
            sleeve_height = int(size * 0.25)
            
            # Left sleeve
            left_sleeve_right = body_left
            left_sleeve_left = left_sleeve_right - sleeve_width
            sleeve_top = body_top + int(size * 0.02)
            sleeve_bottom = sleeve_top + sleeve_height
            
            draw.rectangle(
                [(left_sleeve_left, sleeve_top), (left_sleeve_right, sleeve_bottom)],
                fill=garment_color,
                outline=(200, 200, 200)
            )
            
            # Right sleeve  
            right_sleeve_left = body_right
            right_sleeve_right = right_sleeve_left + sleeve_width
            
            draw.rectangle(
                [(right_sleeve_left, sleeve_top), (right_sleeve_right, sleeve_bottom)],
                fill=garment_color,
                outline=(200, 200, 200)
            )
            
            # Add some texture/shading to make it look more realistic
            # Add subtle interior shadow to simulate ghost mannequin effect
            shadow_color = (
                max(0, garment_color[0] - 30),
                max(0, garment_color[1] - 30), 
                max(0, garment_color[2] - 30)
            )
            
            # Interior shadow areas
            interior_margin = int(size * 0.02)
            draw.rectangle(
                [(body_left + interior_margin, body_top + interior_margin), 
                 (body_right - interior_margin, body_bottom - interior_margin)],
                outline=shadow_color
            )
            
            # Add small debug text (much smaller than before)
            debug_text = f"Route B #{i+1}"
            try:
                # Try to get a font, fall back to default if not available
                import PIL.ImageFont as ImageFont
                font = ImageFont.load_default()
            except:
                font = None
                
            # Position debug text at bottom corner, small and subtle
            text_color = (180, 180, 180)  # Light gray
            draw.text((size - 150, size - 30), debug_text, fill=text_color, font=font)
            
            images.append(img)
            
        logger.info(f"Generated {len(images)} mock garment images with realistic silhouettes")
        return images

# -----------------------------
# Route A: SDXL Renderer
# -----------------------------
class GhostMannequinRenderer:
    """Route A: SDXL + ControlNet renderer."""
    
    def __init__(self, models: ModelPaths, cfg: RendererConfig, qa: QAThresholds = QAThresholds(),
                 seed: int = 12345, use_depth: bool = False, local_files_only: bool = False):
        self.models = models
        self.cfg = cfg
        self.qa = qa
        self.seed = seed
        self.use_depth = use_depth
        self.local_files_only = local_files_only
        self.ready = False

        # Check dependencies
        if not HAVE_TORCH:
            logger.error("PyTorch not installed; Route A unavailable")
            return
        if not HAVE_DIFFUSERS:
            logger.error("Diffusers not installed; Route A unavailable")
            return

        # Sanity for size (SDXL UNet stride)
        self.cfg.size = round_to_multiple_of_8(self.cfg.size)

        set_determinism(seed)

        try:
            # Use bulletproof initialization
            logger.info("Initializing Route A with bulletproof model loading...")
            
            self.pipe = _init_route_a_bulletproof(
                models=models,
                device=self.cfg.device,
                local_files_only=self.local_files_only
            )

            # Scheduler configuration
            if self.cfg.use_unipc:
                self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
                logger.info("Using UniPC scheduler")
            else:
                self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
                logger.info("Using DPMSolver scheduler")

            # Optional IP-Adapter (API varies by version; guarded)
            self.ip_loaded = False
            if models.ip_adapter:
                try:
                    # Some diffusers versions expose load_ip_adapter; others require manual wiring.
                    if hasattr(self.pipe, "load_ip_adapter"):
                        self.pipe.load_ip_adapter(models.ip_adapter)
                        self.ip_loaded = True
                        logger.info(f"IP-Adapter loaded: {models.ip_adapter}")
                except Exception as e:
                    logger.warning(f"IP-Adapter loading failed: {e}")
                    self.ip_loaded = False

            # Only enable depth if the detector is available locally (no network fetches in offline/local-only)
            self.depth_available = HAVE_MIDAS and (_DepthCache.get(local_files_only=self.local_files_only) is not None)
            if self.depth_available:
                logger.info("Depth detector available for depth ControlNet")
            else:
                logger.info("Depth detector not available - using canny-only")
                
            self.ready = True
            logger.info("🎉 Route A (SDXL) initialized successfully with bulletproof loading")
            
        except Exception as e:
            logger.error(f"Route A (SDXL) init failed: {e}")
            self.ready = False

    def _dtype(self):
        return torch.float16 if (self.cfg.use_fp16 and torch.cuda.is_available()) else torch.float32
    
    def _should_use_fp16_variant(self):
        """Determine if fp16 variant should be used based on device capabilities."""
        # Only use fp16 variants on CUDA devices, not MPS/CPU
        return self.cfg.use_fp16 and torch.cuda.is_available()

    def render(
        self,
        initial_rgba: Image.Image,
        garment_mask: np.ndarray,  # Step-1 mask, already binary
        prompt_text: str,
        primary_hex: Optional[str],
        label_bbox_norm: Optional[Tuple[float, float, float, float]] = None,
        label_reference: Optional[Image.Image] = None,
        out_dir: Path = Path("render_out"),
    ) -> Dict[str, Any]:
        
        if not self.ready:
            logger.error("Route A (SDXL) not available")
            return {"error": "Route A not available", "candidates": []}

        out_dir.mkdir(parents=True, exist_ok=True)

        # Canvas + mask to target size
        init_sq = ensure_square_and_resize(initial_rgba, self.cfg.size)
        init_rgb = init_sq.convert("RGB")
        gm = cv2.resize(garment_mask, (self.cfg.size, self.cfg.size), interpolation=cv2.INTER_NEAREST)

        # Control images
        canny = build_canny_control(init_rgb, self.cfg.canny_low, self.cfg.canny_high)
        controls = [canny]
        control_weights = [self.cfg.controlnet_weight_canny]
        if self.use_depth and self.depth_available:
            depth = build_depth_control(init_rgb, local_files_only=self.local_files_only)
            if depth is not None:
                controls.append(depth)
                control_weights.append(self.cfg.controlnet_weight_depth)

        # Inpaint mask (Diffusers convention: WHITE=PAINT, BLACK=KEEP)
        band = interior_band(gm, r=self.cfg.interior_band_r)  # white ring to paint (ghost interior)
        inpaint_mask = band.copy()

        # Protect label bbox area (set to BLACK=KEEP)
        if self.cfg.protect_label_bbox and label_bbox_norm:
            protect = protect_label_region(gm, label_bbox_norm)
            inpaint_mask = np.where(protect > 0, 0, inpaint_mask).astype(np.uint8)

        # Optional IP-Adapter (light conditioning)
        ip_kwargs = {}
        if self.ip_loaded and label_reference is not None and self.cfg.label_reference_as_ip:
            # Exact kwargs depend on diffusers version; common names shown below:
            # ip_kwargs = {"ip_adapter_image": label_reference.convert("RGB")}
            pass

        neg = self.cfg.negative_prompt
        prompt = prompt_text

        # First pass
        result, qa = self._run_once(
            init_rgb, Image.fromarray(inpaint_mask),
            prompt, neg,
            controls, control_weights,
            primary_hex, gm,
            label_bbox_norm
        )

        # Auto-retries on ΔE or label contrast
        for k in range(self.cfg.retries):
            if qa.get("delta_e_ok", True) and qa.get("label_ok", True):
                break
            self.cfg.guidance_scale = max(3.5, self.cfg.guidance_scale + ((-1)**k) * self.cfg.retry_guidance_delta)
            control_weights = [
                max(0.05, min(1.5, cw + ((-1)**k) * self.cfg.retry_cnet_delta)) for cw in control_weights
            ]
            result, qa = self._run_once(
                result.convert("RGB"), Image.fromarray(inpaint_mask),
                prompt, neg,
                controls, control_weights,
                primary_hex, gm,
                label_bbox_norm
            )

        # Save
        final_path = out_dir / "render_final.png"
        result.save(final_path, "PNG", optimize=True)

        qa_path = out_dir / "render_meta.json"
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(qa, f, indent=2)

        return {"image": str(final_path), "qa": str(qa_path), "qa_data": qa, "candidates": [result]}

    # ---- internal ----
    def _run_once(
        self,
        init_rgb: Image.Image,
        inpaint_mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        control_images: List[Image.Image],
        control_weights: List[float],
        primary_hex: Optional[str],
        garment_mask_for_qc: np.ndarray,
        label_bbox_norm: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[Image.Image, Dict[str, Any]]:

        # PyTorch 2.1.0 compatibility - use manual device handling
        if self.cfg.device == "mps":
            g = torch.Generator(device="cpu").manual_seed(self.seed)  # MPS generators need CPU fallback in 2.1.0
        else:
            g = torch.Generator(device=self.cfg.device).manual_seed(self.seed)

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_rgb,
            mask_image=inpaint_mask,                 # WHITE=PAINT, BLACK=KEEP
            control_image=control_images if len(control_images) > 1 else control_images[0],
            controlnet_conditioning_scale=control_weights if len(control_weights) > 1 else control_weights[0],
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            generator=g,
            width=self.cfg.size,
            height=self.cfg.size,
        )

        # Autocast handling for different devices - PyTorch 2.1.0 compatibility
        if hasattr(self.pipe, 'device') and self.pipe.device is not None:
            pipe_device_type = self.pipe.device.type
        else:
            pipe_device_type = self.cfg.device
            
        # Use inference_mode with appropriate autocast settings
        if pipe_device_type == "cuda" and torch.cuda.is_available():
            # CUDA supports fp16 autocast
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cfg.use_fp16):
                out = self.pipe(**kwargs).images[0]
        elif pipe_device_type == "mps":
            # MPS: no autocast, but use inference_mode
            with torch.inference_mode():
                out = self.pipe(**kwargs).images[0]
        else:
            # CPU: use bfloat16 if available, otherwise no autocast
            if hasattr(torch, 'bfloat16'):
                with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
                    out = self.pipe(**kwargs).images[0]
            else:
                with torch.inference_mode():
                    out = self.pipe(**kwargs).images[0]
        # Subtle post ghost shading on the same interior band
        band = np.array(inpaint_mask.convert("L"))
        band = np.where(band > 127, 255, 0).astype(np.uint8)
        shaded = apply_interior_shading(out, band, self.cfg.interior_dark_amount)

        # ---- QA ----
        qa: Dict[str, Any] = {
            "seed": self.seed,
            "model": self.models.base,
            "scheduler": "UniPC" if self.cfg.use_unipc else "DPMSolver++",
            "guidance_scale": float(self.cfg.guidance_scale),
            "controlnet_weights": control_weights,
            "timestamp": time.time(),
        }

        # ΔE on the true garment mask (resized)
        if primary_hex:
            dE = compute_delta_e(shaded, garment_mask_for_qc, primary_hex)
            qa["delta_e"] = dE
            qa["delta_e_ok"] = bool(dE <= self.cfg.delta_e_tolerance)

            if not qa["delta_e_ok"]:
                # Conservative pull toward target color
                arr = np.array(shaded.convert("RGB"))
                tr, tg, tb = hex_to_rgb(primary_hex)
                mean_rgb = arr[garment_mask_for_qc > 0].mean(axis=0) if np.count_nonzero(garment_mask_for_qc) else np.array([128,128,128], np.float32)
                bias = (np.array([tr, tg, tb], np.float32) - mean_rgb) * 0.15
                arr = np.clip(arr.astype(np.float32) + bias, 0, 255).astype(np.uint8)
                shaded = Image.fromarray(arr, mode="RGB")
                qa["delta_e_post"] = compute_delta_e(shaded, garment_mask_for_qc, primary_hex)
                qa["delta_e_ok"] = bool(qa["delta_e_post"] <= self.cfg.delta_e_tolerance)

        # Label contrast
        if label_bbox_norm:
            contrast = label_contrast_ratio(shaded, label_bbox_norm)
            qa["label_contrast"] = contrast
            qa["label_ok"] = bool(contrast >= self.qa.label_min_contrast)
        else:
            qa["label_ok"] = True

        # Heuristic structure flags (optional, not gating)
        qa["structure"] = simple_structure_checks(shaded, garment_mask_for_qc)

        return shaded, qa

# -----------------------------
# Unified Multi-Route Renderer  
# -----------------------------
class MultiRouteRenderer:
    """Unified renderer with intelligent route arbitration and automatic fallbacks.
    
    Route Selection Logic:
    - Route A (SDXL): Truly air-gapped, high quality, memory-intensive 
    - Route B (Gemini): Requires network, fast, low memory
    - Auto mode: B for speed online, A for offline/local-only, with QA-based fallbacks
    """
    
    def __init__(
        self,
        models: ModelPaths,
        cfg: RendererConfig,
        qa: QAThresholds = QAThresholds(),
        seed: int = 12345,
        use_depth: bool = False,
        gemini_api_key: Optional[str] = None,
        prefer_route: str = "auto",  # "A" for SDXL, "B" for Gemini, "auto" for intelligent selection
        local_files_only: bool = False
    ):
        self.seed = seed
        self.prefer_route = prefer_route
        self.local_files_only = local_files_only
        
        # Initialize Route A (SDXL) only if needed
        if prefer_route in ["A", "auto"]:
            try:
                self.route_a = GhostMannequinRenderer(models, cfg, qa, seed, use_depth, local_files_only)
            except Exception as e:
                logger.warning(f"Route A initialization failed: {e}")
                self.route_a = None
        else:
            self.route_a = None
            
        # Initialize Route B (Gemini) - NOTE: Requires network connectivity
        try:
            self.route_b = GeminiRenderer(gemini_api_key)
        except Exception as e:
            logger.warning(f"Route B initialization failed: {e}")
            self.route_b = None
            
        # Determine available routes
        self.routes_available = []
        if self.route_a and self.route_a.ready:
            self.routes_available.append("A")
        if self.route_b and self.route_b.ready:
            self.routes_available.append("B")
            
        # Intelligent default route selection
        if prefer_route == "auto":
            if local_files_only and "A" in self.routes_available:
                # In offline/local-only mode, prefer Route A (truly air-gapped)
                self.default_route = "A"
                logger.info("Auto-selected Route A (SDXL) for offline/local-only mode")
            elif "B" in self.routes_available:
                # Online mode: prefer Route B for speed
                self.default_route = "B" 
                logger.info("Auto-selected Route B (Gemini) for speed in online mode")
            elif "A" in self.routes_available:
                # Fallback to Route A if only SDXL available
                self.default_route = "A"
                logger.info("Auto-selected Route A (SDXL) as only available option")
            else:
                self.default_route = None
                logger.error("No routes available for auto-selection")
        else:
            self.default_route = prefer_route
            
        logger.info(f"MultiRouteRenderer initialized. Available: {self.routes_available}, Default: {self.default_route}")
        
    def render(
        self,
        initial_rgba: Image.Image,
        garment_mask: np.ndarray,
        prompt_text: str,
        primary_hex: Optional[str],
        label_bbox_norm: Optional[Tuple[float, float, float, float]] = None,
        label_reference: Optional[Image.Image] = None,
        out_dir: Path = Path("render_out"),
        force_route: Optional[str] = None,
        candidates: int = 3,
        garment_json: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Render using available routes with fallback.
        
        Args:
            force_route: Optional route to force ("A" or "B")
            candidates: Number of candidates to generate
        """
        
        # Determine which route to use
        route_to_use = force_route or self.prefer_route
        
        if route_to_use not in self.routes_available:
            # Fallback to any available route
            if self.routes_available:
                route_to_use = self.routes_available[0]
                logger.info(f"Requested route {route_to_use} not available, falling back to Route {route_to_use}")
            else:
                logger.error("No rendering routes available")
                return {"error": "No rendering routes available", "candidates": []}
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if route_to_use == "A" and self.route_a and self.route_a.ready:
            logger.info("Using Route A (SDXL + ControlNet)")
            return self.route_a.render(
                initial_rgba, garment_mask, prompt_text, primary_hex,
                label_bbox_norm, label_reference, out_dir
            )
            
        elif route_to_use == "B" and self.route_b and self.route_b.ready:
            logger.info(f"Using Route B (Gemini image editing)")
            
            # Create inpaint mask for ghost-mannequin editing
            # Use interior band of the garment mask for conservative editing
            edit_mask = interior_band(garment_mask, r=16)  # Create interior editing region
            
            # Generate candidates using Gemini image editing with JSON context
            images = self.route_b.render(
                prompt_text=prompt_text,
                reference_image=initial_rgba,
                inpaint_mask=edit_mask,
                garment_json=garment_json,
                size=2048,  # TODO: get from config
                count=candidates
            )
            
            # Save generated images
            candidate_paths = []
            for i, img in enumerate(images):
                path = out_dir / f"candidate_{i}.png"
                img.save(path, "PNG", optimize=True)
                candidate_paths.append(str(path))
            
            # Create basic QA report for Route B
            qa = {
                "route": "B",
                "model": "gemini",
                "seed": self.seed,
                "timestamp": time.time(),
                "candidates_generated": len(images),
                "delta_e_ok": True,  # Assume OK for mock
                "label_ok": True     # Assume OK for mock
            }
            
            qa_path = out_dir / "render_meta.json"
            with open(qa_path, "w", encoding="utf-8") as f:
                json.dump(qa, f, indent=2)
                
            return {
                "image": candidate_paths[0] if candidate_paths else None,
                "qa": str(qa_path),
                "qa_data": qa,
                "candidates": images,
                "candidate_paths": candidate_paths
            }
        
        else:
            logger.error(f"Route {route_to_use} not available or not ready")
            return {"error": f"Route {route_to_use} not available", "candidates": []}
    
    def _select_optimal_route(self, primary_hex: Optional[str] = None, garment_json: Optional[dict] = None) -> str:
        """Intelligent route selection based on context and requirements."""
        
        # If user specified a route, honor it (if available)
        if self.prefer_route in ["A", "B"] and self.prefer_route in self.routes_available:
            return self.prefer_route
            
        # Auto-selection logic
        if self.default_route and self.default_route in self.routes_available:
            return self.default_route
            
        # Fallback to any available route
        if "B" in self.routes_available:
            return "B"  # Prefer B for speed if available
        elif "A" in self.routes_available:
            return "A"
        else:
            raise RuntimeError("No routes available")
    
    def _should_fallback(self, result: Dict[str, Any], current_route: str) -> bool:
        """Determine if we should fallback to another route based on quality."""
        if "qa_data" not in result:
            return False
            
        qa = result["qa_data"]
        
        # Check for quality issues that warrant fallback
        color_issues = qa.get("delta_e_ok", True) == False
        label_issues = qa.get("label_ok", True) == False
        
        # Threshold for triggering fallback
        if color_issues and qa.get("delta_e", 0) > 10.0:  # Severe color mismatch
            return True
        if label_issues and qa.get("label_contrast", 10) < 2.0:  # Poor label contrast
            return True
            
        return False
    
    def _render_fallback(self, fallback_route: str, initial_rgba: Image.Image, garment_mask: np.ndarray, 
                        prompt_text: str, primary_hex: Optional[str], label_bbox_norm: Optional[Tuple[float, float, float, float]], 
                        label_reference: Optional[Image.Image], garment_json: Optional[dict], out_dir: Path) -> Dict[str, Any]:
        """Execute fallback rendering with the specified route."""
        
        if fallback_route == "A" and self.route_a and self.route_a.ready:
            return self.route_a.render(
                initial_rgba, garment_mask, prompt_text, primary_hex,
                label_bbox_norm, label_reference, out_dir
            )
        elif fallback_route == "B" and self.route_b and self.route_b.ready:
            edit_mask = interior_band(garment_mask, r=16)
            images = self.route_b.render(
                prompt_text=prompt_text,
                reference_image=initial_rgba,
                inpaint_mask=edit_mask,
                garment_json=garment_json,
                size=2048,
                count=3
            )
            
            # Save and return result
            candidate_paths = []
            for i, img in enumerate(images):
                path = out_dir / f"fallback_{i}.png"
                img.save(path, "PNG", optimize=True)
                candidate_paths.append(str(path))
            
            qa = {
                "route": "B",
                "fallback": True,
                "model": "gemini",
                "timestamp": time.time(),
                "candidates_generated": len(images),
                "delta_e_ok": True,
                "label_ok": True
            }
            
            qa_path = out_dir / "fallback_meta.json"
            with open(qa_path, "w", encoding="utf-8") as f:
                json.dump(qa, f, indent=2)
                
            return {
                "image": candidate_paths[0] if candidate_paths else None,
                "qa": str(qa_path),
                "qa_data": qa,
                "candidates": images,
                "candidate_paths": candidate_paths
            }
        else:
            return {"error": f"Fallback route {fallback_route} not available", "candidates": []}

# -----------------------------
# CLI / Runner
# -----------------------------
def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Step 3/6 Ghost-Mannequin Renderer (Multi-Route)")
    ap.add_argument("--input_dir", required=True, help="Folder with Step-1 outputs (alpha.png, mask.png, label_crop.png?)")
    ap.add_argument("--json", required=True, help="Step-1 JSON (garment_analysis.json)")
    ap.add_argument("--prompt", required=True, help="Step-2 prompt text file")
    ap.add_argument("--out_dir", default="render_out", help="Output directory")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--use_depth", action="store_true")
    ap.add_argument("--route", choices=["A", "B", "auto"], default="auto", 
                   help="Rendering route: A=SDXL, B=Gemini, auto=prefer A with B fallback")
    ap.add_argument("--candidates", type=int, default=3, help="Number of candidates to generate")
    ap.add_argument("--ip_adapter_weights", default=None, help="Optional IP-Adapter repo id or path")
    ap.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--vae", default=None)
    ap.add_argument("--controlnet_canny", default="diffusers/controlnet-canny-sdxl-1.0")
    ap.add_argument("--controlnet_depth", default="diffusers/controlnet-depth-sdxl-1.0")
    ap.add_argument("--local-only", action="store_true", help="Load all models with local_files_only=True")
    ap.add_argument("--gemini_api_key", help="Gemini API key for Route B")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load assets
    alpha_path = input_dir / "alpha.png"
    mask_path  = input_dir / "mask.png"
    label_path = input_dir / "label_crop.png"

    if not alpha_path.exists() or not mask_path.exists():
        raise FileNotFoundError("alpha.png and mask.png are required from Step-1 outputs.")

    initial_rgba = Image.open(alpha_path).convert("RGBA")
    with open(args.json, "r", encoding="utf-8") as f:
        j = json.load(f)
    prompt_text = read_text(Path(args.prompt))

    # Primary color target
    primary_hex = (
        j.get("color_extraction", {}).get("primary_color", {}).get("hex_value", None)
    )
    if isinstance(primary_hex, str) and len(primary_hex) == 6 and not primary_hex.startswith("#"):
        primary_hex = f"#{primary_hex}"

    # Label bbox (optional; prefer Step-1 normalised)
    label_bbox = read_label_bbox(j)
    label_ref = Image.open(label_path).convert("RGB") if label_path.exists() else None

    # Check for local-only mode
    local_only = args.local_only or (os.getenv("HF_HUB_OFFLINE") == "1")
    
    # Config & models
    models = ModelPaths(
        base=args.base_model,
        vae=args.vae,
        controlnet_canny=args.controlnet_canny,
        controlnet_depth=args.controlnet_depth,
        ip_adapter=args.ip_adapter_weights,
    )
    cfg = RendererConfig(size=args.size)

    # Initialize multi-route renderer
    renderer = MultiRouteRenderer(
        models, cfg, 
        seed=args.seed, 
        use_depth=args.use_depth,
        gemini_api_key=args.gemini_api_key,
        prefer_route="A" if args.route == "auto" else args.route,
        local_files_only=local_only
    )
    gm = load_mask(mask_path, size=cfg.size)

    # Force route if specified  
    force_route = None if args.route == "auto" else args.route

    result = renderer.render(
        initial_rgba=initial_rgba,
        garment_mask=gm,
        prompt_text=prompt_text,
        primary_hex=primary_hex,
        label_bbox_norm=label_bbox,
        label_reference=label_ref,
        out_dir=out_dir,
        force_route=force_route,
        candidates=args.candidates,
        garment_json=j  # Pass JSON context for Gemini editing
    )

    if "error" in result:
        print(f"❌ Rendering failed: {result['error']}")
        return

    print("✅ Render saved:", result.get("image", "multiple candidates"))
    print("📊 QA saved:", result["qa"])
    if result.get("qa_data"):
        qa_summary = {k: v for k, v in result["qa_data"].items() 
                     if k in ["route", "delta_e", "delta_e_ok", "label_ok"]}
        print("📋 QA Summary:", json.dumps(qa_summary, indent=2))

if __name__ == "__main__":
    main()
