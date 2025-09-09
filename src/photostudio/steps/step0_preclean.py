#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step0_preclean.py
Step 0/6 — Background removal + Matting + Openings + Control image.

Outputs (to --out):
  - alpha.png            (RGBA, extracted garment with refined alpha)
  - mask.png             (binary L, 0/255)
  - mask_soft.png        (soft alpha 0..255)
  - openings.png         (binary L: interior openings/holes)
  - openings_band.png    (thin band around openings for ghost shading)
  - control_rgb.png      (garment composited on pure white, square)
  - step0_report.json    (metrics + provenance)

Designed to feed Step 1–3:
  Step 1 (analysis) can trust mask.png / openings.png
  Step 2 uses colors, but now quality starts cleaner
  Step 3 can use control_rgb.png + openings_band.png for inpaint

CLI:
  python step0_preclean.py \
     --image ./raw/IMG_8208.jpeg \
     --out ./step0 \
     --target-size 2048 \
     --backend auto \
     --save-intermediates
"""

from __future__ import annotations
import argparse, json, math, os, sys, time, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
from PIL import Image, ImageOps

import cv2

# ---------- Optional heavy deps (graceful fallbacks) ----------
_HAVE_SAM2 = False
_HAVE_REMBG = False
_HAVE_RVM   = False

# Try SAM2
try:
    # Depending on your environment this import could be e.g.:
    #   from sam2.build_sam import sam_model_registry
    #   from sam2.automatic_mask_generator import SamAutomaticMaskGenerator
    # or a thin wrapper. We just flag availability and call through a helper.
    import sam2  # noqa: F401
    _HAVE_SAM2 = True
except Exception:
    _HAVE_SAM2 = False

# Try rembg
try:
    from rembg import remove as rembg_remove, new_session as rembg_session
    _HAVE_REMBG = True
except Exception:
    _HAVE_REMBG = False

# Optional matting (Refine alpha with Robust Video Matting)
try:
    import torch  # noqa: F401
    import rvm  # noqa: F401  # robustvideomatting package registers module
    _HAVE_RVM = True
except Exception:
    _HAVE_RVM = False

# ---------- Small utils ----------
def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _sha16(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()[:16]

def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)

def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img

def _exif_correct(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img

def _resize_square_keep(img: Image.Image, size: int, fill=(255,255,255,0)) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGBA" if img.mode == "RGBA" else "RGB", (s, s), fill)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    if s != size:
        canvas = canvas.resize((size, size), Image.LANCZOS)
    return canvas

def _binarize(a: np.ndarray, thr: int=128) -> np.ndarray:
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    return (a >= thr).astype(np.uint8) * 255

def _soft_to_binary(alpha: np.ndarray, min_area_px: int = 500) -> np.ndarray:
    g = alpha.astype(np.uint8)
    # Otsu then clean
    thr, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Remove tiny specks
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(bw)
    for c in cnts:
        if cv2.contourArea(c) >= min_area_px:
            cv2.drawContours(keep, [c], -1, 255, -1)
    # Close small holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k, iterations=2)
    return keep

def _openings_from_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (openings, openings_band) where openings are interior 'holes'
    of the garment region, and openings_band is a thin ring around them.
    """
    m = (mask > 0).astype(np.uint8) * 255
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = np.zeros_like(m)
    if hier is not None:
        for i, c in enumerate(cnts):
            # holes are contours with parent >= 0
            if hier[0][i][3] >= 0 and cv2.contourArea(c) > 60:
                cv2.drawContours(holes, [c], -1, 255, -1)
    # Thin band (for ghost shading)
    r = max(6, round(0.004 * max(m.shape)))  # ~0.4% of canvas
    dil = cv2.dilate(holes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
    band = cv2.subtract(dil, holes)
    return holes, band

def safe_grabcut_refine(bgr_u8: np.ndarray, mask_u8: np.ndarray, iters: int = 5) -> np.ndarray:
    """
    Robust GrabCut refinement with proper seed generation and fallbacks.
    
    Args:
        bgr_u8: HxWx3 BGR uint8 image (cv2 format)
        mask_u8: HxW uint8 {0,255} (garment=255)
        iters: GrabCut iterations
        
    Returns: refined binary mask {0,255}
    """
    h, w = mask_u8.shape
    
    # Build seeds with proper erosion/dilation
    kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    sure_fg = cv2.erode(mask_u8, kernel_fg, iterations=1)           # confident interior
    dil = cv2.dilate(mask_u8, kernel_bg, iterations=1)
    sure_bg = (255 - np.clip(dil, 0, 255)).astype(np.uint8)         # confident exterior

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)                   # default = probable background
    gc[sure_bg > 0] = cv2.GC_BGD
    gc[sure_fg > 0] = cv2.GC_FGD

    # Seed sanity: must have both BG and FG
    if not (np.any(gc == cv2.GC_BGD) and np.any(gc == cv2.GC_FGD)):
        # Fallback A: rectangle init on padded bbox
        ys, xs = np.where(mask_u8 > 0)
        if ys.size == 0:  # degenerate—just return input
            return (mask_u8 > 0).astype(np.uint8) * 255
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        pad = int(0.05 * max(w, h))
        rect = (max(0, x0 - pad), max(0, y0 - pad),
                min(w - 1, x1 + pad) - max(0, x0 - pad),
                min(h - 1, y1 + pad) - max(0, y0 - pad))
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(bgr_u8, gc, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
        except Exception:
            # Fallback B: return input mask unchanged
            return (mask_u8 > 0).astype(np.uint8) * 255
    else:
        # Normal mask-init path
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(bgr_u8, gc, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)
        except Exception:
            return (mask_u8 > 0).astype(np.uint8) * 255

    refined = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    # Guard against degenerate results
    coverage = np.count_nonzero(refined) / (h * w)
    if coverage < 0.01 or coverage > 0.95:
        # Revert to input mask if result is degenerate
        return (mask_u8 > 0).astype(np.uint8) * 255
    
    return refined

def _guided_refine_alpha(rgb: np.ndarray, a_soft: np.ndarray) -> np.ndarray:
    """
    Fast guided-refinement fallback (OpenCV). If ximgproc not present, do bilateral smooth.
    """
    try:
        import cv2.ximgproc as xip  # type: ignore
        guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        a = a_soft.astype(np.float32) / 255.0
        refined = xip.guidedFilter(guide=guide, src=a, radius=16, eps=1e-4)
        return _to_uint8(refined * 255.0)
    except Exception:
        # Bilateral on alpha edges
        a = a_soft.astype(np.uint8)
        refined = cv2.bilateralFilter(a, d=9, sigmaColor=40, sigmaSpace=40)
        return refined

# ---------- Foreground extractors ----------
def _segment_sam2(rgb: Image.Image) -> Optional[np.ndarray]:
    """
    Placeholder SAM2 path: return soft alpha (0..255) or None if SAM2 not configured.
    Implement your in-house SAM2 mask generator here if you have a wrapper.
    """
    if not _HAVE_SAM2:
        return None
    try:
        # Pseudocode — adjust to your SAM2 API
        # from sam2.automatic_mask_generator import SamAutomaticMaskGenerator
        # from sam2.build_sam import sam_model_registry
        # model = sam_model_registry["vit_h"](checkpoint="sam2_h.pth")
        # model.to(device="cuda" if torch.cuda.is_available() else "cpu")
        # amg = SamAutomaticMaskGenerator(model, points_per_side=48, pred_iou_thresh=0.86)
        # arr = np.array(rgb.convert("RGB"))
        # masks = amg.generate(arr)
        # largest = max(masks, key=lambda m: m["area"])
        # soft = (largest["segmentation"].astype(np.float32) * 255.0)
        # return soft.astype(np.uint8)
        return None  # remove once you wire your SAM2
    except Exception:
        return None

def _segment_rembg(rgb: Image.Image) -> Optional[np.ndarray]:
    if not _HAVE_REMBG:
        return None
    try:
        sess = rembg_session("isnet-general-use")  # high-quality portrait/general
    except Exception:
        sess = None
    try:
        # Use alpha matting for better edges
        out_bytes = rembg_remove(
            rgb, session=sess,
            alpha_matting=True,
            alpha_matting_erode_size=10,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            post_process_mask=True
        )
        # rembg returns bytes (PNG RGBA)
        from io import BytesIO
        pil = Image.open(BytesIO(out_bytes)).convert("RGBA")
        a = np.array(pil)[..., 3]
        return a
    except Exception:
        return None

def _refine_with_rvm(rgb: np.ndarray, a_soft: np.ndarray) -> np.ndarray:
    """
    Optional RVM refine (if installed). We use the alpha as trimap prior.
    """
    if not _HAVE_RVM:
        return a_soft
    try:
        # Lightweight single-frame refine: RVM expects video, but we can feed one frame.
        # If your environment uses a different API, adapt here.
        return a_soft  # keep as placeholder unless you wire your RVM call
    except Exception:
        return a_soft

# ---------- Core pipeline ----------
@dataclass
class Step0Config:
    target_size: int = 2048
    backend: str = "auto"  # "sam2" | "rembg" | "auto"
    save_intermediates: bool = False
    use_grabcut_refine: bool = True  # Enable robust GrabCut refinement

def foreground_extract(img: Image.Image, cfg: Step0Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (soft_alpha 0..255, binary_mask 0/255).
    """
    rgb = _ensure_rgb(img).convert("RGB")
    a_soft: Optional[np.ndarray] = None

    use_backend = cfg.backend
    if use_backend == "auto":
        # Try SAM2 first if available
        if _HAVE_SAM2:
            a_soft = _segment_sam2(rgb)
        if a_soft is None:
            a_soft = _segment_rembg(rgb)
    elif use_backend == "sam2":
        a_soft = _segment_sam2(rgb)
        if a_soft is None:
            a_soft = _segment_rembg(rgb)
    else:
        a_soft = _segment_rembg(rgb)

    if a_soft is None:
        # Very last resort: grabcut around center
        arr = np.array(rgb)
        h, w = arr.shape[:2]
        rect = (int(w*0.15), int(h*0.1), int(w*0.7), int(h*0.8))
        mask = np.zeros((h,w), np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        cv2.grabCut(arr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        a_soft = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Optional robust GrabCut refinement (fixes segmentation issues)
    if cfg.use_grabcut_refine:
        rgb_bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        initial_mask = _soft_to_binary(a_soft, min_area_px=800)
        initial_coverage = np.count_nonzero(initial_mask) / (initial_mask.shape[0] * initial_mask.shape[1])
        
        refined_mask = safe_grabcut_refine(rgb_bgr, initial_mask, iters=3)
        refined_coverage = np.count_nonzero(refined_mask) / (refined_mask.shape[0] * refined_mask.shape[1])
        
        print(f"GrabCut refinement: {initial_coverage:.3f} → {refined_coverage:.3f} coverage")
        
        # Use refined mask for alpha processing
        a_soft = refined_mask.astype(np.float32)
    
    # Alpha refine
    a_soft = _refine_with_rvm(np.array(rgb), a_soft)
    a_soft = _guided_refine_alpha(np.array(rgb), a_soft)

    # Final binary mask
    m_bin = _soft_to_binary(a_soft, min_area_px=800)
    return a_soft, m_bin

def compose_on_white(rgb: Image.Image, a_soft: np.ndarray) -> Image.Image:
    arr = np.array(rgb.convert("RGB"), dtype=np.uint8)
    A = (a_soft.astype(np.float32) / 255.0)[..., None]
    out = (arr.astype(np.float32) * A + 255.0 * (1.0 - A)).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")

def rgba_from_alpha(rgb: Image.Image, a_soft: np.ndarray) -> Image.Image:
    arr = np.array(rgb.convert("RGB"), dtype=np.uint8)
    rgba = np.dstack([arr, a_soft.astype(np.uint8)])
    return Image.fromarray(rgba, mode="RGBA")

def compute_metrics(mask: np.ndarray) -> Dict[str, float]:
    h, w = mask.shape
    area = float(np.count_nonzero(mask))
    coverage = area / float(h*w)
    # boundary smoothness proxy
    edges = cv2.Canny(mask, 50, 150)
    edge_len = float(np.count_nonzero(edges))
    smoothness = 1.0 - min(1.0, edge_len / (h*w*0.12 + 1e-6))
    # holes
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hier is not None:
        for i, c in enumerate(cnts):
            if hier[0][i][3] >= 0 and cv2.contourArea(c) > 60: holes += 1
    return {"mask_coverage": coverage, "boundary_smoothness": smoothness, "hole_count": float(holes)}

def main():
    ap = argparse.ArgumentParser(description="Step 0/6 — Pre-clean (BG removal + Matting + Openings)")
    ap.add_argument("--image", required=True, help="Input image (JPEG/PNG)")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--target-size", type=int, default=2048)
    ap.add_argument("--backend", choices=["auto","sam2","rembg"], default="auto")
    ap.add_argument("--save-intermediates", action="store_true")
    ap.add_argument("--no-grabcut-refine", action="store_true", help="Disable GrabCut refinement")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Step0Config(
        target_size=args.target_size, 
        backend=args.backend, 
        save_intermediates=args.save_intermediates,
        use_grabcut_refine=not args.no_grabcut_refine
    )

    # Load + EXIF
    src = Image.open(args.image)
    src = _exif_correct(src)

    # Normalize canvas now (square)
    src_sq = _resize_square_keep(src, cfg.target_size, fill=(255,255,255))
    rgb_sq = _ensure_rgb(src_sq).convert("RGB")

    # Foreground extraction
    a_soft, m_bin = foreground_extract(rgb_sq, cfg)

    # Openings
    openings, band = _openings_from_mask(m_bin)

    # RGBA + control rgb
    alpha_rgba = rgba_from_alpha(rgb_sq, a_soft)
    control_rgb = compose_on_white(rgb_sq, a_soft)

    # Saves
    alpha_path   = out_dir / "alpha.png"
    mask_path    = out_dir / "mask.png"
    mask_soft    = out_dir / "mask_soft.png"
    openings_p   = out_dir / "openings.png"
    openings_b   = out_dir / "openings_band.png"
    control_path = out_dir / "control_rgb.png"

    alpha_rgba.save(alpha_path, "PNG", optimize=True)
    Image.fromarray(m_bin, mode="L").save(mask_path, "PNG", optimize=True)
    Image.fromarray(a_soft, mode="L").save(mask_soft, "PNG", optimize=True)
    Image.fromarray(openings, mode="L").save(openings_p, "PNG", optimize=True)
    Image.fromarray(band, mode="L").save(openings_b, "PNG", optimize=True)
    control_rgb.save(control_path, "PNG", optimize=True)

    # Report
    met = compute_metrics(m_bin)
    rep = {
        "created_at": _now_iso(),
        "source": str(Path(args.image).resolve()),
        "target_size": cfg.target_size,
        "backend_selected": args.backend,
        "env": {
            "sam2_available": bool(_HAVE_SAM2),
            "rembg_available": bool(_HAVE_REMBG),
            "rvm_available": bool(_HAVE_RVM),
        },
        "outputs": {
            "alpha": str(alpha_path),
            "mask": str(mask_path),
            "mask_soft": str(mask_soft),
            "openings": str(openings_p),
            "openings_band": str(openings_b),
            "control_rgb": str(control_path),
        },
        "metrics": met
    }
    (out_dir / "step0_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    # Optional debug
    if cfg.save_intermediates:
        dbg = np.dstack([np.array(control_rgb), np.full_like(m_bin, 255)])
        # overlay openings band in red
        ov = np.array(control_rgb).copy()
        ov[band > 0] = [255, 0, 0]
        Image.fromarray(ov).save(out_dir / "debug_openings_overlay.png", "PNG", optimize=True)

    print("✅ Step 0 complete.")
    print(f"alpha:         {alpha_path}")
    print(f"mask:          {mask_path}")
    print(f"openings band: {openings_b}")
    print(f"control_rgb:   {control_path}")
    print(f"report:        {out_dir/'step0_report.json'}")

if __name__ == "__main__":
    main()
