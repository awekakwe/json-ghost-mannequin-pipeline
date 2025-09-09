#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_color_refine.py
Refined color extraction per region using Step 0 masks:
- Uses alpha/mask/openings from ./step0
- Samples ONLY safe garment pixels (eroded core), excludes labels/hardware
- Separates OUTER SHELL vs LINING (around openings) vs BORDER/RIBBING band
- Robust LAB stats + KMeans palette (optional) + glare/outlier filtering
- Updates garment_analysis.json color_extraction block in-place

Usage:
  python step1_color_refine.py \
    --step0 ./step0 \
    --json-in  ./step1/garment_analysis.json \
    --json-out ./step1/garment_analysis.json
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
from PIL import Image
import cv2
from skimage import color as skcolor

# ----------------- I/O -----------------
def read_img_gray(p: Path) -> np.ndarray:
    g = Image.open(p).convert("L")
    return np.array(g)

def read_img_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))

def write_json(p: Path, d: dict):
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")

# ----------------- utils -----------------
def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)

def erode(mask: np.ndarray, r: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    return cv2.erode(mask, k)

def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    return cv2.dilate(mask, k)

def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask > 0))

def rgb_to_hex(rgb_u8: np.ndarray) -> str:
    r, g, b = [int(x) for x in rgb_u8]
    return f"#{r:02x}{g:02x}{b:02x}"

def robust_mean_rgb(rgb: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Robust mean in LAB-space with outlier & glare filtering, returns RGB uint8 and stats.
    """
    sel = (m > 0)
    if not np.any(sel):
        return np.array([255, 255, 255], np.uint8), {"n": 0}

    pix = rgb[sel] / 255.0
    lab = skcolor.rgb2lab(pix.reshape(-1, 1, 3)).reshape(-1, 3)

    # Remove extreme L* highlights/shadows + chroma outliers
    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]
    p1, p99 = np.percentile(L, [1, 99])
    keep = (L >= p1) & (L <= p99)
    lab = lab[keep]
    if lab.shape[0] < 50:
        lab = skcolor.rgb2lab(pix.reshape(-1,1,3)).reshape(-1,3)  # fallback

    # Median center & MAD filter
    med = np.median(lab, axis=0)
    mad = np.median(np.abs(lab - med), axis=0) + 1e-6
    z = np.abs(lab - med) / mad
    lab_f = lab[np.all(z < 6.0, axis=1)]  # keep within 6 MAD

    lab_mean = np.mean(lab_f, axis=0)
    rgb_mean = skcolor.lab2rgb(lab_mean.reshape(1,1,3)).reshape(3)
    rgb_u8 = to_uint8(rgb_mean * 255.0)

    # simple spread measure
    if lab_f.shape[0] > 1:
        dE = skcolor.deltaE_ciede2000(
            lab_f.reshape(-1, 1, 3), 
            lab_mean.reshape(1, 1, 3)
        ).ravel()
    else:
        dE = np.array([0.0])
    
    stats = {
        "n": int(lab_f.shape[0]),
        "mean_Lab": [float(x) for x in lab_mean],
        "p95_delta_e": float(np.percentile(dE, 95)) if len(dE) > 0 else 0.0,
        "max_delta_e": float(np.max(dE)) if len(dE) > 0 else 0.0,
    }
    return rgb_u8, stats

def kmeans_palette(rgb: np.ndarray, m: np.ndarray, k: int = 3) -> List[Dict]:
    sel = (m > 0)
    if np.count_nonzero(sel) < 200:
        return []
    pts = rgb[sel].reshape(-1, 3).astype(np.float32)

    # KMeans in LAB for perceptual separation
    lab = skcolor.rgb2lab((pts/255.0).reshape(-1,1,3)).reshape(-1,3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    try:
        compactness, labels, centers = cv2.kmeans(lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    except Exception:
        return []

    out = []
    for i in range(k):
        idx = (labels.ravel() == i)
        share = float(np.mean(idx))
        c_lab = centers[i].astype(np.float64)
        c_rgb = skcolor.lab2rgb(c_lab.reshape(1,1,3)).reshape(3)
        c_rgb_u8 = to_uint8(c_rgb * 255.0)
        out.append({
            "hex_value": rgb_to_hex(c_rgb_u8),
            "rgb_values": [int(x) for x in c_rgb_u8],
            "percentage": round(100.0 * share, 2)
        })
    # sort by share desc
    out.sort(key=lambda d: d["percentage"], reverse=True)
    return out

# ----------------- region masks -----------------
def build_region_masks(mask: np.ndarray, openings: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Produces:
      - shell_safe: eroded interior (no edges)
      - lining_band: thin ring just inside openings (for inside color)
      - border_band: garment border (for ribbing/cuffs/hem estimation)
    """
    h, w = mask.shape
    r_core = max(4, round(0.02 * max(h, w)))        # ~2% shrink
    r_border = max(6, round(0.03 * max(h, w)))      # ~3% band

    m = (mask > 0).astype(np.uint8) * 255
    core = erode(m, r_core)
    shell_safe = core.copy()

    # Lining: a small dilate of openings but stay inside garment
    open_in = (openings > 0).astype(np.uint8) * 255
    if np.any(open_in):
        lin = dilate(open_in, max(3, r_core//2))
        lining_band = cv2.bitwise_and(lin, m)
    else:
        lining_band = np.zeros_like(m)

    # Border band (outer shell rim): (mask - core)
    border = cv2.subtract(m, core)
    # keep only thicker rim
    border_band = erode(border, max(2, r_core//2))

    return {
        "shell_safe": shell_safe,
        "lining_band": lining_band,
        "border_band": border_band
    }

# ----------------- main logic -----------------
def main():
    ap = argparse.ArgumentParser(description="Step 1 color refine per region")
    ap.add_argument("--step0", required=True, help="Folder with alpha.png, mask.png, openings.png")
    ap.add_argument("--json-in", required=True)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()

    step0 = Path(args.step0)
    rgb = read_img_rgb(step0/"control_rgb.png") if (step0/"control_rgb.png").exists() else read_img_rgb(step0/"alpha.png")
    mask = read_img_gray(step0/"mask.png")
    openings = read_img_gray(step0/"openings.png") if (step0/"openings.png").exists() else np.zeros_like(mask)

    regions = build_region_masks(mask, openings)

    result: Dict[str, Dict] = {"regions": {}}

    # Outer shell
    shell_rgb, shell_stats = robust_mean_rgb(rgb, regions["shell_safe"])
    result["regions"]["outer_shell"] = {
        "hex_value": rgb_to_hex(shell_rgb),
        "rgb_values": [int(x) for x in shell_rgb],
        "stats": shell_stats
    }

    # Lining (if visible)
    if mask_area(regions["lining_band"]) > 150:
        lining_rgb, lining_stats = robust_mean_rgb(rgb, regions["lining_band"])
        result["regions"]["lining"] = {
            "hex_value": rgb_to_hex(lining_rgb),
            "rgb_values": [int(x) for x in lining_rgb],
            "stats": lining_stats
        }

    # Border (ribbing/hem cuffs proxy)
    if mask_area(regions["border_band"]) > 150:
        border_rgb, border_stats = robust_mean_rgb(rgb, regions["border_band"])
        result["regions"]["border"] = {
            "hex_value": rgb_to_hex(border_rgb),
            "rgb_values": [int(x) for x in border_rgb],
            "stats": border_stats
        }

    # Palette on shell for secondary/pattern colors
    palette = kmeans_palette(rgb, regions["shell_safe"], k=3)
    result["primary_color"] = result["regions"]["outer_shell"]
    result["secondary_colors"] = palette

    # load json, update block
    jpath_in = Path(args.json_in)
    data = json.loads(jpath_in.read_text(encoding="utf-8"))
    data.setdefault("color_extraction", {})
    data["color_extraction"].update({
        "primary_color": {
            "hex_value": result["primary_color"]["hex_value"],
            "rgb_values": result["primary_color"]["rgb_values"],
            "color_name": None  # optional: add a small name map if you want
        },
        "secondary_colors": palette,
        "per_region_colors": result["regions"]
    })
    # write out
    write_json(Path(args.json_out), data)

    print("✅ color refine complete")
    print(" primary:", result['primary_color']['hex_value'])
    if "lining" in result["regions"]:
        print(" lining :", result['regions']['lining']['hex_value'])
    if "border" in result["regions"]:
        print(" border :", result['regions']['border']['hex_value'])

if __name__ == "__main__":
    main()
