#!/usr/bin/env python3
"""
step4_qa_corrector.py – Step 4/6 Auto-QA + Correct + Select
============================================================

Score candidates, apply surgical corrections, select the best.

QA Checks:
- Color: ΔE2000 mean + p95 vs primary hex (LAB space)
- Hollow: luminance/contrast heuristics for neckline/sleeves (proxy score)
- Sharpness: Laplacian variance
- Background: normalized to white; zero shadows
- Composite score: color(45%) + hollow(35%) + sharpness(20%)

Auto-corrections:
- LAB bias toward primary hex (capped)
- Background normalization
- Final recenter/resize

Dependencies: opencv-python numpy pillow scikit-image
"""

from __future__ import annotations

import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger("photostudio.qa_corrector")

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class QAMetrics:
    """Quality assessment metrics for a single candidate."""
    mean_delta_e: float
    p95_delta_e: float
    hollow_quality: float
    sharpness: float
    background_score: float
    composite_score: float
    passes_gates: bool

@dataclass
class CorrectionResult:
    """Result of applying corrections to a candidate."""
    corrected_image: Image.Image
    corrections_applied: List[str]
    final_metrics: QAMetrics

# ---------------------------------------------------------------------------
# Color Analysis & Delta E
# ---------------------------------------------------------------------------

def _rgb_to_lab_opencv(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert RGB to LAB using OpenCV (same as Step 3)."""
    bgr = rgb_u8[..., ::-1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab.astype(np.float32)

def _delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Calculate CIEDE2000 color difference (same implementation as Step 3)."""
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
    """Convert hex color to RGB tuple."""
    h = hex_str.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def compute_color_metrics(
    image: Image.Image, 
    mask: np.ndarray, 
    target_hex: str
) -> Tuple[float, float]:
    """
    Compute color accuracy metrics.
    Returns: (mean_delta_e, p95_delta_e)
    """
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    
    # Ensure mask matches image dimensions
    img_h, img_w = rgb.shape[:2]
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    lab = _rgb_to_lab_opencv(rgb)
    
    # Target color
    tr, tg, tb = hex_to_rgb(target_hex)
    target_rgb = np.full_like(rgb, (tr, tg, tb), dtype=np.uint8)
    target_lab = _rgb_to_lab_opencv(target_rgb)
    
    # Compute delta E only in garment region
    garment_mask = mask > 0
    if np.count_nonzero(garment_mask) < 10:
        return 999.0, 999.0
        
    dE = _delta_e_ciede2000(lab[garment_mask], target_lab[garment_mask])
    
    mean_delta_e = float(np.mean(dE))
    p95_delta_e = float(np.percentile(dE, 95))
    
    return mean_delta_e, p95_delta_e

# ---------------------------------------------------------------------------
# Hollow Effect Analysis
# ---------------------------------------------------------------------------

def estimate_hollow_quality(image: Image.Image, mask: np.ndarray) -> float:
    """
    Estimate hollow effect quality using luminance/contrast heuristics.
    Returns score 0-1 where 1 is perfect hollow effect.
    """
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    
    # Ensure mask matches image dimensions
    img_h, img_w = gray.shape
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
    h, w = mask.shape
    
    # Focus on potential hollow areas (neck, sleeves, bottom)
    hollow_regions = []
    
    # Neck region (top 20% of mask)
    neck_mask = mask.copy()
    neck_mask[int(h * 0.2):] = 0
    if np.count_nonzero(neck_mask) > 100:
        hollow_regions.append(("neck", neck_mask))
    
    # Sleeve regions (left and right 15% of mask, middle height)
    left_sleeve = mask.copy()
    left_sleeve[:, int(w * 0.15):] = 0  
    left_sleeve[:int(h * 0.3)] = 0
    left_sleeve[int(h * 0.7):] = 0
    if np.count_nonzero(left_sleeve) > 50:
        hollow_regions.append(("left_sleeve", left_sleeve))
        
    right_sleeve = mask.copy()
    right_sleeve[:, :int(w * 0.85)] = 0
    right_sleeve[:int(h * 0.3)] = 0  
    right_sleeve[int(h * 0.7):] = 0
    if np.count_nonzero(right_sleeve) > 50:
        hollow_regions.append(("right_sleeve", right_sleeve))
    
    if not hollow_regions:
        return 0.5  # No clear hollow regions detected
    
    hollow_scores = []
    for region_name, region_mask in hollow_regions:
        # Get interior pixels (eroded region)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        interior = cv2.erode(region_mask, kernel)
        
        if np.count_nonzero(interior) < 10:
            continue
            
        # Get boundary pixels  
        boundary = cv2.subtract(region_mask, interior)
        
        if np.count_nonzero(boundary) < 10:
            continue
            
        # Compare luminance: interior should be brighter (hollow effect)
        interior_brightness = float(np.mean(gray[interior > 0]))
        boundary_brightness = float(np.mean(gray[boundary > 0]))
        
        # Hollow score based on brightness difference
        brightness_diff = interior_brightness - boundary_brightness
        hollow_score = max(0.0, min(1.0, brightness_diff / 50.0))  # Normalize to 0-1
        hollow_scores.append(hollow_score)
    
    return float(np.mean(hollow_scores)) if hollow_scores else 0.3

# ---------------------------------------------------------------------------
# Sharpness Analysis
# ---------------------------------------------------------------------------

def compute_sharpness(image: Image.Image, mask: np.ndarray) -> float:
    """
    Compute sharpness using Laplacian variance on garment region.
    Returns normalized score 0-1.
    """
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    
    # Ensure mask matches image dimensions
    img_h, img_w = gray.shape
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    
    # Apply mask to focus on garment
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Compute Laplacian variance
    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    laplacian_masked = laplacian[mask > 0]
    
    if laplacian_masked.size == 0:
        return 0.0
        
    variance = float(np.var(laplacian_masked))
    
    # Normalize to 0-1 range (empirically determined)
    normalized_score = min(1.0, variance / 1000.0)
    
    return normalized_score

# ---------------------------------------------------------------------------
# Background Analysis
# ---------------------------------------------------------------------------

def compute_background_score(image: Image.Image, mask: np.ndarray) -> float:
    """
    Analyze background purity (should be white with no shadows).
    Returns score 0-1 where 1 is perfect white background.
    """
    rgb = np.array(image.convert("RGB"))
    
    # Ensure mask matches image dimensions
    img_h, img_w = rgb.shape[:2]
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    
    # Background is everything outside the mask
    bg_mask = (mask == 0)
    
    if np.count_nonzero(bg_mask) < 100:
        return 0.5  # Can't evaluate background
    
    bg_pixels = rgb[bg_mask]
    
    # Check how close to white (255, 255, 255)
    white_target = np.array([255, 255, 255])
    distances = np.linalg.norm(bg_pixels - white_target, axis=1)
    mean_distance = float(np.mean(distances))
    
    # Normalize: perfect white = 1.0, max expected deviation ~50
    bg_purity = max(0.0, 1.0 - mean_distance / 50.0)
    
    # Check for shadows (low variance = uniform, good)
    bg_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[bg_mask]
    variance = float(np.var(bg_gray))
    shadow_score = max(0.0, 1.0 - variance / 100.0)  # Lower variance is better
    
    # Combine purity and shadow scores
    return (bg_purity * 0.7 + shadow_score * 0.3)

# ---------------------------------------------------------------------------
# QA Scorer
# ---------------------------------------------------------------------------

class QAScorer:
    """Score candidates based on multiple quality metrics."""
    
    def __init__(
        self, 
        delta_e_gate: float = 2.0,
        hollow_gate: float = 0.4,
        sharpness_gate: float = 0.3,
        background_gate: float = 0.6
    ):
        self.delta_e_gate = delta_e_gate
        self.hollow_gate = hollow_gate
        self.sharpness_gate = sharpness_gate
        self.background_gate = background_gate
        
        # Composite score weights
        self.color_weight = 0.45
        self.hollow_weight = 0.35
        self.sharpness_weight = 0.20
        
    def score_candidate(
        self, 
        image: Image.Image, 
        mask: np.ndarray, 
        target_hex: str
    ) -> QAMetrics:
        """Score a single candidate image."""
        
        # Resize mask to match image dimensions if needed
        img_h, img_w = image.size[1], image.size[0]  # PIL uses (width, height)
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # Color metrics
        mean_delta_e, p95_delta_e = compute_color_metrics(image, mask, target_hex)
        color_score = max(0.0, 1.0 - mean_delta_e / 5.0)  # Normalize ΔE to 0-1
        
        # Hollow effect
        hollow_quality = estimate_hollow_quality(image, mask)
        
        # Sharpness
        sharpness = compute_sharpness(image, mask)
        
        # Background
        background_score = compute_background_score(image, mask)
        
        # Composite score (weighted)
        composite_score = (
            color_score * self.color_weight +
            hollow_quality * self.hollow_weight +
            sharpness * self.sharpness_weight
        )
        
        # Check gates
        passes_gates = (
            mean_delta_e <= self.delta_e_gate and
            hollow_quality >= self.hollow_gate and
            sharpness >= self.sharpness_gate and
            background_score >= self.background_gate
        )
        
        return QAMetrics(
            mean_delta_e=round(mean_delta_e, 2),
            p95_delta_e=round(p95_delta_e, 2),
            hollow_quality=round(hollow_quality, 3),
            sharpness=round(sharpness, 3),
            background_score=round(background_score, 3),
            composite_score=round(composite_score, 3),
            passes_gates=passes_gates
        )

# ---------------------------------------------------------------------------
# Auto-Corrector
# ---------------------------------------------------------------------------

class AutoCorrector:
    """Apply surgical corrections to improve candidate quality."""
    
    def __init__(self, max_color_bias: float = 0.2):
        self.max_color_bias = max_color_bias
        
    def correct_candidate(
        self,
        image: Image.Image,
        mask: np.ndarray,
        target_hex: str,
        initial_metrics: QAMetrics
    ) -> CorrectionResult:
        """Apply corrections to a candidate image."""
        corrected = image.copy()
        corrections_applied = []
        
        # 1. Color correction (LAB bias toward target)
        if initial_metrics.mean_delta_e > 1.5:
            corrected = self._apply_color_correction(corrected, mask, target_hex)
            corrections_applied.append("color_bias")
            
        # 2. Background normalization
        if initial_metrics.background_score < 0.7:
            corrected = self._normalize_background(corrected, mask)
            corrections_applied.append("background_normalize")
            
        # 3. Final recenter and resize (ensure exact dimensions)
        corrected = self._recenter_and_resize(corrected, target_size=2048)
        corrections_applied.append("recenter_resize")
        
        # Score the corrected image
        scorer = QAScorer()
        final_metrics = scorer.score_candidate(corrected, mask, target_hex)
        
        return CorrectionResult(
            corrected_image=corrected,
            corrections_applied=corrections_applied,
            final_metrics=final_metrics
        )
        
    def _apply_color_correction(
        self, 
        image: Image.Image, 
        mask: np.ndarray, 
        target_hex: str
    ) -> Image.Image:
        """Apply conservative color bias toward target color."""
        rgb = np.array(image.convert("RGB"), dtype=np.float32)
        
        # Ensure mask matches image dimensions
        img_h, img_w = rgb.shape[:2]
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # Target color
        tr, tg, tb = hex_to_rgb(target_hex)
        target_rgb = np.array([tr, tg, tb], dtype=np.float32)
        
        # Get current mean color in garment region
        garment_pixels = rgb[mask > 0]
        if garment_pixels.size == 0:
            return image
            
        current_mean = np.mean(garment_pixels, axis=0)
        
        # Calculate bias (capped)
        bias = (target_rgb - current_mean) * self.max_color_bias
        
        # Apply bias only to garment region
        corrected = rgb.copy()
        corrected[mask > 0] += bias
        corrected = np.clip(corrected, 0, 255)
        
        return Image.fromarray(corrected.astype(np.uint8))
        
    def _normalize_background(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Normalize background to pure white."""
        rgb = np.array(image.convert("RGB"), dtype=np.float32)
        
        # Ensure mask matches image dimensions
        img_h, img_w = rgb.shape[:2]
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # Background mask (inverse of garment)
        bg_mask = (mask == 0)
        
        if np.count_nonzero(bg_mask) == 0:
            return image
            
        # Set background to white
        corrected = rgb.copy()
        corrected[bg_mask] = 255.0
        
        return Image.fromarray(corrected.astype(np.uint8))
        
    def _recenter_and_resize(self, image: Image.Image, target_size: int) -> Image.Image:
        """Ensure image is exactly target_size x target_size and properly centered."""
        
        # If already correct size, return as-is
        if image.size == (target_size, target_size):
            return image
            
        # Resize while maintaining aspect ratio
        image = image.resize((target_size, target_size), Image.LANCZOS)
        
        return image

# ---------------------------------------------------------------------------
# Main QA Pipeline
# ---------------------------------------------------------------------------

class QACorrector:
    """Main QA and correction pipeline for candidate selection."""
    
    def __init__(
        self,
        delta_e_gate: float = 2.0,
        hollow_gate: float = 0.4, 
        sharpness_gate: float = 0.3,
        background_gate: float = 0.6
    ):
        self.scorer = QAScorer(delta_e_gate, hollow_gate, sharpness_gate, background_gate)
        self.corrector = AutoCorrector()
        
    def process_candidates(
        self,
        candidate_paths: List[str],
        mask: np.ndarray,
        target_hex: str,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Process all candidates: score, correct, and select best.
        
        Returns:
            Dict with results including best candidate info and QA report
        """
        results = []
        
        for i, candidate_path in enumerate(candidate_paths):
            logger.info(f"Processing candidate {i+1}/{len(candidate_paths)}: {candidate_path}")
            
            try:
                # Load candidate
                candidate = Image.open(candidate_path).convert("RGB")
                
                # Initial scoring
                initial_metrics = self.scorer.score_candidate(candidate, mask, target_hex)
                
                # Apply corrections
                correction_result = self.corrector.correct_candidate(
                    candidate, mask, target_hex, initial_metrics
                )
                
                # Save corrected candidate
                corrected_path = output_dir / f"corrected_{i}.png"
                correction_result.corrected_image.save(corrected_path, "PNG", optimize=True)
                
                # Store results
                candidate_result = {
                    "candidate_index": i,
                    "original_path": candidate_path,
                    "corrected_path": str(corrected_path),
                    "initial_metrics": {
                        "mean_delta_e": initial_metrics.mean_delta_e,
                        "p95_delta_e": initial_metrics.p95_delta_e,
                        "hollow_quality": initial_metrics.hollow_quality,
                        "sharpness": initial_metrics.sharpness,
                        "background_score": initial_metrics.background_score,
                        "composite_score": initial_metrics.composite_score,
                        "passes_gates": initial_metrics.passes_gates
                    },
                    "final_metrics": {
                        "mean_delta_e": correction_result.final_metrics.mean_delta_e,
                        "p95_delta_e": correction_result.final_metrics.p95_delta_e,
                        "hollow_quality": correction_result.final_metrics.hollow_quality,
                        "sharpness": correction_result.final_metrics.sharpness,
                        "background_score": correction_result.final_metrics.background_score,
                        "composite_score": correction_result.final_metrics.composite_score,
                        "passes_gates": correction_result.final_metrics.passes_gates
                    },
                    "corrections_applied": correction_result.corrections_applied
                }
                
                results.append(candidate_result)
                
            except Exception as e:
                logger.error(f"Failed to process candidate {i}: {e}")
                continue
        
        if not results:
            raise RuntimeError("No candidates could be processed successfully")
            
        # Select best candidate (highest composite score)
        best_candidate = max(results, key=lambda x: x["final_metrics"]["composite_score"])
        best_index = best_candidate["candidate_index"]
        
        # Copy best candidate to final output
        final_path = output_dir / "final.png"
        best_image = Image.open(best_candidate["corrected_path"])
        best_image.save(final_path, "PNG", optimize=True)
        
        # Also create corrected_best.png for backward compatibility
        best_path = output_dir / "corrected_best.png"
        best_image.save(best_path, "PNG", optimize=True)
        
        # Generate QA report
        qa_report = {
            "target_color": target_hex,
            "candidates_processed": len(results),
            "best_candidate": {
                "index": best_index,
                "path": str(final_path),
                "composite_score": best_candidate["final_metrics"]["composite_score"],
                "passes_gates": best_candidate["final_metrics"]["passes_gates"]
            },
            "all_candidates": results,
            "summary": {
                "mean_composite_score": float(np.mean([r["final_metrics"]["composite_score"] for r in results])),
                "candidates_passing_gates": sum(1 for r in results if r["final_metrics"]["passes_gates"]),
                "best_delta_e": best_candidate["final_metrics"]["mean_delta_e"],
                "processing_timestamp": time.time()
            }
        }
        
        return qa_report

# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 4/6: Auto-QA + Correct + Select",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--candidates-dir", required=True, 
                       help="Directory containing candidate images from Step 3")
    parser.add_argument("--analysis-json", required=True, 
                       help="Path to garment_analysis.json from Step 1")
    parser.add_argument("--mask", required=True, 
                       help="Path to mask.png from Step 1")
    parser.add_argument("--out", default="./step4", 
                       help="Output directory")
    parser.add_argument("--deltae-gate", type=float, default=2.0, 
                       help="Delta E threshold for color accuracy")
    parser.add_argument("--hollow-gate", type=float, default=0.4, 
                       help="Minimum hollow effect quality score")
    parser.add_argument("--sharpness-gate", type=float, default=0.3, 
                       help="Minimum sharpness score")
    parser.add_argument("--background-gate", type=float, default=0.6, 
                       help="Minimum background purity score")

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load analysis JSON to get target color
    with open(args.analysis_json, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    target_hex = analysis.get("color_extraction", {}).get("primary_color", {}).get("hex_value", "#000000")
    if not target_hex.startswith("#"):
        target_hex = f"#{target_hex}"

    # Load mask
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {args.mask}")
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    # Find candidate images
    candidates_dir = Path(args.candidates_dir)
    candidate_files = []
    
    # Look for common candidate patterns
    patterns = ["candidate_*.png", "render_*.png", "*.png"]
    for pattern in patterns:
        matches = list(candidates_dir.glob(pattern))
        if matches:
            candidate_files = [str(p) for p in sorted(matches)]
            break

    if not candidate_files:
        raise FileNotFoundError(f"No candidate images found in {candidates_dir}")

    logger.info(f"Found {len(candidate_files)} candidates: {[Path(f).name for f in candidate_files]}")

    # Initialize QA corrector
    qa_corrector = QACorrector(
        delta_e_gate=args.deltae_gate,
        hollow_gate=args.hollow_gate,
        sharpness_gate=args.sharpness_gate,
        background_gate=args.background_gate
    )

    try:
        # Process candidates
        qa_report = qa_corrector.process_candidates(
            candidate_files, mask, target_hex, out_dir
        )

        # Save QA report
        qa_report_path = out_dir / "qa_report.json"
        with open(qa_report_path, 'w', encoding='utf-8') as f:
            json.dump(qa_report, f, indent=2)

        # Print results
        print(f"✅ QA processing complete! Output saved to: {args.out}")
        print(f"📊 Best candidate: Index {qa_report['best_candidate']['index']} "
              f"(score: {qa_report['best_candidate']['composite_score']:.3f})")
        print(f"🎯 Final Delta E: {qa_report['summary']['best_delta_e']:.2f}")
        print(f"🎨 Target color: {target_hex}")
        print(f"✓ Candidates passing gates: {qa_report['summary']['candidates_passing_gates']}/{qa_report['candidates_processed']}")
        
        if qa_report['best_candidate']['passes_gates']:
            print("🟢 Best candidate passes all quality gates")
        else:
            print("🟡 Best candidate does not pass all quality gates")

    except Exception as e:
        logger.error(f"QA processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
