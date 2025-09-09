"""
Quality validation tests against reference dataset.
Tests pipeline output quality against known good examples.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple

# Import quality assessment modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from photostudio.steps.step4_qa_corrector import ColorAccuracyAssessment, HollowQualityAssessment


class TestReferenceDatasetValidation:
    """Quality validation against reference examples."""
    
    @pytest.fixture
    def reference_examples(self, test_data_dir):
        """Create reference examples with known quality metrics."""
        examples = []
        
        # Example 1: High-quality shirt
        example1 = {
            "name": "high_quality_shirt",
            "image": self._create_reference_image((512, 512), "shirt", quality="high"),
            "expected_metrics": {
                "delta_e": 1.2,
                "hollow_quality": 0.8,
                "sharpness": 0.9,
                "background_score": 0.95,
                "composite_score": 0.85
            },
            "should_pass_gates": True
        }
        
        # Example 2: Medium-quality dress
        example2 = {
            "name": "medium_quality_dress", 
            "image": self._create_reference_image((512, 512), "dress", quality="medium"),
            "expected_metrics": {
                "delta_e": 2.8,
                "hollow_quality": 0.5,
                "sharpness": 0.6,
                "background_score": 0.85,
                "composite_score": 0.6
            },
            "should_pass_gates": False
        }
        
        # Example 3: Poor-quality jacket
        example3 = {
            "name": "poor_quality_jacket",
            "image": self._create_reference_image((512, 512), "jacket", quality="poor"),
            "expected_metrics": {
                "delta_e": 4.2,
                "hollow_quality": 0.2,
                "sharpness": 0.3,
                "background_score": 0.7,
                "composite_score": 0.35
            },
            "should_pass_gates": False
        }
        
        examples.extend([example1, example2, example3])
        return examples
    
    def _create_reference_image(self, size: Tuple[int, int], garment_type: str, quality: str) -> Image.Image:
        """Create synthetic reference images with controlled quality characteristics."""
        width, height = size
        img = Image.new('RGB', (width, height), color='white')
        pixels = np.array(img)
        
        # Define quality-based parameters
        quality_params = {
            "high": {"noise_level": 5, "edge_sharpness": 0.9, "color_consistency": 0.95},
            "medium": {"noise_level": 15, "edge_sharpness": 0.6, "color_consistency": 0.8},
            "poor": {"noise_level": 30, "edge_sharpness": 0.3, "color_consistency": 0.6}
        }
        
        params = quality_params[quality]
        
        # Create garment shape based on type
        if garment_type == "shirt":
            # Main body
            body_color = [70, 130, 180] if quality == "high" else [80, 140, 190]
            pixels[100:400, 150:350] = body_color
            
            # Sleeves
            pixels[100:250, 50:150] = body_color  # Left sleeve
            pixels[100:250, 350:450] = body_color  # Right sleeve
            
        elif garment_type == "dress":
            # Dress silhouette
            dress_color = [200, 100, 150] if quality == "high" else [180, 120, 160]
            pixels[80:450, 180:320] = dress_color
            
        elif garment_type == "jacket":
            # Jacket with collar
            jacket_color = [50, 50, 50] if quality == "high" else [70, 70, 70]
            pixels[90:380, 140:360] = jacket_color
            
            # Sleeves
            pixels[90:280, 40:140] = jacket_color  # Left sleeve
            pixels[90:280, 360:460] = jacket_color  # Right sleeve
        
        # Add quality-based noise and artifacts
        if quality != "high":
            noise = np.random.normal(0, params["noise_level"], pixels.shape).astype(np.int16)
            pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Simulate edge quality
        if quality == "poor":
            # Add blur to simulate poor edge quality
            from PIL import ImageFilter
            img = Image.fromarray(pixels)
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
            pixels = np.array(img)
        
        return Image.fromarray(pixels)
    
    def test_reference_color_accuracy(self, reference_examples):
        """Test color accuracy assessment against reference examples."""
        color_assessor = ColorAccuracyAssessment()
        
        for example in reference_examples:
            # Create reference color (dominant garment color)
            reference_rgb = [70, 130, 180]  # Steel blue
            reference_lab = color_assessor._rgb_to_lab(reference_rgb)
            
            # Create mock mask for garment area
            mask = Image.new('L', example["image"].size, color=0)
            mask_pixels = np.array(mask)
            mask_pixels[100:400, 150:350] = 255  # Main garment area
            mask = Image.fromarray(mask_pixels, mode='L')
            
            # Assess color accuracy
            delta_e = color_assessor.assess_color_accuracy(
                example["image"],
                mask,
                reference_lab
            )
            
            expected_delta_e = example["expected_metrics"]["delta_e"]
            
            # Allow some tolerance for synthetic data
            tolerance = 1.0
            assert abs(delta_e - expected_delta_e) <= tolerance, \
                f"Color accuracy mismatch for {example['name']}: expected {expected_delta_e}, got {delta_e}"
    
    def test_reference_hollow_quality(self, reference_examples):
        """Test hollow quality assessment against reference examples."""
        hollow_assessor = HollowQualityAssessment()
        
        for example in reference_examples:
            # Create mock mask
            mask = Image.new('L', example["image"].size, color=0)
            mask_pixels = np.array(mask)
            mask_pixels[100:400, 150:350] = 255  # Main area
            mask = Image.fromarray(mask_pixels, mode='L')
            
            # Assess hollow quality
            hollow_score = hollow_assessor.assess_hollow_quality(example["image"], mask)
            
            expected_hollow = example["expected_metrics"]["hollow_quality"]
            
            # Allow tolerance for synthetic assessment
            tolerance = 0.2
            assert abs(hollow_score - expected_hollow) <= tolerance, \
                f"Hollow quality mismatch for {example['name']}: expected {expected_hollow}, got {hollow_score}"
    
    def test_quality_gate_thresholds(self, reference_examples):
        """Test that quality gates correctly identify pass/fail cases."""
        # Standard quality thresholds
        thresholds = {
            "delta_e": 2.0,
            "hollow_quality": 0.4,
            "sharpness": 0.3,
            "background_score": 0.8,
            "composite_score": 0.6
        }
        
        for example in reference_examples:
            metrics = example["expected_metrics"]
            expected_pass = example["should_pass_gates"]
            
            # Evaluate each gate
            gates_passed = {
                "color": metrics["delta_e"] <= thresholds["delta_e"],
                "hollow": metrics["hollow_quality"] >= thresholds["hollow_quality"],
                "sharpness": metrics["sharpness"] >= thresholds["sharpness"],
                "background": metrics["background_score"] >= thresholds["background_score"],
                "composite": metrics["composite_score"] >= thresholds["composite_score"]
            }
            
            overall_pass = all(gates_passed.values())
            
            assert overall_pass == expected_pass, \
                f"Quality gate mismatch for {example['name']}: expected {'pass' if expected_pass else 'fail'}, " + \
                f"got {'pass' if overall_pass else 'fail'}. Gates: {gates_passed}"
    
    def test_consistency_across_similar_inputs(self):
        """Test that similar inputs produce consistent quality scores."""
        # Create multiple similar images
        similar_images = []
        for i in range(5):
            img = self._create_reference_image((512, 512), "shirt", "high")
            similar_images.append(img)
        
        # Assess quality for each
        color_assessor = ColorAccuracyAssessment()
        hollow_assessor = HollowQualityAssessment()
        
        reference_lab = color_assessor._rgb_to_lab([70, 130, 180])
        
        scores = []
        for img in similar_images:
            # Create consistent mask
            mask = Image.new('L', img.size, color=0)
            mask_pixels = np.array(mask)
            mask_pixels[100:400, 150:350] = 255
            mask = Image.fromarray(mask_pixels, mode='L')
            
            delta_e = color_assessor.assess_color_accuracy(img, mask, reference_lab)
            hollow_score = hollow_assessor.assess_hollow_quality(img, mask)
            
            scores.append({"delta_e": delta_e, "hollow": hollow_score})
        
        # Check consistency (should have low variance)
        delta_e_values = [s["delta_e"] for s in scores]
        hollow_values = [s["hollow"] for s in scores]
        
        delta_e_std = np.std(delta_e_values)
        hollow_std = np.std(hollow_values)
        
        # Should have relatively low standard deviation for similar inputs
        assert delta_e_std < 0.5, f"Delta-E scores too inconsistent: std={delta_e_std}"
        assert hollow_std < 0.1, f"Hollow quality scores too inconsistent: std={hollow_std}"
    
    def test_quality_regression_detection(self, reference_examples):
        """Test ability to detect quality regressions."""
        # Simulate a processing pipeline with known quality degradation
        
        for example in reference_examples:
            original_image = example["image"]
            
            # Create degraded version
            degraded_image = self._degrade_image_quality(original_image)
            
            # Assess both versions
            color_assessor = ColorAccuracyAssessment()
            reference_lab = color_assessor._rgb_to_lab([70, 130, 180])
            
            # Create mask
            mask = Image.new('L', original_image.size, color=0)
            mask_pixels = np.array(mask)
            mask_pixels[100:400, 150:350] = 255
            mask = Image.fromarray(mask_pixels, mode='L')
            
            # Compare quality scores
            original_delta_e = color_assessor.assess_color_accuracy(original_image, mask, reference_lab)
            degraded_delta_e = color_assessor.assess_color_accuracy(degraded_image, mask, reference_lab)
            
            # Degraded version should have worse (higher) delta-E
            assert degraded_delta_e > original_delta_e, \
                f"Quality regression not detected for {example['name']}: " + \
                f"original ΔE={original_delta_e:.2f}, degraded ΔE={degraded_delta_e:.2f}"
            
            # Should detect significant degradation
            degradation_amount = degraded_delta_e - original_delta_e
            assert degradation_amount > 0.5, \
                f"Quality degradation too small to be meaningful: {degradation_amount:.2f}"
    
    def _degrade_image_quality(self, image: Image.Image) -> Image.Image:
        """Create a degraded version of an image for regression testing."""
        pixels = np.array(image)
        
        # Add noise
        noise = np.random.normal(0, 20, pixels.shape).astype(np.int16)
        degraded_pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add blur
        degraded_img = Image.fromarray(degraded_pixels)
        from PIL import ImageFilter
        degraded_img = degraded_img.filter(ImageFilter.GaussianBlur(radius=1.5))
        
        return degraded_img
    
    def test_edge_case_handling(self):
        """Test quality assessment with edge cases."""
        # Test with very small garment area
        small_area_img = Image.new('RGB', (512, 512), color='white')
        pixels = np.array(small_area_img)
        pixels[250:260, 250:260] = [100, 100, 100]  # Tiny 10x10 garment
        small_area_img = Image.fromarray(pixels)
        
        small_mask = Image.new('L', (512, 512), color=0)
        mask_pixels = np.array(small_mask)
        mask_pixels[250:260, 250:260] = 255
        small_mask = Image.fromarray(mask_pixels, mode='L')
        
        # Should handle gracefully without crashing
        color_assessor = ColorAccuracyAssessment()
        hollow_assessor = HollowQualityAssessment()
        
        try:
            reference_lab = [50, 0, 0]  # Neutral gray
            delta_e = color_assessor.assess_color_accuracy(small_area_img, small_mask, reference_lab)
            hollow_score = hollow_assessor.assess_hollow_quality(small_area_img, small_mask)
            
            # Should return valid values
            assert isinstance(delta_e, (int, float))
            assert isinstance(hollow_score, (int, float))
            assert delta_e >= 0
            assert 0 <= hollow_score <= 1
            
        except Exception as e:
            pytest.fail(f"Edge case handling failed: {e}")
    
    def test_performance_benchmarks(self, reference_examples):
        """Test quality assessment performance benchmarks."""
        import time
        
        # Benchmark color accuracy assessment
        color_assessor = ColorAccuracyAssessment()
        reference_lab = [50, 0, 0]
        
        start_time = time.time()
        
        for example in reference_examples:
            mask = Image.new('L', example["image"].size, color=0)
            mask_pixels = np.array(mask)
            mask_pixels[100:400, 150:350] = 255
            mask = Image.fromarray(mask_pixels, mode='L')
            
            color_assessor.assess_color_accuracy(example["image"], mask, reference_lab)
        
        color_assessment_time = time.time() - start_time
        
        # Should complete quickly (under 1 second for small dataset)
        assert color_assessment_time < 1.0, \
            f"Color assessment too slow: {color_assessment_time:.3f}s for {len(reference_examples)} images"
        
        # Benchmark hollow quality assessment
        hollow_assessor = HollowQualityAssessment()
        
        start_time = time.time()
        
        for example in reference_examples:
            mask = Image.new('L', example["image"].size, color=0)
            mask_pixels = np.array(mask)
            mask_pixels[100:400, 150:350] = 255
            mask = Image.fromarray(mask_pixels, mode='L')
            
            hollow_assessor.assess_hollow_quality(example["image"], mask)
        
        hollow_assessment_time = time.time() - start_time
        
        assert hollow_assessment_time < 2.0, \
            f"Hollow assessment too slow: {hollow_assessment_time:.3f}s for {len(reference_examples)} images"
        
        print(f"Performance benchmarks:")
        print(f"  Color assessment: {color_assessment_time:.3f}s ({len(reference_examples)} images)")
        print(f"  Hollow assessment: {hollow_assessment_time:.3f}s ({len(reference_examples)} images)")


class TestQualityMetricsValidation:
    """Validate individual quality metrics against known standards."""
    
    def test_delta_e_calculation_accuracy(self):
        """Test Delta-E calculation against known color science standards."""
        color_assessor = ColorAccuracyAssessment()
        
        # Test known color pairs with expected Delta-E values
        test_pairs = [
            # (color1_lab, color2_lab, expected_delta_e)
            ([50, 0, 0], [50, 0, 0], 0.0),  # Identical colors
            ([50, 0, 0], [60, 0, 0], 10.0),  # L* difference of 10
            ([50, 0, 0], [50, 10, 0], 10.0),  # a* difference of 10
            ([50, 0, 0], [50, 0, 10], 10.0),  # b* difference of 10
        ]
        
        for color1, color2, expected_delta_e in test_pairs:
            calculated_delta_e = color_assessor._calculate_delta_e_2000(color1, color2)
            
            # Allow some tolerance for floating point precision
            tolerance = 0.1
            assert abs(calculated_delta_e - expected_delta_e) <= tolerance, \
                f"Delta-E calculation error: expected {expected_delta_e}, got {calculated_delta_e}"
    
    def test_sharpness_metric_validity(self):
        """Test sharpness metric with synthetic sharp and blurry images."""
        from photostudio.steps.step4_qa_corrector import SharpnessAssessment
        
        sharpness_assessor = SharpnessAssessment()
        
        # Create sharp image (checkerboard pattern)
        sharp_img = Image.new('RGB', (256, 256), color='white')
        pixels = np.array(sharp_img)
        
        # Create checkerboard
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                if (i // 32 + j // 32) % 2 == 0:
                    pixels[i:i+32, j:j+32] = [0, 0, 0]
        
        sharp_img = Image.fromarray(pixels)
        
        # Create blurry version
        from PIL import ImageFilter
        blurry_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=5))
        
        # Assess sharpness
        sharp_score = sharpness_assessor.assess_sharpness(sharp_img)
        blurry_score = sharpness_assessor.assess_sharpness(blurry_img)
        
        # Sharp image should have higher sharpness score
        assert sharp_score > blurry_score, \
            f"Sharpness metric failed: sharp={sharp_score:.3f}, blurry={blurry_score:.3f}"
        
        # Scores should be in reasonable ranges
        assert sharp_score > 0.5, f"Sharp image score too low: {sharp_score:.3f}"
        assert blurry_score < 0.5, f"Blurry image score too high: {blurry_score:.3f}"
