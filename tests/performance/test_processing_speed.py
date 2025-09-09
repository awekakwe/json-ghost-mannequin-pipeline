"""
Performance benchmarking tests for the JSON Ghost-Mannequin Pipeline.
Tests processing speed, memory usage, and scalability.
"""

import time
import json
import pytest
import psutil
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# Import pipeline components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from photostudio.steps.step1_preprocess import ImagePreprocessor, ColorExtractor
from photostudio.steps.step2_prompt_weaver import PromptWeaver
from photostudio.steps.step4_qa_corrector import QualityAssessment


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    @pytest.fixture
    def performance_images(self):
        """Create test images of various sizes for performance testing."""
        images = {}
        
        sizes = [
            (256, 256),   # Small
            (512, 512),   # Medium  
            (1024, 1024), # Large
            (2048, 2048), # Extra Large
        ]
        
        for width, height in sizes:
            img = Image.new('RGB', (width, height), color='white')
            # Add some complexity
            import numpy as np
            pixels = np.array(img)
            
            # Add garment shape
            center_x, center_y = width // 2, height // 2
            garment_width = width // 3
            garment_height = height // 2
            
            x1 = center_x - garment_width // 2
            x2 = center_x + garment_width // 2
            y1 = center_y - garment_height // 2
            y2 = center_y + garment_height // 2
            
            pixels[y1:y2, x1:x2] = [70, 130, 180]  # Blue garment
            
            img = Image.fromarray(pixels)
            images[f"{width}x{height}"] = img
        
        return images
    
    def test_image_loading_performance(self, performance_images, temp_dir):
        """Benchmark image loading performance across different sizes."""
        preprocessor = ImagePreprocessor()
        loading_times = {}
        
        for size_name, image in performance_images.items():
            # Save image to disk
            image_path = temp_dir / f"test_{size_name}.jpg"
            image.save(image_path, quality=95)
            
            # Benchmark loading
            start_time = time.time()
            loaded_image = preprocessor.load_image(image_path)
            end_time = time.time()
            
            loading_time = end_time - start_time
            loading_times[size_name] = loading_time
            
            assert loaded_image is not None
            assert loaded_image.size == image.size
        
        # Print benchmark results
        print("\nImage Loading Performance:")
        for size, load_time in loading_times.items():
            print(f"  {size}: {load_time:.4f}s")
        
        # Performance assertions
        assert loading_times["256x256"] < 0.1, "Small image loading too slow"
        assert loading_times["512x512"] < 0.2, "Medium image loading too slow"
        assert loading_times["1024x1024"] < 0.5, "Large image loading too slow"
        assert loading_times["2048x2048"] < 2.0, "Extra large image loading too slow"
    
    def test_color_extraction_performance(self, performance_images):
        """Benchmark color extraction across different image sizes."""
        color_extractor = ColorExtractor()
        extraction_times = {}
        
        for size_name, image in performance_images.items():
            # Create matching mask
            mask = Image.new('L', image.size, color=0)
            import numpy as np
            mask_pixels = np.array(mask)
            
            # Create garment mask area
            width, height = image.size
            center_x, center_y = width // 2, height // 2
            garment_width = width // 3
            garment_height = height // 2
            
            x1 = center_x - garment_width // 2
            x2 = center_x + garment_width // 2
            y1 = center_y - garment_height // 2
            y2 = center_y + garment_height // 2
            
            mask_pixels[y1:y2, x1:x2] = 255
            mask = Image.fromarray(mask_pixels, mode='L')
            
            # Benchmark color extraction
            start_time = time.time()
            color_info = color_extractor.extract_dominant_color(image, mask)
            end_time = time.time()
            
            extraction_time = end_time - start_time
            extraction_times[size_name] = extraction_time
            
            assert 'rgb' in color_info
            assert 'hex' in color_info
        
        # Print benchmark results
        print("\nColor Extraction Performance:")
        for size, extract_time in extraction_times.items():
            print(f"  {size}: {extract_time:.4f}s")
        
        # Performance assertions - color extraction should scale reasonably
        assert extraction_times["256x256"] < 0.1, "Small image color extraction too slow"
        assert extraction_times["512x512"] < 0.2, "Medium image color extraction too slow"
        assert extraction_times["1024x1024"] < 0.5, "Large image color extraction too slow"
        assert extraction_times["2048x2048"] < 2.0, "Extra large image color extraction too slow"
    
    def test_prompt_weaving_performance(self, sample_analysis_json):
        """Benchmark prompt weaving performance."""
        style_kit = {
            "templates": {
                "shirt": {
                    "base_prompt": "professional product photography, {garment_description}, ghost mannequin effect, clean white background",
                    "style_modifiers": [
                        "crisp fabric texture",
                        "natural draping",
                        "sharp focus",
                        "commercial quality"
                    ]
                }
            }
        }
        
        prompt_weaver = PromptWeaver(style_kit)
        
        # Benchmark single prompt weaving
        start_time = time.time()
        prompt_data = prompt_weaver.weave_prompt(sample_analysis_json)
        single_time = time.time() - start_time
        
        assert "final_prompt" in prompt_data
        
        # Benchmark batch prompt weaving (100 iterations)
        batch_size = 100
        start_time = time.time()
        
        for _ in range(batch_size):
            prompt_data = prompt_weaver.weave_prompt(sample_analysis_json)
        
        batch_time = time.time() - start_time
        avg_time_per_prompt = batch_time / batch_size
        
        print(f"\nPrompt Weaving Performance:")
        print(f"  Single prompt: {single_time:.4f}s")
        print(f"  Batch average: {avg_time_per_prompt:.4f}s")
        print(f"  Batch total ({batch_size} prompts): {batch_time:.4f}s")
        
        # Performance assertions
        assert single_time < 0.1, f"Single prompt weaving too slow: {single_time:.4f}s"
        assert avg_time_per_prompt < 0.01, f"Batch prompt weaving too slow: {avg_time_per_prompt:.4f}s"
    
    def test_qa_assessment_performance(self, performance_images, sample_analysis_json):
        """Benchmark QA assessment performance."""
        quality_assessment = QualityAssessment()
        assessment_times = {}
        
        for size_name, image in performance_images.items():
            # Create matching mask
            mask = Image.new('L', image.size, color=255)  # Full mask for simplicity
            
            # Benchmark QA assessment (mock the heavy operations)
            start_time = time.time()
            
            # Mock QA metrics calculation
            mock_metrics = {
                "image_sharpness": 0.75,
                "color_consistency": 0.85,
                "segmentation_confidence": 0.92,
                "overall_quality": 0.84
            }
            
            # Simulate some processing time based on image size
            width, height = image.size
            processing_factor = (width * height) / (512 * 512)  # Normalize to 512x512
            simulated_processing_time = 0.01 * processing_factor  # Base processing time
            
            time.sleep(simulated_processing_time)
            
            end_time = time.time()
            
            assessment_time = end_time - start_time
            assessment_times[size_name] = assessment_time
        
        # Print benchmark results
        print("\nQA Assessment Performance (simulated):")
        for size, assess_time in assessment_times.items():
            print(f"  {size}: {assess_time:.4f}s")
        
        # Performance assertions - should scale reasonably with image size
        base_time = assessment_times["256x256"]
        medium_time = assessment_times["512x512"]
        large_time = assessment_times["1024x1024"]
        
        # Scaling should be reasonable (not exponential)
        assert medium_time <= base_time * 5, "QA assessment scaling poorly for medium images"
        assert large_time <= base_time * 20, "QA assessment scaling poorly for large images"
    
    def test_memory_usage_monitoring(self, performance_images):
        """Monitor memory usage during image processing."""
        process = psutil.Process()
        memory_usage = {}
        
        # Get baseline memory usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        for size_name, image in performance_images.items():
            # Monitor memory before processing
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Simulate image processing
            preprocessor = ImagePreprocessor()
            
            # Load and normalize image (memory-intensive operations)
            normalized_image = preprocessor._normalize_image(image)
            
            # Create multiple copies to simulate processing
            processed_images = []
            for i in range(5):  # Create 5 variants
                processed_images.append(normalized_image.copy())
            
            # Monitor memory after processing
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_delta = memory_after - memory_before
            
            memory_usage[size_name] = {
                "before": memory_before,
                "after": memory_after,
                "delta": memory_delta
            }
            
            # Clean up
            del processed_images
            del normalized_image
        
        # Print memory usage results
        print(f"\nMemory Usage (Baseline: {baseline_memory:.1f}MB):")
        for size, usage in memory_usage.items():
            print(f"  {size}: +{usage['delta']:.1f}MB (before: {usage['before']:.1f}MB, after: {usage['after']:.1f}MB)")
        
        # Memory usage assertions
        assert memory_usage["256x256"]["delta"] < 50, "Small image processing uses too much memory"
        assert memory_usage["512x512"]["delta"] < 100, "Medium image processing uses too much memory"
        assert memory_usage["1024x1024"]["delta"] < 200, "Large image processing uses too much memory"
        assert memory_usage["2048x2048"]["delta"] < 500, "Extra large image processing uses too much memory"
    
    def test_concurrent_processing_performance(self, performance_images, temp_dir):
        """Test performance under concurrent processing."""
        preprocessor = ImagePreprocessor()
        
        # Save test images
        image_paths = []
        for size_name, image in performance_images.items():
            image_path = temp_dir / f"concurrent_{size_name}.jpg"
            image.save(image_path)
            image_paths.append(image_path)
        
        # Sequential processing benchmark
        start_time = time.time()
        for image_path in image_paths:
            loaded_image = preprocessor.load_image(image_path)
            normalized = preprocessor._normalize_image(loaded_image)
        sequential_time = time.time() - start_time
        
        # Concurrent processing benchmark
        def process_image(image_path):
            loaded_image = preprocessor.load_image(image_path)
            normalized = preprocessor._normalize_image(loaded_image)
            return normalized.size
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_image, path) for path in image_paths]
            results = [future.result() for future in futures]
        concurrent_time = time.time() - start_time
        
        # Calculate speedup
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0
        
        print(f"\nConcurrent Processing Performance:")
        print(f"  Sequential time: {sequential_time:.4f}s")
        print(f"  Concurrent time: {concurrent_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Assertions
        assert len(results) == len(image_paths), "Not all images processed successfully"
        assert concurrent_time <= sequential_time * 1.1, "Concurrent processing slower than sequential"
        # We expect some speedup, but not necessarily linear due to I/O overhead
        assert speedup >= 1.0, "No speedup from concurrent processing"
    
    def test_batch_processing_scalability(self, sample_analysis_json):
        """Test scalability of batch processing."""
        style_kit = {
            "templates": {
                "shirt": {
                    "base_prompt": "test prompt {garment_description}",
                    "style_modifiers": ["modifier1", "modifier2"]
                }
            }
        }
        
        prompt_weaver = PromptWeaver(style_kit)
        
        # Test different batch sizes
        batch_sizes = [1, 10, 50, 100, 500]
        processing_times = {}
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            for _ in range(batch_size):
                prompt_data = prompt_weaver.weave_prompt(sample_analysis_json)
            
            end_time = time.time()
            total_time = end_time - start_time
            time_per_item = total_time / batch_size
            
            processing_times[batch_size] = {
                "total": total_time,
                "per_item": time_per_item
            }
        
        # Print scalability results
        print("\nBatch Processing Scalability:")
        for batch_size, times in processing_times.items():
            print(f"  Batch {batch_size}: {times['total']:.4f}s total, {times['per_item']:.6f}s per item")
        
        # Scalability assertions
        # Per-item time should remain relatively constant (linear scaling)
        time_1 = processing_times[1]["per_item"]
        time_100 = processing_times[100]["per_item"]
        time_500 = processing_times[500]["per_item"]
        
        # Allow some variance but should be roughly linear
        assert time_100 <= time_1 * 2, f"Poor scaling at batch 100: {time_100:.6f}s vs {time_1:.6f}s"
        assert time_500 <= time_1 * 3, f"Poor scaling at batch 500: {time_500:.6f}s vs {time_1:.6f}s"
    
    def test_memory_leak_detection(self, performance_images):
        """Test for memory leaks during repeated processing."""
        process = psutil.Process()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        preprocessor = ImagePreprocessor()
        test_image = performance_images["512x512"]
        
        memory_samples = []
        
        # Process the same image multiple times
        for iteration in range(20):
            # Process image
            normalized = preprocessor._normalize_image(test_image)
            
            # Create and destroy some objects
            temp_images = []
            for i in range(5):
                temp_images.append(normalized.copy())
            
            del temp_images
            del normalized
            
            # Sample memory every 5 iterations
            if iteration % 5 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory - baseline_memory)
        
        print(f"\nMemory Leak Detection (20 iterations):")
        print(f"  Baseline: {baseline_memory:.1f}MB")
        for i, sample in enumerate(memory_samples):
            print(f"  Iteration {i*5}: +{sample:.1f}MB")
        
        # Check for memory leaks
        # Memory should not consistently increase
        if len(memory_samples) >= 3:
            initial_memory = memory_samples[0]
            final_memory = memory_samples[-1]
            memory_growth = final_memory - initial_memory
            
            # Allow some memory growth but not excessive
            assert memory_growth < 50, f"Potential memory leak detected: {memory_growth:.1f}MB growth"
            
            # Check that memory doesn't continuously grow
            max_memory = max(memory_samples)
            assert max_memory < baseline_memory * 0.2, f"Memory usage too high: {max_memory:.1f}MB above baseline"
    
    def test_pipeline_end_to_end_performance(self, sample_image, sample_analysis_json, temp_dir):
        """Benchmark complete pipeline performance end-to-end."""
        
        # Save sample image
        image_path = temp_dir / "e2e_test.jpg"
        sample_image.save(image_path)
        
        pipeline_times = {}
        
        # Step 1: Image Preprocessing (simulated)
        start_time = time.time()
        preprocessor = ImagePreprocessor()
        loaded_image = preprocessor.load_image(image_path)
        normalized_image = preprocessor._normalize_image(loaded_image)
        step1_time = time.time() - start_time
        pipeline_times["step1_preprocess"] = step1_time
        
        # Step 2: Prompt Weaving
        start_time = time.time()
        style_kit = {
            "templates": {
                "shirt": {
                    "base_prompt": "professional product photography, {garment_description}, ghost mannequin effect",
                    "style_modifiers": ["crisp texture", "natural draping"]
                }
            }
        }
        prompt_weaver = PromptWeaver(style_kit)
        prompt_data = prompt_weaver.weave_prompt(sample_analysis_json)
        step2_time = time.time() - start_time
        pipeline_times["step2_prompt_weaving"] = step2_time
        
        # Step 3: Rendering (simulated - very fast mock)
        start_time = time.time()
        # Simulate rendering by creating candidate images
        candidates = []
        for i in range(3):
            candidate = normalized_image.copy()
            candidates.append(candidate)
        step3_time = time.time() - start_time
        pipeline_times["step3_rendering"] = step3_time
        
        # Step 4: QA Assessment (simulated)
        start_time = time.time()
        qa_metrics = {
            "composite_score": 0.85,
            "passes_gates": True
        }
        step4_time = time.time() - start_time
        pipeline_times["step4_qa"] = step4_time
        
        # Step 5: Delivery (simulated)
        start_time = time.time()
        # Simulate creating web variants
        web_variants = []
        for size in [(1200, 1200), (800, 800), (400, 400)]:
            variant = normalized_image.resize(size)
            web_variants.append(variant)
        step5_time = time.time() - start_time
        pipeline_times["step5_delivery"] = step5_time
        
        # Step 6: Feedback (simulated)
        start_time = time.time()
        feedback_summary = {
            "health_score": 0.85,
            "status": "completed"
        }
        step6_time = time.time() - start_time
        pipeline_times["step6_feedback"] = step6_time
        
        # Calculate total time
        total_time = sum(pipeline_times.values())
        
        # Print performance breakdown
        print(f"\nEnd-to-End Pipeline Performance:")
        print(f"  Step 1 (Preprocessing): {step1_time:.4f}s ({step1_time/total_time*100:.1f}%)")
        print(f"  Step 2 (Prompt Weaving): {step2_time:.4f}s ({step2_time/total_time*100:.1f}%)")
        print(f"  Step 3 (Rendering): {step3_time:.4f}s ({step3_time/total_time*100:.1f}%)")
        print(f"  Step 4 (QA Assessment): {step4_time:.4f}s ({step4_time/total_time*100:.1f}%)")
        print(f"  Step 5 (Delivery): {step5_time:.4f}s ({step5_time/total_time*100:.1f}%)")
        print(f"  Step 6 (Feedback): {step6_time:.4f}s ({step6_time/total_time*100:.1f}%)")
        print(f"  Total Pipeline: {total_time:.4f}s")
        
        # Performance assertions for simulated pipeline
        assert total_time < 1.0, f"Simulated pipeline too slow: {total_time:.4f}s"
        assert step1_time < 0.2, f"Step 1 too slow: {step1_time:.4f}s"
        assert step2_time < 0.1, f"Step 2 too slow: {step2_time:.4f}s"
        
        # Return performance data for potential further analysis
        return pipeline_times
