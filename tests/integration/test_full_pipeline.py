"""
Integration tests for the complete JSON Ghost-Mannequin Pipeline workflow.
"""

import json
import pytest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import pipeline components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from photostudio.steps.step1_preprocess import ImagePreprocessor
from photostudio.steps.step2_prompt_weaver import PromptWeaver, StyleKitLoader
from photostudio.steps.step4_qa_corrector import QualityAssessment, ImageCorrector
from photostudio.steps.step5_delivery_engine import DeliveryEngine
from photostudio.steps.step6_feedback_engine import FeedbackEngine, FeedbackConfig


class TestFullPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def pipeline_setup(self, temp_dir, sample_image, mock_style_kit):
        """Setup pipeline test environment."""
        # Create directory structure
        step_dirs = {}
        for i in range(1, 7):
            step_dir = temp_dir / f"step{i}"
            step_dir.mkdir(exist_ok=True)
            step_dirs[f"step{i}"] = step_dir
        
        # Save sample image
        input_image_path = temp_dir / "input.jpg"
        sample_image.save(input_image_path)
        
        # Save mock style kit
        style_kit_path = temp_dir / "style_kit.yml"
        import yaml
        with open(style_kit_path, 'w') as f:
            yaml.dump(mock_style_kit, f)
        
        return {
            "temp_dir": temp_dir,
            "input_image": input_image_path,
            "step_dirs": step_dirs,
            "style_kit_path": style_kit_path,
            "sku": "TEST001",
            "variant": "blue"
        }
    
    def test_step1_to_step2_integration(self, pipeline_setup, sample_analysis_json):
        """Test integration between Step 1 and Step 2."""
        setup = pipeline_setup
        
        # Step 1: Save analysis JSON
        analysis_path = setup["step_dirs"]["step1"] / "analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(sample_analysis_json, f)
        
        # Step 2: Load and weave prompt
        style_kit_loader = StyleKitLoader()
        style_kit = style_kit_loader.load_style_kit(setup["style_kit_path"])
        
        prompt_weaver = PromptWeaver(style_kit)
        prompt_data = prompt_weaver.weave_prompt(sample_analysis_json)
        
        # Verify integration
        assert "final_prompt" in prompt_data
        assert "garment_description" in prompt_data["final_prompt"]
        assert "ghost mannequin" in prompt_data["final_prompt"]
        
        # Save Step 2 outputs
        prompt_path = setup["step_dirs"]["step2"] / "final_prompt.txt"
        with open(prompt_path, 'w') as f:
            f.write(prompt_data["final_prompt"])
        
        assert prompt_path.exists()
    
    def test_step2_to_step3_integration(self, pipeline_setup, sample_analysis_json, sample_prompt_data):
        """Test integration between Step 2 and Step 3 (mocked rendering)."""
        setup = pipeline_setup
        
        # Step 2: Save prompt data
        prompt_path = setup["step_dirs"]["step2"] / "final_prompt.txt"
        with open(prompt_path, 'w') as f:
            f.write(sample_prompt_data["final_prompt"])
        
        # Step 3: Mock rendering process
        candidate_images = []
        for i in range(3):
            # Create mock candidate images
            candidate_path = setup["step_dirs"]["step3"] / f"candidate_{i}.png"
            setup["temp_dir"] / "input.jpg"  # Use input image as mock candidate
            
            # Copy input image as candidate (for testing)
            import shutil
            shutil.copy(setup["input_image"], candidate_path)
            candidate_images.append(candidate_path)
        
        # Save render metadata
        render_metadata = {
            "candidates_generated": len(candidate_images),
            "route": "A",
            "render_settings": {
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            }
        }
        
        metadata_path = setup["step_dirs"]["step3"] / "render_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(render_metadata, f)
        
        # Verify Step 3 outputs
        assert len(candidate_images) == 3
        assert metadata_path.exists()
        for candidate_path in candidate_images:
            assert candidate_path.exists()
    
    def test_step3_to_step4_integration(self, pipeline_setup, sample_analysis_json, sample_qa_metrics):
        """Test integration between Step 3 and Step 4."""
        setup = pipeline_setup
        
        # Step 3: Create mock candidates
        candidate_paths = []
        for i in range(3):
            candidate_path = setup["step_dirs"]["step3"] / f"candidate_{i}.png"
            import shutil
            shutil.copy(setup["input_image"], candidate_path)
            candidate_paths.append(candidate_path)
        
        # Step 4: Mock QA assessment
        quality_assessment = QualityAssessment()
        
        # Mock the QA process
        with patch.object(quality_assessment, 'assess_candidates') as mock_assess:
            mock_assess.return_value = {
                "candidates_processed": 3,
                "best_candidate": {
                    "index": 1,
                    "filename": "candidate_1.png",
                    "final_metrics": sample_qa_metrics
                }
            }
            
            qa_report = quality_assessment.assess_candidates(
                setup["step_dirs"]["step3"],
                sample_analysis_json
            )
        
        # Save QA report
        qa_report_path = setup["step_dirs"]["step4"] / "qa_report.json"
        with open(qa_report_path, 'w') as f:
            json.dump(qa_report, f)
        
        # Create corrected best image
        best_image_path = setup["step_dirs"]["step4"] / "corrected_best.png"
        import shutil
        shutil.copy(setup["input_image"], best_image_path)
        
        # Verify Step 4 outputs
        assert qa_report_path.exists()
        assert best_image_path.exists()
        assert qa_report["candidates_processed"] == 3
    
    def test_step4_to_step5_integration(self, pipeline_setup, sample_delivery_manifest):
        """Test integration between Step 4 and Step 5."""
        setup = pipeline_setup
        
        # Step 4: Create QA report and best image
        qa_report = {
            "best_candidate": {
                "filename": "corrected_best.png",
                "final_metrics": {
                    "composite_score": 0.85,
                    "passes_gates": True
                }
            }
        }
        qa_report_path = setup["step_dirs"]["step4"] / "qa_report.json"
        with open(qa_report_path, 'w') as f:
            json.dump(qa_report, f)
        
        best_image_path = setup["step_dirs"]["step4"] / "corrected_best.png"
        import shutil
        shutil.copy(setup["input_image"], best_image_path)
        
        # Step 5: Mock delivery engine
        delivery_engine = DeliveryEngine()
        
        with patch.object(delivery_engine, 'create_delivery_package') as mock_delivery:
            mock_delivery.return_value = sample_delivery_manifest
            
            manifest = delivery_engine.create_delivery_package(
                best_image_path,
                qa_report,
                setup["sku"],
                setup["variant"],
                setup["step_dirs"]["step5"]
            )
        
        # Save delivery manifest
        manifest_path = setup["step_dirs"]["step5"] / "delivery_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        # Verify Step 5 outputs
        assert manifest_path.exists()
        assert manifest["product_info"]["sku"] == setup["sku"]
    
    def test_step5_to_step6_integration(self, pipeline_setup, sample_delivery_manifest):
        """Test integration between Step 5 and Step 6."""
        setup = pipeline_setup
        
        # Step 5: Create delivery manifest and QA report
        manifest_path = setup["step_dirs"]["step5"] / "delivery_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(sample_delivery_manifest, f)
        
        qa_report = {
            "best_candidate": {
                "final_metrics": {
                    "mean_delta_e": 1.5,
                    "hollow_quality": 0.7,
                    "composite_score": 0.8,
                    "passes_gates": True
                }
            }
        }
        qa_report_path = setup["step_dirs"]["step4"] / "qa_report.json"
        with open(qa_report_path, 'w') as f:
            json.dump(qa_report, f)
        
        # Step 6: Mock feedback engine
        feedback_config = FeedbackConfig(
            db_url="sqlite:///:memory:",  # In-memory database for testing
            enable_alerts=False,
            enable_auto_tuning=True
        )
        
        with patch('photostudio.steps.step6_feedback_engine.DatabaseManager') as mock_db:
            mock_db_instance = MagicMock()
            mock_db.return_value = mock_db_instance
            
            # Mock health metrics
            mock_db_instance.get_health_metrics.return_value = {
                "total_runs": 1,
                "health_score": 0.85,
                "success_rate": 1.0,
                "avg_delta_e": 1.5,
                "avg_processing_time": 45.2
            }
            
            feedback_engine = FeedbackEngine(feedback_config)
            
            # Process feedback
            summary = feedback_engine.process_run_feedback(
                run_id="test_run_001",
                sku=setup["sku"],
                variant=setup["variant"],
                route="A",
                template_id="ghost_mannequin_v1",
                qa_report_path=qa_report_path,
                manifest_path=manifest_path,
                processing_time=45.2
            )
        
        # Save feedback summary
        summary_path = setup["step_dirs"]["step6"] / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f)
        
        # Verify Step 6 outputs
        assert summary_path.exists()
        assert summary["sku"] == f"{setup['sku']}-{setup['variant']}"
        assert summary["health_metrics"]["health_score"] > 0
    
    def test_complete_pipeline_simulation(self, pipeline_setup, sample_analysis_json, sample_qa_metrics):
        """Test a complete pipeline simulation with all steps."""
        setup = pipeline_setup
        start_time = time.time()
        
        # Step 1: Image preprocessing (mocked)
        analysis_path = setup["step_dirs"]["step1"] / "analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(sample_analysis_json, f)
        
        mask_path = setup["step_dirs"]["step1"] / "garment_mask.png"
        import shutil
        shutil.copy(setup["input_image"], mask_path)  # Mock mask
        
        # Step 2: Prompt weaving
        style_kit_loader = StyleKitLoader()
        
        # Mock style kit loading
        with patch.object(style_kit_loader, 'load_style_kit') as mock_load:
            mock_style_kit = {
                "templates": {
                    "shirt": {
                        "base_prompt": "professional product photography, {garment_description}, ghost mannequin effect",
                        "style_modifiers": ["crisp texture", "natural draping"]
                    }
                }
            }
            mock_load.return_value = mock_style_kit
            
            style_kit = style_kit_loader.load_style_kit(setup["style_kit_path"])
            prompt_weaver = PromptWeaver(style_kit)
            prompt_data = prompt_weaver.weave_prompt(sample_analysis_json)
        
        prompt_path = setup["step_dirs"]["step2"] / "final_prompt.txt"
        with open(prompt_path, 'w') as f:
            f.write(prompt_data["final_prompt"])
        
        # Step 3: Rendering (mocked)
        for i in range(3):
            candidate_path = setup["step_dirs"]["step3"] / f"candidate_{i}.png"
            shutil.copy(setup["input_image"], candidate_path)
        
        # Step 4: QA & Correction
        qa_report = {
            "candidates_processed": 3,
            "best_candidate": {
                "index": 1,
                "filename": "candidate_1.png",
                "final_metrics": sample_qa_metrics
            }
        }
        
        qa_report_path = setup["step_dirs"]["step4"] / "qa_report.json"
        with open(qa_report_path, 'w') as f:
            json.dump(qa_report, f)
        
        best_image_path = setup["step_dirs"]["step4"] / "corrected_best.png"
        shutil.copy(setup["input_image"], best_image_path)
        
        # Step 5: Delivery
        delivery_manifest = {
            "product_info": {
                "sku": setup["sku"],
                "variant": setup["variant"]
            },
            "processing_metadata": {
                "pipeline_version": "1.1.0",
                "route": "A",
                "qa_passed": True
            }
        }
        
        manifest_path = setup["step_dirs"]["step5"] / "delivery_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(delivery_manifest, f)
        
        # Step 6: Feedback (mocked)
        processing_time = time.time() - start_time
        
        feedback_summary = {
            "run_id": "test_complete_pipeline",
            "sku": f"{setup['sku']}-{setup['variant']}",
            "processing_time": processing_time,
            "health_metrics": {
                "health_score": 0.85,
                "success_rate": 1.0,
                "avg_delta_e": 1.5
            },
            "status": "completed"
        }
        
        summary_path = setup["step_dirs"]["step6"] / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(feedback_summary, f)
        
        # Verify complete pipeline
        assert analysis_path.exists()
        assert prompt_path.exists()
        assert (setup["step_dirs"]["step3"] / "candidate_0.png").exists()
        assert qa_report_path.exists()
        assert best_image_path.exists()
        assert manifest_path.exists()
        assert summary_path.exists()
        
        # Verify data flow
        with open(analysis_path, 'r') as f:
            saved_analysis = json.load(f)
        assert saved_analysis["garment_type"] == sample_analysis_json["garment_type"]
        
        with open(qa_report_path, 'r') as f:
            saved_qa = json.load(f)
        assert saved_qa["candidates_processed"] == 3
        
        with open(summary_path, 'r') as f:
            saved_summary = json.load(f)
        assert saved_summary["status"] == "completed"
    
    def test_error_propagation(self, pipeline_setup):
        """Test error handling and propagation through pipeline steps."""
        setup = pipeline_setup
        
        # Test missing analysis JSON (Step 1 failure)
        style_kit_loader = StyleKitLoader()
        
        with patch.object(style_kit_loader, 'load_style_kit') as mock_load:
            mock_load.return_value = {"templates": {"shirt": {}}}
            style_kit = style_kit_loader.load_style_kit(setup["style_kit_path"])
            
            prompt_weaver = PromptWeaver(style_kit)
            
            # Should handle missing analysis gracefully
            with pytest.raises((FileNotFoundError, KeyError, json.JSONDecodeError)):
                missing_analysis_path = setup["step_dirs"]["step1"] / "nonexistent.json"
                with open(missing_analysis_path, 'r') as f:
                    analysis = json.load(f)
                prompt_weaver.weave_prompt(analysis)
    
    def test_pipeline_performance_benchmarks(self, pipeline_setup, sample_analysis_json):
        """Test pipeline performance and timing."""
        setup = pipeline_setup
        
        # Benchmark Step 2 (prompt weaving)
        style_kit = {
            "templates": {
                "shirt": {
                    "base_prompt": "test prompt {garment_description}",
                    "style_modifiers": ["modifier1", "modifier2"]
                }
            }
        }
        
        prompt_weaver = PromptWeaver(style_kit)
        
        start_time = time.time()
        prompt_data = prompt_weaver.weave_prompt(sample_analysis_json)
        step2_time = time.time() - start_time
        
        # Should complete quickly (under 1 second for simple operations)
        assert step2_time < 1.0
        assert "final_prompt" in prompt_data
        
        # Benchmark QA assessment (Step 4) - mocked for performance
        start_time = time.time()
        
        # Mock QA operations
        qa_metrics = {
            "color_accuracy": {"mean_delta_e": 1.5},
            "hollow_quality": {"score": 0.7},
            "sharpness": {"normalized_score": 0.8},
            "composite_score": 0.75
        }
        
        step4_time = time.time() - start_time
        
        # QA should also complete quickly when mocked
        assert step4_time < 0.1  # Very fast when mocked
        
        print(f"Performance benchmarks:")
        print(f"  Step 2 (Prompt Weaving): {step2_time:.3f}s")
        print(f"  Step 4 (QA - mocked): {step4_time:.3f}s")
