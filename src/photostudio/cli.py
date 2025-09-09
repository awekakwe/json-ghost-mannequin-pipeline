#!/usr/bin/env python3
"""
CLI entry point for the JSON Ghost-Mannequin Pipeline.

Provides convenient command-line interfaces for all pipeline steps.
"""

import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from photostudio.steps.step0_preclean import main as step0_main
from photostudio.steps.step1_color_refine import main as step1_color_refine_main
from photostudio.steps.step1_preprocess import main as step1_main
from photostudio.steps.step2_prompt_weaver import main as step2_main
from photostudio.steps.step3_renderer import main as step3_main
from photostudio.steps.step4_qa_corrector import main as step4_main
from photostudio.steps.step5_delivery_engine import main as step5_main
from photostudio.steps.step6_feedback_engine import main as step6_main


def photostudio_preclean():
    """CLI entry point for Step 0: Background Removal & Pre-cleaning."""
    step0_main()


def photostudio_preprocess():
    """CLI entry point for Step 1: Preprocessing & JSON Generation."""
    step1_main()


def photostudio_weave():
    """CLI entry point for Step 2: Prompt Weaving."""
    step2_main()


def photostudio_render():
    """CLI entry point for Step 3: SDXL/ControlNet Rendering."""
    step3_main()


def photostudio_qa():
    """CLI entry point for Step 4: Auto-QA & Correction."""
    step4_main()


def photostudio_deliver():
    """CLI entry point for Step 5: Package & Publish."""
    step5_main()


def photostudio_feedback():
    """CLI entry point for Step 6: Feedback & Auto-Tuning."""
    step6_main()


def photostudio_pipeline():
    """Run the complete end-to-end pipeline."""
    import argparse
    import json
    import time
    import uuid
    from datetime import datetime
    
    parser = argparse.ArgumentParser(
        description="Complete JSON Ghost-Mannequin Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    photostudio-pipeline --input image.jpg --sku ABC123 --variant red
    
    # Run with custom configuration
    photostudio-pipeline --input image.jpg --sku ABC123 --variant red --config pipeline.yml
    
    # Run with Route B (Gemini fallback)
    photostudio-pipeline --input image.jpg --sku ABC123 --variant red --route B
        """
    )
    
    # Core arguments
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--sku", required=True, help="Product SKU")
    parser.add_argument("--variant", required=True, help="Product variant")
    parser.add_argument("--out", default="./pipeline_output", help="Output directory")
    
    # Pipeline configuration
    parser.add_argument("--config", help="Pipeline configuration YAML file")
    parser.add_argument("--route", choices=["A", "B"], default="A", help="Rendering route (A=SDXL, B=Gemini)")
    parser.add_argument("--template-id", default="ghost_mannequin_v1", help="Template ID")
    parser.add_argument("--run-id", help="Custom run ID (auto-generated if not provided)")
    
    # Optional overrides
    parser.add_argument("--skip-feedback", action="store_true", help="Skip feedback step")
    parser.add_argument("--local-only", action="store_true", help="Use local cached models only (no network)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Generate run ID if not provided
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("photostudio.pipeline")
    
    # Create output directory structure
    out_dir = Path(args.out)
    step0_dir = out_dir / "step0"
    step1_dir = out_dir / "step1"
    step2_dir = out_dir / "step2" 
    step3_dir = out_dir / "step3"
    step4_dir = out_dir / "step4"
    step5_dir = out_dir / "step5"
    step6_dir = out_dir / "step6"
    
    for d in [step0_dir, step1_dir, step2_dir, step3_dir, step4_dir, step5_dir, step6_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    pipeline_start_time = time.time()
    
    try:
        logger.info(f"🚀 Starting pipeline run: {run_id}")
        logger.info(f"📸 Input: {args.input} (SKU: {args.sku}-{args.variant})")
        
        # Step 0: Background Removal & Pre-cleaning
        logger.info("🧹 Step 0: Background Removal & Pre-cleaning...")
        step0_args = [
            "--image", args.input,
            "--out", str(step0_dir),
            "--target-size", "2048",
            "--backend", "auto"
        ]
        if args.debug:
            step0_args.append("--save-intermediates")
        
        # Execute step 0
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["photostudio-preclean"] + step0_args
            step0_main()
        finally:
            sys.argv = old_argv
        logger.info("✅ Step 0 complete")
        
        # Step 1: Preprocessing & JSON Generation (now uses cleaned image)
        logger.info("📋 Step 1: Preprocessing & JSON Generation...")
        step1_args = [
            "--image", str(step0_dir / "control_rgb.png"),  # Use cleaned image from Step 0
            "--out", str(step1_dir)
        ]
        # Note: Step 1 doesn't have a --debug flag, so we don't add it
        
        # Execute step 1
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["photostudio-preprocess"] + step1_args
            step1_main()
        finally:
            sys.argv = old_argv
        logger.info("✅ Step 1 complete")
        
        # Step 1.5: Color Refinement (using Step 0 masks)
        logger.info("🎨 Step 1.5: Advanced Color Refinement...")
        step1_5_args = [
            "--step0", str(step0_dir),
            "--json-in", str(step1_dir / "garment_analysis.json"),
            "--json-out", str(step1_dir / "garment_analysis.json")  # Update in place
        ]
        try:
            sys.argv = ["photostudio-color-refine"] + step1_5_args
            step1_color_refine_main()
        finally:
            sys.argv = old_argv
        logger.info("✅ Step 1.5 complete - Colors refined with regional analysis")
        
        # Step 2: Prompt Weaving
        logger.info("🔀 Step 2: Prompt Weaving...")
        step2_args = [
            "--json", str(step1_dir / "garment_analysis.json"),
            "--out", str(step2_dir)
        ]
        try:
            sys.argv = ["photostudio-weave"] + step2_args
            step2_main()
        finally:
            sys.argv = old_argv
        logger.info("✅ Step 2 complete")
        
        # Step 3: Rendering
        logger.info(f"🎨 Step 3: Rendering (Route {args.route})...")
        step3_args = [
            "--input_dir", str(step1_dir),
            "--json", str(step1_dir / "garment_analysis.json"),
            "--prompt", str(step2_dir / "prompt.txt"),
            "--route", args.route,
            "--out_dir", str(step3_dir)
        ]
        if args.local_only:
            step3_args.append("--local-only")
        try:
            sys.argv = ["photostudio-render"] + step3_args
            step3_main()
        finally:
            sys.argv = old_argv
        logger.info("✅ Step 3 complete")
        
        # Step 4: Auto-QA & Correction
        logger.info("🔍 Step 4: Auto-QA & Correction...")
        step4_args = [
            "--candidates-dir", str(step3_dir),
            "--analysis-json", str(step1_dir / "garment_analysis.json"),
            "--mask", str(step1_dir / "mask.png"),
            "--out", str(step4_dir)
        ]
        try:
            sys.argv = ["photostudio-qa"] + step4_args
            step4_main()
        finally:
            sys.argv = old_argv
        logger.info("✅ Step 4 complete")
        
        # Step 5: Package & Publish
        logger.info("📦 Step 5: Package & Publish...")
        step5_args = [
            "--final", str(step4_dir / "final.png"),
            "--qa", str(step4_dir / "qa_report.json"),
            "--json", str(step1_dir / "garment_analysis.json"),
            "--sku", args.sku,
            "--variant", args.variant,
            "--out", str(step5_dir)
        ]
        try:
            sys.argv = ["photostudio-deliver"] + step5_args
            step5_main()
        finally:
            sys.argv = old_argv
        logger.info("✅ Step 5 complete")
        
        # Step 6: Feedback & Auto-Tuning
        if not args.skip_feedback:
            logger.info("📁 Step 6: Feedback & Auto-Tuning...")
            processing_time = time.time() - pipeline_start_time
            step6_args = [
                "--run-id", run_id,
                "--sku", args.sku,
                "--variant", args.variant,
                "--route", args.route,
                "--template-id", args.template_id,
                "--qa", str(step4_dir / "qa_report.json"),
                "--manifest", str(step5_dir / "manifests" / "delivery_manifest.json"),
                "--processing-time", str(processing_time),
                "--out", str(step6_dir)
            ]
            try:
                sys.argv = ["photostudio-feedback"] + step6_args
                step6_main()
            finally:
                sys.argv = old_argv
            logger.info("✅ Step 6 complete")
        
        total_time = time.time() - pipeline_start_time
        
        # Create pipeline summary
        summary = {
            "run_id": run_id,
            "input_image": str(args.input),
            "sku": args.sku,
            "variant": args.variant,
            "route": args.route,
            "template_id": args.template_id,
            "processing_time_seconds": round(total_time, 2),
            "output_directory": str(out_dir),
            "steps_completed": 6 if not args.skip_feedback else 5,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        summary_path = out_dir / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETE!")
        print("="*60)
        print(f"📍 Run ID: {run_id}")
        print(f"⏱️  Total Time: {total_time:.1f}s")
        print(f"📂 Output: {out_dir}")
        print(f"📋 Summary: {summary_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {e}")
        
        # Create failure summary
        summary = {
            "run_id": run_id,
            "input_image": str(args.input),
            "sku": args.sku,
            "variant": args.variant,
            "processing_time_seconds": round(time.time() - pipeline_start_time, 2),
            "output_directory": str(out_dir),
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e)
        }
        
        summary_path = out_dir / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        raise


if __name__ == "__main__":
    photostudio_pipeline()
