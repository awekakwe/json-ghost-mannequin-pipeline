"""
Pipeline Steps Module
====================

Contains the 6 core pipeline steps:
- step1_preprocess: Advanced image segmentation & JSON generation
- step2_prompt_weaver: JSON to sectioned natural-language prompts
- step3_renderer: SDXL/ControlNet + Gemini fallback rendering
- step4_qa_corrector: Auto-QA, color correction, and candidate selection
- step5_delivery_engine: Master PNG + web variants packaging
- step6_feedback_engine: Telemetry, health scoring, and auto-tuning
"""

# Import step modules when they're created
# from . import step1_preprocess
# from . import step2_prompt_weaver
# from . import step3_renderer
# from . import step4_qa_corrector
# from . import step5_delivery_engine
# from . import step6_feedback_engine

__all__ = [
    # Will be populated as modules are implemented
]
