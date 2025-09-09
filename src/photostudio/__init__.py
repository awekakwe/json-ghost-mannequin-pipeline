"""
Photostudio.io Ghost-Mannequin Pipeline (Steps 1–6)
===================================================

A deterministic, 6-phase pipeline that turns garment images into professional,
consistent ghost-mannequin renders with automated QA, packaging, and learning loops.

Core Pipeline:
1. Preprocess & JSON Analysis
2. Prompt Weaver  
3. Renderer (SDXL/ControlNet + Gemini fallback)
4. Auto-QA & Correction
5. Package & Publish
6. Feedback & Auto-Tuning

Version: 1.1.0
"""

__version__ = "1.1.0"
__author__ = "Photostudio.io"

from .steps import (
    step1_preprocess,
    step2_prompt_weaver, 
    step3_renderer,
    step4_qa_corrector,
    step5_delivery_engine,
    step6_feedback_engine,
)

__all__ = [
    "step1_preprocess",
    "step2_prompt_weaver", 
    "step3_renderer",
    "step4_qa_corrector",
    "step5_delivery_engine",
    "step6_feedback_engine",
]
