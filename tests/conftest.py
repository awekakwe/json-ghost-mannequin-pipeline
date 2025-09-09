"""
Pytest configuration and fixtures for the JSON Ghost-Mannequin Pipeline tests.
"""

import os
import json
import pytest
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Any

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory path."""
    TEST_DATA_DIR.mkdir(exist_ok=True)
    return TEST_DATA_DIR


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create a sample garment image for testing."""
    # Create a simple 512x512 RGB image with a basic garment shape
    img = Image.new('RGB', (512, 512), color='white')
    
    # Create a simple shirt-like shape (rectangle with sleeves)
    pixels = np.array(img)
    
    # Main body (200x300 rectangle, centered)
    body_start_x, body_start_y = 156, 106  # Centered horizontally, offset vertically
    body_end_x, body_end_y = 356, 406
    pixels[body_start_y:body_end_y, body_start_x:body_end_x] = [70, 130, 180]  # Steel blue
    
    # Left sleeve (100x150 rectangle)
    sleeve_start_x, sleeve_start_y = 56, 106
    sleeve_end_x, sleeve_end_y = 156, 256
    pixels[sleeve_start_y:sleeve_end_y, sleeve_start_x:sleeve_end_x] = [70, 130, 180]
    
    # Right sleeve (100x150 rectangle)
    sleeve_start_x, sleeve_start_y = 356, 106
    sleeve_end_x, sleeve_end_y = 456, 256
    pixels[sleeve_start_y:sleeve_end_y, sleeve_start_x:sleeve_end_x] = [70, 130, 180]
    
    return Image.fromarray(pixels)


@pytest.fixture
def sample_mask():
    """Create a sample garment mask for testing."""
    # Create a binary mask matching the sample image
    mask = Image.new('L', (512, 512), color=0)  # Black background
    pixels = np.array(mask)
    
    # White mask for garment areas
    # Main body
    pixels[106:406, 156:356] = 255
    # Left sleeve
    pixels[106:256, 56:156] = 255
    # Right sleeve
    pixels[106:256, 356:456] = 255
    
    return Image.fromarray(pixels, mode='L')


@pytest.fixture
def sample_analysis_json():
    """Create sample analysis JSON data."""
    return {
        "garment_type": "shirt",
        "dominant_color": {
            "rgb": [70, 130, 180],
            "hex": "#4682B4",
            "name": "steel_blue",
            "lab": [52.5, -4.6, -32.2]
        },
        "fabric_analysis": {
            "texture_complexity": 0.3,
            "pattern_type": "solid",
            "fabric_type": "cotton",
            "sheen_level": 0.2
        },
        "construction_analysis": {
            "garment_category": "tops",
            "style": "basic_shirt",
            "sleeve_type": "short",
            "neckline": "crew",
            "fit": "regular"
        },
        "quality_metrics": {
            "image_sharpness": 0.75,
            "color_consistency": 0.85,
            "segmentation_confidence": 0.92,
            "overall_quality": 0.84
        },
        "processing_metadata": {
            "timestamp": "2024-12-08T21:00:00Z",
            "pipeline_version": "1.1.0",
            "processing_time_ms": 2500
        }
    }


@pytest.fixture
def sample_prompt_data():
    """Create sample prompt weaving data."""
    return {
        "base_prompt": "professional product photography, steel blue cotton shirt, ghost mannequin effect, clean white background, studio lighting",
        "style_modifiers": [
            "crisp fabric texture",
            "natural draping",
            "sharp focus",
            "commercial quality"
        ],
        "negative_prompt": "wrinkled fabric, shadows, colored background, model, person, hands",
        "final_prompt": "professional product photography, steel blue cotton shirt, ghost mannequin effect, clean white background, studio lighting, crisp fabric texture, natural draping, sharp focus, commercial quality",
        "variations": [
            {
                "id": 1,
                "prompt": "professional product photography, steel blue cotton shirt, invisible mannequin, pristine white background, commercial lighting, crisp fabric texture, natural draping, sharp focus, premium quality"
            }
        ]
    }


@pytest.fixture
def sample_qa_metrics():
    """Create sample QA metrics data."""
    return {
        "color_accuracy": {
            "mean_delta_e": 1.8,
            "p95_delta_e": 2.4,
            "passes_threshold": True
        },
        "hollow_quality": {
            "score": 0.65,
            "passes_threshold": True
        },
        "sharpness": {
            "laplacian_variance": 1250.5,
            "normalized_score": 0.42,
            "passes_threshold": True
        },
        "background": {
            "whiteness_score": 0.95,
            "purity_score": 0.92,
            "passes_threshold": True
        },
        "composite_score": 0.73,
        "passes_all_gates": True
    }


@pytest.fixture
def mock_style_kit():
    """Create mock style kit configuration."""
    return {
        "templates": {
            "shirt": {
                "base_prompt": "professional product photography, {garment_description}, ghost mannequin effect, clean white background, studio lighting",
                "style_modifiers": [
                    "crisp fabric texture",
                    "natural draping",
                    "sharp focus",
                    "commercial quality"
                ],
                "negative_prompt": "wrinkled fabric, shadows, colored background, model, person, hands"
            }
        },
        "guardrails": {
            "forbidden_terms": ["model", "person", "human", "wearing"],
            "required_terms": ["white background", "professional", "commercial"]
        },
        "rendering": {
            "sdxl": {
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
                "strength": 0.8
            }
        }
    }


@pytest.fixture
def mock_tuning_rules():
    """Create mock auto-tuning rules."""
    return {
        "rules": [
            {
                "name": "high_delta_e_guidance_increase",
                "trigger_condition": "high_delta_e",
                "parameter": "guidance_scale",
                "adjustment": 0.5,
                "confidence": 0.8,
                "conditions": {
                    "avg_delta_e_threshold": 2.5,
                    "min_runs": 5
                }
            }
        ]
    }


@pytest.fixture
def sample_delivery_manifest():
    """Create sample delivery manifest data."""
    return {
        "product_info": {
            "sku": "TEST001",
            "variant": "blue",
            "template_id": "ghost_mannequin_v1"
        },
        "assets": {
            "master": [
                {
                    "filename": "TEST001_blue_master.png",
                    "format": "PNG",
                    "dimensions": [2048, 2048],
                    "file_size_bytes": 4567890,
                    "sha256": "abc123def456789"
                }
            ],
            "web": [
                {
                    "filename": "TEST001_blue_1200.webp",
                    "format": "WebP",
                    "dimensions": [1200, 1200],
                    "file_size_bytes": 234567,
                    "quality": 85
                }
            ]
        },
        "processing_metadata": {
            "pipeline_version": "1.1.0",
            "route": "A",
            "processing_time_seconds": 45.2,
            "qa_passed": True
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ['TESTING'] = '1'
    os.environ['GOOGLE_API_KEY'] = 'test-api-key'
    os.environ['HF_TOKEN'] = 'test-hf-token'
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_file_system(temp_dir):
    """Create a mock file system structure for testing."""
    structure = {
        'step1': temp_dir / 'step1',
        'step2': temp_dir / 'step2',
        'step3': temp_dir / 'step3',
        'step4': temp_dir / 'step4',
        'step5': temp_dir / 'step5',
        'step6': temp_dir / 'step6'
    }
    
    for step_dir in structure.values():
        step_dir.mkdir(parents=True, exist_ok=True)
    
    return structure


# Helper functions for tests
def create_test_image(width=512, height=512, color=(128, 128, 128)):
    """Create a test image with specified dimensions and color."""
    return Image.new('RGB', (width, height), color=color)


def create_test_mask(width=512, height=512, mask_value=255):
    """Create a test mask with specified dimensions."""
    return Image.new('L', (width, height), color=mask_value)


def save_test_json(data: Dict[str, Any], filepath: Path):
    """Save test data as JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
