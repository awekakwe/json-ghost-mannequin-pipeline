# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Architecture Overview

This is the **JSON Ghost-Mannequin Pipeline** - a sophisticated 6-step AI-powered system for automated e-commerce fashion photography processing. The pipeline transforms garment product photos into professional ghost-mannequin images using computer vision, machine learning, and generative AI.

### Core Pipeline Flow
1. **Step 1**: Preprocess & JSON Generation - Advanced segmentation, fabric analysis, color extraction
2. **Step 2**: Prompt Weaving - Convert analysis to structured natural language prompts  
3. **Step 3**: SDXL/ControlNet Rendering - Dual-route AI rendering (Route A: SDXL, Route B: Gemini)
4. **Step 4**: Auto-QA & Correction - Quality scoring, color correction, best candidate selection
5. **Step 5**: Package & Publish - Multi-format output with cloud storage integration
6. **Step 6**: Feedback & Auto-Tuning - Telemetry, health monitoring, intelligent parameter optimization

### Dual Route Architecture
- **Route A**: SDXL 1.0 + ControlNet + IP-Adapter for high-quality diffusion-based rendering
- **Route B**: Gemini Pro Vision fallback for performance/compatibility scenarios
- Auto-switching based on quality metrics and performance thresholds

### Quality Assurance System
The pipeline includes comprehensive quality validation:
- **Color Accuracy**: CIEDE2000 ΔE calculations in LAB color space
- **Hollow Quality**: Ghost-mannequin effect assessment via luminance analysis
- **Sharpness**: Laplacian variance for image clarity measurement
- **Background Purity**: White background quality scoring
- **Composite Score**: Weighted overall quality metric

Quality gates enforce: ΔE ≤ 2.0, Hollow Quality ≥ 0.4, Sharpness ≥ 0.3, Background Score ≥ 0.8

## Key Development Commands

### Environment Setup
```bash
# Install with all optional dependencies
pip install -e ".[all]"

# Install for development  
pip install -e ".[dev]"

# Core dependencies only
pip install -e .

# Set required environment variables
export GOOGLE_API_KEY="your-gemini-api-key"
export HF_TOKEN="your-huggingface-token"
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/photostudio tests/

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m integration         # Integration tests only
pytest -m gpu                 # GPU-dependent tests only

# Run individual step tests
pytest tests/test_step1_preprocess.py
pytest tests/test_step3_renderer.py

# Quality validation tests
pytest tests/quality/test_reference_dataset.py
pytest tests/performance/test_processing_speed.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Pipeline Operations
```bash
# Run complete pipeline
photostudio-pipeline --input shirt.jpg --sku ABC123 --variant red

# Run individual steps
photostudio-preprocess --input shirt.jpg --out ./step1
photostudio-weave --json ./step1/analysis.json --out ./step2
photostudio-render --image shirt.jpg --mask ./step1/garment_mask.png --prompt ./step2/final_prompt.txt --route A --out ./step3

# Debug mode with detailed logging
photostudio-pipeline --input shirt.jpg --sku ABC123 --variant red --debug

# Use Gemini fallback route
photostudio-pipeline --input shirt.jpg --sku ABC123 --variant red --route B
```

### Development Workflow
```bash
# Create test image for development
python -c "from tests.conftest import create_test_image; create_test_image().save('test_garment.jpg')"

# Run pipeline on test data with debug output
photostudio-pipeline --input test_garment.jpg --sku TEST001 --variant test --debug --out ./debug_output

# Analyze quality metrics
photostudio-qa --candidates ./debug_output/step3 --reference-json ./debug_output/step1/analysis.json --out ./debug_output/step4
```

## Project Structure

### Core Source Code (`src/photostudio/`)
- `steps/step1_preprocess.py` - Advanced computer vision: GrabCut segmentation, LAB color analysis, fabric texture analysis via LBP, Gemini integration
- `steps/step2_prompt_weaver.py` - Template-based prompt generation with guardrails and A/B variations
- `steps/step3_renderer.py` - SDXL/ControlNet pipeline and Gemini Vision fallback
- `steps/step4_qa_corrector.py` - Multi-metric quality assessment and surgical corrections
- `steps/step5_delivery_engine.py` - Multi-format asset generation and cloud storage integration
- `steps/step6_feedback_engine.py` - Telemetry collection, health monitoring, auto-tuning rule engine

### Configuration System
- `config/style_kit.yml` - Prompt templates, style guardrails, rendering parameters by garment category
- `config/tuning_rules.yml` - Auto-tuning conditions and parameter adjustments with confidence scoring

### Entry Points
- `cli.py` - Main CLI entry point with complete pipeline orchestration
- `pyproject.toml` - Modern Python packaging with optional dependency groups

## Important Implementation Details

### Quality Metrics Implementation
The QA system uses scientifically-grounded metrics:
- Color accuracy measured in perceptually-uniform LAB space using CIEDE2000
- Hollow quality uses luminance-based analysis to detect ghost-mannequin effect
- Sharpness calculated via Laplacian variance with normalization
- Background purity combines whiteness and uniformity scoring

### Auto-Tuning System
Intelligent parameter optimization based on:
- Historical performance metrics with confidence decay
- Multi-condition rule triggering (fabric type, color, quality scores)
- Cooldown periods to prevent oscillation
- Emergency rules for critical failure scenarios
- Rollback capability when adjustments don't improve outcomes

### Dual Route Architecture
Route selection logic:
- Route A (SDXL): Default for highest quality, requires GPU
- Route B (Gemini): Fallback for speed, compatibility, or Route A failures
- Auto-switching based on processing time thresholds and quality gates
- Fabric-specific route recommendations (e.g., Gemini for complex textures)

### Configuration-Driven Design
- Garment-specific prompt templates with style modifiers
- Fabric-aware parameter adjustments (cotton, silk, denim, wool, polyester, linen)
- Quality thresholds configurable per deployment environment
- Template variations for A/B testing prompt effectiveness

## Testing Strategy

### Test Structure
- `tests/conftest.py` - Comprehensive fixtures for synthetic test data generation
- Unit tests for individual pipeline steps with mock dependencies
- Integration tests for complete pipeline execution
- Performance benchmarks with GPU/CPU fallback testing
- Quality validation against reference datasets

### Test Data Generation
The test system generates realistic synthetic data:
- Sample garment images with proper aspect ratios and colors
- Corresponding segmentation masks with realistic shapes
- Mock analysis JSON with proper schema compliance
- QA metrics with realistic quality scores and failure scenarios

## Dependencies & Optional Features

### Core Dependencies
- Computer Vision: opencv-python, Pillow, scikit-image
- ML/AI: torch, diffusers, transformers, google-generativeai
- Data Processing: numpy, scikit-learn, PyYAML
- CLI: click, rich, tqdm

### Optional Dependency Groups
- `generation`: SDXL/ControlNet dependencies (torch ecosystem)
- `gemini`: Google Gemini API integration
- `storage`: Cloud storage (boto3, google-cloud-storage)
- `database`: SQLAlchemy for telemetry persistence
- `dev`: Testing and development tools

### GPU Considerations
- SDXL Route A requires significant GPU memory
- CPU fallback available but slower
- xformers excluded on macOS due to compatibility issues
- Memory optimization through batch size reduction

## Development Guidelines

### Adding New Garment Categories
1. Add template in `config/style_kit.yml` under `templates:`
2. Update `GarmentCategory` enum in preprocessing step
3. Add fabric-specific adjustments if needed
4. Update test fixtures in `conftest.py`

### Modifying Quality Metrics
1. Update thresholds in `config/style_kit.yml` under `quality_thresholds:`
2. Modify QA step implementation for new metric calculation
3. Add corresponding test cases with expected ranges
4. Update auto-tuning rules to respond to new metrics

### Adding Auto-Tuning Rules
1. Define rule in `config/tuning_rules.yml` with trigger conditions
2. Specify parameter adjustments and confidence levels
3. Set appropriate cooldown periods to prevent oscillation
4. Test rule effectiveness with synthetic failure scenarios

### Pipeline Extension
Each step is designed as an independent module:
- Clear input/output contracts via JSON schemas
- Consistent CLI interface patterns
- Comprehensive error handling with recovery options
- Telemetry integration for monitoring and debugging

