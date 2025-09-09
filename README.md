# JSON Ghost-Mannequin Pipeline

A sophisticated 6-step pipeline for automated ghost-mannequin photography processing, designed for e-commerce fashion photography at scale.

## 🎯 Overview

Transform garment product photos into professional ghost-mannequin images using AI-powered processing with advanced quality control, auto-tuning, and comprehensive analytics.

### Pipeline Steps
1. **Preprocess & JSON Generation** - Advanced segmentation, fabric analysis, color extraction
2. **Prompt Weaving** - Convert analysis to structured natural language prompts
3. **SDXL/ControlNet Rendering** - Dual-route AI rendering with Route A (SDXL) and Route B (Gemini)
4. **Auto-QA & Correction** - Quality scoring, color correction, and best candidate selection
5. **Package & Publish** - Multi-format output with cloud storage integration
6. **Feedback & Auto-Tuning** - Telemetry, health monitoring, and intelligent parameter optimization

## 🚀 Quick Start

### Installation

```bash
git clone <repository-url>
cd json-ghost-mannequin-pipeline
pip install -e .
```

### Environment Setup

```bash
# Required API keys
export GOOGLE_API_KEY="your-gemini-api-key"
export HF_TOKEN="your-huggingface-token"

# Optional cloud storage
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"
export GCP_SERVICE_ACCOUNT_PATH="path/to/gcp-credentials.json"
```

### Basic Usage

```bash
# Run complete pipeline
photostudio-pipeline --input shirt.jpg --sku ABC123 --variant red

# Run individual steps
photostudio-preprocess --input shirt.jpg --out ./step1
photostudio-weave --json ./step1/analysis.json --out ./step2
photostudio-render --image shirt.jpg --mask ./step1/garment_mask.png --prompt ./step2/final_prompt.txt --out ./step3
```

## 📋 Detailed Usage

### Complete Pipeline

The main `photostudio-pipeline` command orchestrates all 6 steps:

```bash
photostudio-pipeline \
  --input /path/to/garment.jpg \
  --sku SHIRT001 \
  --variant blue \
  --out ./results \
  --route A \
  --template-id ghost_mannequin_v1
```

**Options:**
- `--input` - Input garment image path (required)
- `--sku` - Product SKU identifier (required)  
- `--variant` - Product variant (color/style) (required)
- `--out` - Output directory (default: `./pipeline_output`)
- `--route` - Rendering route: A (SDXL) or B (Gemini) (default: A)
- `--template-id` - Template identifier for style consistency
- `--run-id` - Custom run identifier (auto-generated if not provided)
- `--skip-feedback` - Skip feedback/analytics step
- `--debug` - Enable debug logging

### Individual Steps

#### Step 1: Preprocessing & JSON Generation

```bash
photostudio-preprocess \
  --input garment.jpg \
  --out ./step1 \
  --gemini-analysis \
  --debug
```

**Outputs:**
- `analysis.json` - Comprehensive garment analysis
- `garment_mask.png` - Segmentation mask
- `quality_report.json` - Quality metrics

#### Step 2: Prompt Weaving

```bash
photostudio-weave \
  --json ./step1/analysis.json \
  --style-kit ./config/style_kit.yml \
  --variations 3 \
  --out ./step2
```

**Outputs:**
- `final_prompt.txt` - Primary structured prompt
- `prompt_variations.json` - A/B test variations

#### Step 3: Rendering

```bash
photostudio-render \
  --image garment.jpg \
  --mask ./step1/garment_mask.png \
  --json ./step1/analysis.json \
  --prompt ./step2/final_prompt.txt \
  --route A \
  --candidates 4 \
  --out ./step3
```

**Outputs:**
- `candidate_*.png` - Generated image candidates
- `render_metadata.json` - Generation parameters and QA

#### Step 4: Auto-QA & Correction

```bash
photostudio-qa \
  --candidates ./step3 \
  --reference-json ./step1/analysis.json \
  --out ./step4
```

**Outputs:**
- `qa_report.json` - Detailed quality analysis
- `corrected_best.png` - Best candidate with corrections

#### Step 5: Package & Publish

```bash
photostudio-deliver \
  --best-image ./step4/corrected_best.png \
  --qa-report ./step4/qa_report.json \
  --sku SHIRT001 \
  --variant blue \
  --cloud-provider aws \
  --out ./step5
```

**Outputs:**
- `master/` - High-resolution PNG with ICC profiles
- `web/` - WebP variants (1200px, 800px, 400px)
- `thumbnails/` - Thumbnail sizes
- `delivery_manifest.json` - Complete asset inventory

#### Step 6: Feedback & Auto-Tuning

```bash
photostudio-feedback \
  --run-id run_20241208_143022_a1b2c3d4 \
  --sku SHIRT001 \
  --variant blue \
  --route A \
  --template-id ghost_mannequin_v1 \
  --qa ./step4/qa_report.json \
  --manifest ./step5/delivery_manifest.json \
  --processing-time 45.2 \
  --out ./step6
```

**Outputs:**
- `summary.json` - Run summary with health metrics
- `feedback.db` - Telemetry database (SQLite)

## 📊 Quality Metrics

The pipeline tracks comprehensive quality metrics:

- **Color Accuracy** - ΔE 2000 measurements in LAB space
- **Hollow Quality** - Ghost-mannequin effect assessment via luminance analysis  
- **Sharpness** - Laplacian variance for image clarity
- **Background Purity** - White background quality score
- **Composite Score** - Weighted overall quality metric

### Quality Gates

Images must pass all quality gates:
- ΔE ≤ 2.0 (excellent color accuracy)
- Hollow Quality ≥ 0.4 (clear ghost-mannequin effect)
- Sharpness ≥ 0.3 (adequate image clarity)
- Background Score ≥ 0.8 (clean white background)

## 🔧 Configuration

### Style Kit Configuration

Configure prompt templates, style guardrails, and rendering parameters in `config/style_kit.yml`:

```yaml
# Example style configuration
templates:
  shirt:
    base_prompt: "professional product photography, {garment_description}, ghost mannequin effect, clean white background"
    style_modifiers:
      - "crisp fabric texture"
      - "natural draping" 
      - "commercial quality"
    negative_prompt: "wrinkled fabric, shadows, model, person"

guardrails:
  forbidden_terms: ["model", "person", "wearing"]
  required_terms: ["white background", "professional"]

rendering:
  sdxl:
    guidance_scale: 7.5
    num_inference_steps: 50
```

### Auto-Tuning Rules

Define automatic parameter adjustments in `config/tuning_rules.yml`:

```yaml
rules:
  - name: "high_delta_e_guidance_increase"
    trigger_condition: "high_delta_e"
    parameter: "guidance_scale"
    adjustment: 0.5
    confidence: 0.8
    conditions:
      avg_delta_e_threshold: 2.5
      min_runs: 5
```

## 📈 Analytics & Monitoring

### Health Dashboard

Monitor pipeline health through comprehensive metrics:

```bash
# View recent health metrics
photostudio-feedback --run-id latest --health-report

# Export analytics data
photostudio-feedback --export-analytics --format json --out analytics.json
```

### Alert Configuration

Configure alerts for health degradation:

```bash
# Enable Slack alerts
photostudio-pipeline \
  --input shirt.jpg \
  --sku ABC123 \
  --variant red \
  --enable-alerts \
  --webhook-url https://hooks.slack.com/your-webhook
```

## 🏗️ Architecture

### Route A: SDXL + ControlNet Pipeline
- **SDXL 1.0** - Base diffusion model
- **ControlNet** - Structure guidance via depth/edge maps
- **IP-Adapter** - Image-to-image style transfer
- **Advanced QA** - Multi-metric quality assessment

### Route B: Gemini Vision Fallback  
- **Gemini Pro Vision** - Google's multimodal AI
- **Structured prompting** - JSON-guided generation
- **Quality validation** - Same QA pipeline as Route A

### Quality Assurance
- **Color Science** - CIEDE2000 ΔE calculations in LAB space
- **Perceptual Metrics** - Sharpness via Laplacian variance
- **Semantic Analysis** - Hollow effect detection using luminance maps
- **Auto-Correction** - Surgical color/background corrections

## 🧪 Examples

### Example Workflow Files

The `examples/` directory contains sample workflows:

```
examples/
├── basic_shirt/
│   ├── input.jpg          # Sample shirt image
│   ├── run.sh            # Basic processing script
│   └── expected_output/   # Reference outputs
├── advanced_dress/
│   ├── input.jpg          # Complex dress example
│   ├── custom_config.yml  # Custom style settings
│   └── run.sh            # Advanced processing
└── batch_processing/
    ├── batch_run.sh       # Process multiple images
    └── inputs/           # Multiple test images
```

### Running Examples

```bash
# Basic shirt example
cd examples/basic_shirt
./run.sh

# Advanced dress with custom styling
cd examples/advanced_dress  
./run.sh

# Batch processing
cd examples/batch_processing
./batch_run.sh
```

## 🔬 Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run specific step tests
pytest tests/test_step1_preprocess.py
pytest tests/test_step3_renderer.py

# Run with coverage
pytest --cov=photostudio tests/
```

### Integration Tests

```bash
# Test complete pipeline
pytest tests/integration/test_full_pipeline.py

# Test quality gates
pytest tests/integration/test_quality_validation.py
```

### Quality Validation

```bash
# Validate against reference dataset
pytest tests/quality/test_reference_dataset.py

# Performance benchmarks
pytest tests/performance/test_processing_speed.py
```

## 📦 Dependencies

### Core Dependencies
- **opencv-python** - Image processing and computer vision
- **Pillow** - Image manipulation and format support
- **torch** - PyTorch for deep learning models
- **diffusers** - Hugging Face diffusion models (SDXL)
- **controlnet-aux** - ControlNet preprocessing
- **google-generativeai** - Gemini API integration

### Optional Dependencies
- **boto3** - AWS S3 integration
- **google-cloud-storage** - Google Cloud Storage
- **sqlalchemy** - Database ORM for telemetry
- **requests** - HTTP client for webhooks
- **pytest** - Testing framework

### Development Dependencies
- **black** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking

## 🚨 Troubleshooting

### Common Issues

**GPU Memory Issues**
```bash
# Reduce batch size or use CPU fallback
export CUDA_VISIBLE_DEVICES=""
photostudio-pipeline --input shirt.jpg --sku ABC123 --variant red --route B
```

**API Rate Limits**
```bash
# Use built-in retry logic and rate limiting
photostudio-pipeline --input shirt.jpg --sku ABC123 --variant red --retry-delays 1,2,4
```

**Quality Gate Failures**
```bash
# Lower quality thresholds temporarily
photostudio-qa --candidates ./step3 --delta-e-threshold 3.0 --hollow-threshold 0.3 --out ./step4
```

### Debug Mode

Enable detailed logging:

```bash
photostudio-pipeline --input shirt.jpg --sku ABC123 --variant red --debug
```

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Email: support@photostudio.io
- Documentation: [docs.photostudio.io](https://docs.photostudio.io)

---

**Built with ❤️ for fashion e-commerce at scale**
