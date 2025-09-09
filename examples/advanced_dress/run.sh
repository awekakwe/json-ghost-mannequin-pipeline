#!/bin/bash
# Advanced Dress Processing Example
# Demonstrates custom configuration and Route B fallback

set -e

echo "🚀 Advanced Dress Processing Example"
echo "====================================="

# Configuration
INPUT_IMAGE="input.jpg"
SKU="DRESS001"
VARIANT="floral"
OUTPUT_DIR="./output"
CUSTOM_CONFIG="custom_config.yml"

# Check if input image exists
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "❌ Error: Input image '$INPUT_IMAGE' not found!"
    echo "Please add a sample dress image as 'input.jpg' in this directory."
    exit 1
fi

# Check if custom config exists
if [ ! -f "$CUSTOM_CONFIG" ]; then
    echo "❌ Error: Custom config '$CUSTOM_CONFIG' not found!"
    echo "Please ensure the custom configuration file exists."
    exit 1
fi

# Clean previous output
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "📸 Processing: $INPUT_IMAGE (SKU: $SKU-$VARIANT)"
echo "⚙️  Using custom config: $CUSTOM_CONFIG"
echo "📂 Output: $OUTPUT_DIR"
echo ""

# First, try Route A with custom configuration
echo "🎨 Attempting Route A (SDXL) with custom styling..."
if photostudio-pipeline \
    --input "$INPUT_IMAGE" \
    --sku "$SKU" \
    --variant "$VARIANT" \
    --out "${OUTPUT_DIR}_route_a" \
    --route A \
    --template-id dress_elegant \
    --config "$CUSTOM_CONFIG" \
    --debug; then
    
    echo "✅ Route A successful!"
    FINAL_OUTPUT="${OUTPUT_DIR}_route_a"
else
    echo "⚠️  Route A failed, falling back to Route B (Gemini)..."
    
    # Fallback to Route B
    photostudio-pipeline \
        --input "$INPUT_IMAGE" \
        --sku "$SKU" \
        --variant "$VARIANT" \
        --out "${OUTPUT_DIR}_route_b" \
        --route B \
        --template-id dress_elegant \
        --config "$CUSTOM_CONFIG" \
        --debug
    
    echo "✅ Route B fallback successful!"
    FINAL_OUTPUT="${OUTPUT_DIR}_route_b"
fi

# Create summary output directory
mv "$FINAL_OUTPUT" "$OUTPUT_DIR"

echo ""
echo "✅ Advanced processing complete!"
echo ""
echo "📋 Results:"
echo "  📁 Final Output: $OUTPUT_DIR/"
echo "  📸 Best Image: $OUTPUT_DIR/step4/corrected_best.png"
echo "  📊 QA Metrics: $OUTPUT_DIR/step4/qa_report.json"
echo "  🎨 Web Assets: $OUTPUT_DIR/step5/web/"
echo "  📈 Analytics: $OUTPUT_DIR/step6/summary.json"
echo ""

# Quality analysis
if [ -f "$OUTPUT_DIR/step4/qa_report.json" ]; then
    echo "📊 Quality Analysis:"
    
    # Extract key metrics using jq if available
    if command -v jq >/dev/null 2>&1; then
        echo "  🎯 Color Accuracy (ΔE): $(cat $OUTPUT_DIR/step4/qa_report.json | jq -r '.best_candidate.final_metrics.mean_delta_e // "N/A"')"
        echo "  👻 Hollow Quality: $(cat $OUTPUT_DIR/step4/qa_report.json | jq -r '.best_candidate.final_metrics.hollow_quality // "N/A"')"
        echo "  📐 Sharpness: $(cat $OUTPUT_DIR/step4/qa_report.json | jq -r '.best_candidate.final_metrics.sharpness // "N/A"')"
        echo "  🏳️  Background Score: $(cat $OUTPUT_DIR/step4/qa_report.json | jq -r '.best_candidate.final_metrics.background_score // "N/A"')"
        echo "  📈 Composite Score: $(cat $OUTPUT_DIR/step4/qa_report.json | jq -r '.best_candidate.final_metrics.composite_score // "N/A"')"
    else
        echo "  📄 Full report: cat $OUTPUT_DIR/step4/qa_report.json"
    fi
fi

echo ""
echo "🚀 Performance Summary:"
if [ -f "$OUTPUT_DIR/pipeline_summary.json" ] && command -v jq >/dev/null 2>&1; then
    echo "  ⏱️  Processing Time: $(cat $OUTPUT_DIR/pipeline_summary.json | jq -r '.processing_time_seconds // "N/A"')s"
    echo "  🛤️  Route Used: $(cat $OUTPUT_DIR/pipeline_summary.json | jq -r '.route // "N/A"')"
    echo "  ✅ Status: $(cat $OUTPUT_DIR/pipeline_summary.json | jq -r '.status // "N/A"')"
fi

echo ""
echo "🔍 View Results:"
echo "  open $OUTPUT_DIR/step4/corrected_best.png"
echo "  open $OUTPUT_DIR/step5/web/"
