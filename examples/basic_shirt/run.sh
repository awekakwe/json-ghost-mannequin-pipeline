#!/bin/bash
# Basic Shirt Processing Example
# Demonstrates simple ghost-mannequin pipeline usage

set -e

echo "🚀 Basic Shirt Processing Example"
echo "=================================="

# Configuration
INPUT_IMAGE="input.jpg"
SKU="SHIRT001"
VARIANT="blue"
OUTPUT_DIR="./output"

# Check if input image exists
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "❌ Error: Input image '$INPUT_IMAGE' not found!"
    echo "Please add a sample shirt image as 'input.jpg' in this directory."
    exit 1
fi

# Clean previous output
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "📸 Processing: $INPUT_IMAGE (SKU: $SKU-$VARIANT)"
echo "📂 Output: $OUTPUT_DIR"
echo ""

# Run the complete pipeline
echo "🔄 Running complete pipeline..."
photostudio-pipeline \
    --input "$INPUT_IMAGE" \
    --sku "$SKU" \
    --variant "$VARIANT" \
    --out "$OUTPUT_DIR" \
    --route A \
    --template-id ghost_mannequin_v1

echo ""
echo "✅ Processing complete!"
echo ""
echo "📋 Results:"
echo "  📁 Step 1 (Preprocessing): $OUTPUT_DIR/step1/"
echo "  📁 Step 2 (Prompt Weaving): $OUTPUT_DIR/step2/"
echo "  📁 Step 3 (Rendering): $OUTPUT_DIR/step3/"
echo "  📁 Step 4 (QA & Correction): $OUTPUT_DIR/step4/"
echo "  📁 Step 5 (Delivery): $OUTPUT_DIR/step5/"
echo "  📁 Step 6 (Feedback): $OUTPUT_DIR/step6/"
echo ""
echo "🎯 Key Output Files:"
echo "  📸 Final Image: $OUTPUT_DIR/step4/corrected_best.png"
echo "  📊 QA Report: $OUTPUT_DIR/step4/qa_report.json"
echo "  📦 Delivery Manifest: $OUTPUT_DIR/step5/delivery_manifest.json"
echo "  📈 Pipeline Summary: $OUTPUT_DIR/pipeline_summary.json"
echo ""
echo "🔍 To view results:"
echo "  open $OUTPUT_DIR/step4/corrected_best.png"
echo "  cat $OUTPUT_DIR/pipeline_summary.json | jq"
