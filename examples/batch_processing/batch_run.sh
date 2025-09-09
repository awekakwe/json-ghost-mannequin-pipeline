#!/bin/bash
# Batch Processing Example
# Process multiple garment images in parallel

set -e

echo "🚀 Batch Processing Example"
echo "============================"

# Configuration
INPUT_DIR="./inputs"
OUTPUT_BASE="./batch_output"
MAX_PARALLEL=3  # Number of parallel processes
ROUTE="A"       # Default route

# Check if input directory exists and has images
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory '$INPUT_DIR' not found!"
    echo "Please create '$INPUT_DIR' and add garment images."
    exit 1
fi

# Count image files
image_count=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)

if [ "$image_count" -eq 0 ]; then
    echo "❌ Error: No image files found in '$INPUT_DIR'!"
    echo "Supported formats: JPG, JPEG, PNG"
    exit 1
fi

echo "📸 Found $image_count images to process"
echo "📂 Input: $INPUT_DIR"
echo "📂 Output: $OUTPUT_BASE"
echo "🔄 Max parallel: $MAX_PARALLEL"
echo ""

# Clean previous output
rm -rf "$OUTPUT_BASE"
mkdir -p "$OUTPUT_BASE"

# Function to process a single image
process_image() {
    local input_file="$1"
    local filename=$(basename "$input_file")
    local name_without_ext="${filename%.*}"
    
    # Generate SKU and variant from filename
    local sku="BATCH_${name_without_ext^^}"  # Uppercase
    local variant="default"
    
    # Try to extract variant from filename if it contains underscore
    if [[ "$name_without_ext" == *"_"* ]]; then
        sku=$(echo "$name_without_ext" | cut -d'_' -f1 | tr '[:lower:]' '[:upper:]')
        variant=$(echo "$name_without_ext" | cut -d'_' -f2- | tr '[:upper:]' '[:lower:]')
    fi
    
    local output_dir="$OUTPUT_BASE/${name_without_ext}"
    
    echo "🔄 Processing: $filename -> $sku-$variant"
    
    # Run pipeline for this image
    if photostudio-pipeline \
        --input "$input_file" \
        --sku "$sku" \
        --variant "$variant" \
        --out "$output_dir" \
        --route "$ROUTE" \
        --template-id ghost_mannequin_v1 \
        > "$output_dir.log" 2>&1; then
        
        echo "✅ Completed: $filename"
        
        # Extract key metrics for summary
        local processing_time=""
        local health_score=""
        local status="completed"
        
        if [ -f "$output_dir/pipeline_summary.json" ] && command -v jq >/dev/null 2>&1; then
            processing_time=$(cat "$output_dir/pipeline_summary.json" | jq -r '.processing_time_seconds // "N/A"')
            status=$(cat "$output_dir/pipeline_summary.json" | jq -r '.status // "unknown"')
        fi
        
        if [ -f "$output_dir/step6/summary.json" ] && command -v jq >/dev/null 2>&1; then
            health_score=$(cat "$output_dir/step6/summary.json" | jq -r '.health_metrics.health_score // "N/A"')
        fi
        
        echo "$filename,$sku,$variant,$status,$processing_time,$health_score" >> "$OUTPUT_BASE/batch_summary.csv"
    else
        echo "❌ Failed: $filename"
        echo "$filename,$sku,$variant,failed,N/A,N/A" >> "$OUTPUT_BASE/batch_summary.csv"
    fi
}

# Export function for parallel execution
export -f process_image
export OUTPUT_BASE ROUTE

# Create CSV header
echo "filename,sku,variant,status,processing_time_seconds,health_score" > "$OUTPUT_BASE/batch_summary.csv"

# Process images in parallel
echo "🔄 Starting batch processing..."
find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | \
    head -20 | \
    xargs -n 1 -P "$MAX_PARALLEL" -I {} bash -c 'process_image "$@"' _ {}

echo ""
echo "✅ Batch processing complete!"
echo ""

# Generate summary report
echo "📊 Batch Processing Summary"
echo "==========================="

if [ -f "$OUTPUT_BASE/batch_summary.csv" ]; then
    total_processed=$(tail -n +2 "$OUTPUT_BASE/batch_summary.csv" | wc -l)
    successful=$(grep ",completed," "$OUTPUT_BASE/batch_summary.csv" | wc -l)
    failed=$(grep ",failed," "$OUTPUT_BASE/batch_summary.csv" | wc -l)
    
    echo "📈 Results:"
    echo "  📸 Total Images: $total_processed"
    echo "  ✅ Successful: $successful"
    echo "  ❌ Failed: $failed"
    echo "  📊 Success Rate: $(echo "scale=1; $successful * 100 / $total_processed" | bc -l)%"
    
    # Average processing time (if jq is available)
    if command -v jq >/dev/null 2>&1 && [ "$successful" -gt 0 ]; then
        # Extract processing times and calculate average
        avg_time=$(awk -F, 'NR>1 && $5!="N/A" {sum+=$5; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}' "$OUTPUT_BASE/batch_summary.csv")
        echo "  ⏱️  Avg Processing Time: ${avg_time}s"
    fi
    
    echo ""
    echo "📁 Individual Results:"
    while IFS=, read -r filename sku variant status proc_time health; do
        if [ "$filename" != "filename" ]; then  # Skip header
            if [ "$status" = "completed" ]; then
                echo "  ✅ $filename ($sku-$variant) - ${proc_time}s"
            else
                echo "  ❌ $filename ($sku-$variant) - FAILED"
            fi
        fi
    done < "$OUTPUT_BASE/batch_summary.csv"
fi

echo ""
echo "📋 Output Structure:"
echo "  📁 $OUTPUT_BASE/"
for dir in "$OUTPUT_BASE"/*/; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        echo "    📁 $dirname/"
        echo "      📸 corrected_best.png"
        echo "      📊 qa_report.json"
        echo "      📦 delivery_manifest.json"
    fi
done

echo ""
echo "📄 Detailed Results:"
echo "  📊 Batch Summary: $OUTPUT_BASE/batch_summary.csv"
echo "  📋 Individual Logs: $OUTPUT_BASE/*.log"
echo ""
echo "🔍 View Results:"
if command -v jq >/dev/null 2>&1; then
    echo "  cat $OUTPUT_BASE/batch_summary.csv | column -t -s,"
else
    echo "  cat $OUTPUT_BASE/batch_summary.csv"
fi

# Check for any failed processing
if [ "$failed" -gt 0 ]; then
    echo ""
    echo "⚠️  Some images failed processing. Check individual log files:"
    find "$OUTPUT_BASE" -name "*.log" -exec echo "  {}" \;
