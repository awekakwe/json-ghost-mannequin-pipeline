#!/bin/bash
# setup_environment.sh - Production Environment Setup for JSON Ghost-Mannequin Pipeline
# Implements all hardening recommendations from the reality check

echo "🚀 Setting up JSON Ghost-Mannequin Pipeline Environment"
echo "=================================================="

# Fix the OpenMP warning (the safe way)
echo "🔧 Checking OpenMP libraries..."
if python -c "import sklearn" 2>&1 | grep -q "libiomp.*libomp"; then
    echo "⚠️  Detected OpenMP conflict. Consider running:"
    echo "   pip uninstall intel-openmp"
    echo "   or set: export KMP_DUPLICATE_LIB_OK=TRUE (not recommended for production)"
else
    echo "✅ OpenMP libraries OK"
fi

# Set environment variables for reliable caching
echo "🗂️  Configuring HuggingFace cache..."
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export DIFFUSERS_CACHE=${DIFFUSERS_CACHE:-$HF_HOME/diffusers}
export HF_HUB_DISABLE_TELEMETRY=1

# Create cache directories
mkdir -p "$HF_HOME"
mkdir -p "$DIFFUSERS_CACHE"

echo "Cache paths:"
echo "  HF_HOME: $HF_HOME"
echo "  DIFFUSERS_CACHE: $DIFFUSERS_CACHE"

# Check system resources
echo "🖥️  System Information:"
python -c "
import psutil
import platform
import torch

print(f'  Platform: {platform.system()} {platform.machine()}')
print(f'  RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB')
print(f'  CUDA Available: {torch.cuda.is_available() if hasattr(torch, \"cuda\") else \"N/A\"}')
print(f'  MPS Available: {hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available() if hasattr(torch, \"backends\") else \"N/A\"}')
"

# Check model cache status
echo "📦 Checking model cache..."
python -c "
import sys
sys.path.insert(0, 'src')
from photostudio.utils.cache_utils import print_cache_status
print_cache_status()
"

# Environment recommendations
echo ""
echo "🎯 Production Environment Recommendations:"
echo "=================================================="
echo "1. For offline operation:"
echo "   export HF_HUB_OFFLINE=1"
echo ""
echo "2. For Apple Silicon optimization:"
echo "   export PHOTOSTUDIO_USE_DEPTH=0  # Disable depth ControlNet"
echo ""
echo "3. For memory-constrained systems:"
echo "   # Pipeline will auto-detect and use 1536px instead of 2048px"
echo ""
echo "4. Example production command:"
echo "   photostudio-pipeline --input image.jpg --sku ABC123 --variant red --route auto --local-only"
echo ""
echo "✅ Environment setup complete!"
