#!/usr/bin/env python3
"""
Cache utilities for robust model loading and validation.

Prevents the classic Route A failures:
- Network calls when models should be cached
- Variant mismatches (fp16 not available locally)
- Missing model files causing pipeline crashes

Usage:
    from photostudio.utils.cache_utils import ensure_cached, from_pretrained_local
    
    # Verify cache before starting pipeline
    ensure_all_models_cached()
    
    # Load with fallbacks
    pipe = from_pretrained_local(
        StableDiffusionXLControlNetInpaintPipeline,
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=[cnet_canny, cnet_depth],
        torch_dtype=torch.float16,
    )
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelSpec:
    """Model specification for cache validation."""
    repo_id: str
    model_type: str  # "diffusion", "controlnet", "vae", etc.
    required: bool = True
    allow_patterns: Optional[List[str]] = None
    variant_fallback: bool = True


# Core models required for Route A
REQUIRED_MODELS = [
    ModelSpec(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        model_type="diffusion",
        required=True,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "model_index.json"]
    ),
    ModelSpec(
        repo_id="diffusers/controlnet-canny-sdxl-1.0",
        model_type="controlnet",
        required=True,
        allow_patterns=["*.safetensors", "*.json"]
    ),
    ModelSpec(
        repo_id="diffusers/controlnet-depth-sdxl-1.0", 
        model_type="controlnet",
        required=False,  # Optional - depth is memory intensive
        allow_patterns=["*.safetensors", "*.json"]
    ),
    ModelSpec(
        repo_id="madebyollin/sdxl-vae-fp16-fix",
        model_type="vae",
        required=False,  # Fallback VAE
        allow_patterns=["*.safetensors", "*.json"]
    )
]


def ping_network(timeout: float = 3.0) -> bool:
    """Quick network connectivity check."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", str(int(timeout * 1000)), "8.8.8.8"],
            capture_output=True,
            timeout=timeout + 1
        )
        return result.returncode == 0
    except Exception:
        return False


def get_cache_info() -> Dict[str, Any]:
    """Get current cache configuration and status."""
    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    diffusers_cache = os.getenv("DIFFUSERS_CACHE", os.path.join(hf_home, "diffusers"))
    
    return {
        "hf_home": hf_home,
        "diffusers_cache": diffusers_cache,
        "hf_offline": os.getenv("HF_HUB_OFFLINE", "0") == "1",
        "cache_exists": Path(hf_home).exists(),
        "network_available": ping_network(),
        "models_status": {}
    }


def ensure_cached(repo_id: str, allow_patterns: Optional[List[str]] = None) -> bool:
    """
    Verify a model is fully cached locally.
    
    Args:
        repo_id: HuggingFace model repository ID
        allow_patterns: Optional patterns to check (e.g. ["*.safetensors"])
        
    Returns:
        True if model is cached, False otherwise
        
    Raises:
        RuntimeError: If model is not cached and offline mode is enabled
    """
    try:
        from huggingface_hub import snapshot_download
        
        logger.debug(f"Checking cache for {repo_id}")
        
        # Try to access cached files only
        snapshot_download(
            repo_id=repo_id,
            local_files_only=True,
            allow_patterns=allow_patterns
        )
        
        logger.debug(f"✅ {repo_id} is cached")
        return True
        
    except Exception as e:
        logger.warning(f"❌ {repo_id} not fully cached: {e}")
        
        # If we're in offline mode, this is fatal
        if os.getenv("HF_HUB_OFFLINE", "0") == "1":
            raise RuntimeError(f"Model not cached and offline mode enabled: {repo_id}")
        
        return False


def ensure_all_models_cached(required_only: bool = False) -> Dict[str, bool]:
    """
    Check cache status for all required models.
    
    Args:
        required_only: Only check required models, skip optional ones
        
    Returns:
        Dict mapping repo_id to cache status
    """
    status = {}
    
    for model_spec in REQUIRED_MODELS:
        if required_only and not model_spec.required:
            continue
            
        try:
            cached = ensure_cached(model_spec.repo_id, model_spec.allow_patterns)
            status[model_spec.repo_id] = cached
        except Exception as e:
            logger.error(f"Cache check failed for {model_spec.repo_id}: {e}")
            status[model_spec.repo_id] = False
    
    return status


def from_pretrained_local(model_class: Type, repo_id: str, **kwargs) -> Any:
    """
    Load model with robust local fallbacks.
    
    Prevents network calls and handles variant mismatches gracefully.
    
    Args:
        model_class: Diffusers model class (e.g. StableDiffusionXLControlNetInpaintPipeline)
        repo_id: HuggingFace repository ID
        **kwargs: Additional arguments for from_pretrained
        
    Returns:
        Loaded model instance
    """
    # Force local-only loading
    kwargs.setdefault("local_files_only", True)
    
    # Extract variant for fallback handling
    variant = kwargs.pop("variant", None)
    
    try:
        # First try with specified variant
        if variant:
            logger.debug(f"Loading {repo_id} with variant={variant}")
            return model_class.from_pretrained(repo_id, variant=variant, **kwargs)
        else:
            logger.debug(f"Loading {repo_id} without variant")
            return model_class.from_pretrained(repo_id, **kwargs)
            
    except Exception as e:
        if variant:
            # Soft fallback: try without variant
            logger.warning(f"Variant {variant} not available for {repo_id}, falling back to default")
            try:
                return model_class.from_pretrained(repo_id, **kwargs)
            except Exception as e2:
                logger.error(f"Failed to load {repo_id} even without variant: {e2}")
                raise e2
        else:
            logger.error(f"Failed to load {repo_id}: {e}")
            raise e


def get_system_info() -> Dict[str, Any]:
    """Get system information for route selection."""
    import psutil
    import platform
    
    # Check for Apple Silicon
    is_apple_silicon = (
        platform.system() == "Darwin" and 
        (platform.machine() == "arm64" or "Apple" in platform.processor())
    )
    
    # Memory info
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    
    # GPU check
    has_cuda = False
    has_mps = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except ImportError:
        pass
    
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "is_apple_silicon": is_apple_silicon,
        "memory_gb": round(memory_gb, 1),
        "has_cuda": has_cuda,
        "has_mps": has_mps,
        "recommended_size": 1536 if memory_gb < 20 else 2048,
        "use_cpu_offload": memory_gb < 16
    }


def should_use_depth_controlnet() -> bool:
    """Determine if depth ControlNet should be used based on system resources."""
    # Check environment override
    use_depth_env = os.getenv("PHOTOSTUDIO_USE_DEPTH", "0")
    if use_depth_env == "1":
        return True
    elif use_depth_env == "0":
        return False
    
    # Auto-detect based on system
    system_info = get_system_info()
    
    # Enable depth if we have plenty of memory and it's cached
    if system_info["memory_gb"] >= 24:
        try:
            return ensure_cached("diffusers/controlnet-depth-sdxl-1.0")
        except Exception:
            return False
    
    return False


def print_cache_status():
    """Print detailed cache status for debugging."""
    cache_info = get_cache_info()
    system_info = get_system_info()
    
    print("🔍 Cache Status Report")
    print("=" * 50)
    print(f"HF_HOME: {cache_info['hf_home']}")
    print(f"Network: {'✅ Available' if cache_info['network_available'] else '❌ Offline'}")
    print(f"Offline Mode: {'✅ Enabled' if cache_info['hf_offline'] else '❌ Disabled'}")
    print(f"System: {system_info['platform']} {system_info['machine']} ({system_info['memory_gb']}GB RAM)")
    print(f"GPU: CUDA={system_info['has_cuda']} MPS={system_info['has_mps']}")
    print()
    
    model_status = ensure_all_models_cached()
    print("📦 Model Cache Status:")
    for repo_id, cached in model_status.items():
        status = "✅ Cached" if cached else "❌ Missing"
        model_spec = next((m for m in REQUIRED_MODELS if m.repo_id == repo_id), None)
        required = " (required)" if model_spec and model_spec.required else " (optional)"
        print(f"  {status}: {repo_id}{required}")
    
    print()
    print(f"💡 Recommended image size: {system_info['recommended_size']}px")
    print(f"💡 Depth ControlNet: {'✅ Enabled' if should_use_depth_controlnet() else '❌ Disabled'}")
    print("=" * 50)


if __name__ == "__main__":
    print_cache_status()
