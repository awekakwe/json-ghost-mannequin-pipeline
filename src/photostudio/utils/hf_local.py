#!/usr/bin/env python3
"""
hf_local.py - Bulletproof HuggingFace local loader with variant fallback

Prevents the classic Route A failures:
- Tries variant="fp16" → None → "fp32" automatically
- Never hits the network when local_files_only=True
- Safe dtype resolution for CUDA/MPS/CPU devices
- Graceful degradation when variants unavailable

Usage:
    from photostudio.utils.hf_local import resolve_torch_dtype, safe_from_pretrained
    
    dtype = resolve_torch_dtype("mps")  # Returns fp32 for MPS/CPU, fp16 for CUDA
    
    model = safe_from_pretrained(
        ControlNetModel,
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=dtype,
        local_files_only=True
    )
"""

from __future__ import annotations
import logging
import os
from typing import Optional, Sequence, Type, Any

logger = logging.getLogger(__name__)

def resolve_torch_dtype(device: str, prefer_half: bool = True) -> 'torch.dtype':
    """
    Resolve the appropriate torch dtype based on device capabilities.
    
    CUDA: fp16 by default (if prefer_half=True)
    MPS/CPU: fp32 (fp16 on MPS is flaky and not well supported)
    
    Args:
        device: Target device ("cuda", "mps", "cpu")
        prefer_half: Whether to prefer fp16 on CUDA devices
        
    Returns:
        torch.dtype: Appropriate dtype for the device
    """
    try:
        import torch
        
        if device == "cuda" and prefer_half:
            return torch.float16
        else:
            return torch.float32  # Safe on MPS/CPU
    except ImportError:
        # Fallback if torch not available
        logger.warning("PyTorch not available, cannot resolve dtype")
        return None


def safe_from_pretrained(
    cls: Type[Any],
    repo_id: str,
    *,
    subfolder: Optional[str] = None,
    variant_candidates: Sequence[Optional[str]] = ("fp16", None, "fp32"),
    torch_dtype: Optional['torch.dtype'] = None,
    local_files_only: bool = True,
    use_safetensors: bool = True,
    low_cpu_mem_usage: bool = True,
    **kwargs,
) -> Any:
    """
    Bulletproof local loader for diffusers models.
    
    Tries multiple 'variant' values in sequence until one succeeds.
    Never goes online (unless you explicitly set local_files_only=False).
    
    Args:
        cls: Diffusers model class (e.g., ControlNetModel, AutoencoderKL)
        repo_id: HuggingFace repository ID
        subfolder: Optional subfolder within the repo
        variant_candidates: Sequence of variants to try (e.g., "fp16", None, "fp32")
        torch_dtype: PyTorch data type
        local_files_only: Force local-only loading (default: True)
        use_safetensors: Prefer safetensors format (default: True)
        low_cpu_mem_usage: Enable low CPU memory usage (default: True)
        **kwargs: Additional arguments for from_pretrained
        
    Returns:
        Loaded model instance
        
    Raises:
        RuntimeError: If no variant could be loaded from local cache
    """
    if torch_dtype is None:
        # Default to fp32 if not specified
        try:
            import torch
            torch_dtype = torch.float32
        except ImportError:
            pass
    
    last_err = None
    
    for variant in variant_candidates:
        try:
            logger.debug(f"Attempting to load {repo_id} with variant={variant}")
            
            obj = cls.from_pretrained(
                repo_id,
                subfolder=subfolder,
                variant=variant,
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
                use_safetensors=use_safetensors,
                low_cpu_mem_usage=low_cpu_mem_usage,
                **kwargs,
            )
            
            logger.info(f"✅ Loaded {repo_id} (variant={variant}) from local cache")
            return obj
            
        except Exception as e:
            last_err = e
            logger.debug(f"Variant {variant} failed for {repo_id}: {e}")
            continue
    
    # If we get here, all variants failed
    error_msg = f"Could not load {repo_id} from local cache with any variant. Last error: {last_err}"
    logger.error(f"❌ {error_msg}")
    raise RuntimeError(error_msg)


def suppress_model_warnings():
    """Suppress common model loading warnings for cleaner logs."""
    import warnings
    import logging
    import os
    
    # Suppress TIMM/SAM registry warnings
    warnings.filterwarnings("ignore", "Importing from timm.models")
    warnings.filterwarnings("ignore", "Overwriting.*in registry")
    
    # Set environment variable to suppress warnings
    os.environ.setdefault("PYTHONWARNINGS", "ignore:::timm")
    
    # Reduce logging levels for noisy libraries
    logging.getLogger("controlnet_aux.segment_anything").setLevel(logging.ERROR)
    logging.getLogger("timm").setLevel(logging.WARNING)
    
    logger.debug("Model loading warnings suppressed")


# Convenience function for common SDXL models
def get_device_optimal_dtype(device: str = None) -> 'torch.dtype':
    """
    Get the optimal dtype for the current or specified device.
    
    Args:
        device: Target device. If None, auto-detect.
        
    Returns:
        Optimal torch.dtype for the device
    """
    if device is None:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"
    
    return resolve_torch_dtype(device, prefer_half=(device == "cuda"))


if __name__ == "__main__":
    # Simple test
    print("🔧 HuggingFace Local Loader Test")
    print(f"Optimal dtype for auto-detected device: {get_device_optimal_dtype()}")
    
    # Test dtype resolution
    for device in ["cuda", "mps", "cpu"]:
        dtype = resolve_torch_dtype(device)
        print(f"Device {device}: {dtype}")
