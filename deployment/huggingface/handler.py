# handler.py — Hugging Face Inference Endpoints (Task: Custom)
import base64, io, os
from typing import Any, Dict, List, Optional

# Defensive imports with proper error handling
try:
    import torch
    from PIL import Image
    from diffusers import (
        StableDiffusionXLControlNetInpaintPipeline,
        ControlNetModel,
        AutoencoderKL,
        DPMSolverMultistepScheduler,
    )
except ImportError as e:
    print(f"CRITICAL: Failed to import required dependencies: {e}")
    raise RuntimeError(f"Import failed - check requirements.txt: {e}") from e

# ---------- utils ----------
def _b64_to_pil(b64: str) -> Image.Image:
    """Convert base64 string to PIL Image, handling data URLs."""
    raw = base64.b64decode(b64.split(",")[-1] if "," in b64 else b64)
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def _pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _read_env(key: str, default: Optional[str] = None) -> str:
    """Read environment variable with validation."""
    v = os.environ.get(key, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing required env var: {key}")
    return v

# ---------- one-time model load ----------
class _Global:
    pipe: Optional[StableDiffusionXLControlNetInpaintPipeline] = None
    device: Optional[torch.device] = None

def _load_pipeline() -> StableDiffusionXLControlNetInpaintPipeline:
    """Load the SDXL ControlNet inpainting pipeline with optimizations."""
    base_model = _read_env("BASE_MODEL")  # e.g. RunDiffusion/Juggernaut-XL-v9
    openpose_id = _read_env("OPENPOSE_ID")  # e.g. thibaud/controlnet-openpose-sdxl-1.0
    canny_id    = _read_env("CANNY_ID")     # e.g. diffusers/controlnet-canny-sdxl-1.0
    vae_id      = os.environ.get("VAE_ID", "madebyollin/sdxl-vae-fp16-fix")

    # device & dtype selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        # half precision is not safe on CPU/MPS for SDXL
        dtype = torch.float32

    # Load ControlNet models
    cnets = [
        ControlNetModel.from_pretrained(openpose_id, torch_dtype=dtype),
        ControlNetModel.from_pretrained(canny_id,    torch_dtype=dtype),
    ]
    
    # Load optimized VAE
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)

    # Load main pipeline
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base_model,
        controlnet=cnets,
        vae=vae,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )

    # Use optimized scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # Apply memory/performance optimizations (safe on CUDA; guarded on others)
    try:
        if device.type == "cuda":
            pipe.enable_model_cpu_offload()   # keeps VRAM lower on A10G
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
    except Exception:
        pass

    _Global.device = device
    return pipe

# The HF runtime instantiates this class once and calls __call__() per request.
class EndpointHandler:
    def __init__(self, model_dir: str = None):
        """Initialize the endpoint handler by loading the pipeline once.
        
        Args:
            model_dir: Model directory path (required by HF toolkit, but we use env vars instead)
        """
        # HF toolkit passes model_dir but we use environment variables for configuration
        self.model_dir = model_dir
        
        # Avoid importing anything heavy above; load here once.
        if _Global.pipe is None:
            _Global.pipe = _load_pipeline()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Step 3 rendering request.
        
        Supports both formats:
        1. Direct format: {"prompt": "...", "control_images": [...], ...}
        2. HF standard format: {"inputs": {"prompt": "...", "control_images": [...], ...}}
        
        Returns:
        {
          "image_base64": "...",                  # main output for Step 4 QA
          "meta": {
            "seed": 12345,
            "steps": 25,
            "guidance": 5.0,
            "size": [1024, 1024],
            "route": "A",
            "pipeline": "SDXL-ControlNet-Inpaint"
          }
        }
        """
        if _Global.pipe is None:
            return {"error": "Pipeline not initialized"}

        # Handle both direct format and HF standard "inputs" wrapper format
        if "inputs" in data:
            # HF standard format: {"inputs": {...}}
            payload = data["inputs"]
            if isinstance(payload, str):
                # If inputs is a string, treat as simple prompt
                payload = {"prompt": payload}
        else:
            # Direct format: {"prompt": ..., "control_images": ...}
            payload = data

        # Extract and validate required parameters
        prompt = payload.get("prompt", "")
        if not prompt:
            return {"error": "prompt is required (should come from Step 2 weaver)"}

        # Optional parameters with defaults
        neg = payload.get("negative_prompt") or None
        steps = int(payload.get("num_inference_steps", 25))
        guidance = float(payload.get("guidance_scale", 5.0))
        width = int(payload.get("width", 1024))
        height = int(payload.get("height", 1024))
        seed = int(payload.get("seed", 12345))

        # Control images from Step 0 isolation
        ctrl_b64s: List[str] = payload.get("control_images") or []
        if len(ctrl_b64s) < 2:
            return {"error": "control_images must include at least [openpose, canny] base64 strings from Step 0"}
        
        try:
            control_images = [_b64_to_pil(b) for b in ctrl_b64s]
        except Exception as e:
            return {"error": f"Failed to decode control images: {str(e)}"}

        # Optional inpaint mask from Step 0
        mask_b64: Optional[str] = payload.get("mask_image")
        mask_img = None
        if mask_b64:
            try:
                mask_img = _b64_to_pil(mask_b64)
            except Exception as e:
                return {"error": f"Failed to decode mask image: {str(e)}"}
                
        # Optional source image for inpainting
        source_b64: Optional[str] = payload.get("image")
        source_img = None
        if source_b64:
            try:
                source_img = _b64_to_pil(source_b64)
            except Exception as e:
                return {"error": f"Failed to decode source image: {str(e)}"}

        # Setup deterministic generation
        gen = torch.Generator(device=_Global.device).manual_seed(seed)

        # For multiple ControlNets, we need to handle image parameters correctly
        num_controlnets = len(control_images)
        
        # Determine the source image for inpainting
        inpaint_source = source_img or mask_img  # Use source_img if provided, fallback to mask_img
        
        # For multiple ControlNets, image parameter must be a list (same image for each ControlNet)
        if num_controlnets > 1 and inpaint_source is not None:
            image_param = [inpaint_source] * num_controlnets
        else:
            image_param = inpaint_source

        try:
            with torch.inference_mode():
                out = _Global.pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    controlnet_conditioning_image=control_images,
                    image=image_param,  # Properly formatted for single/multiple ControlNets
                    mask_image=mask_img if mask_img and source_img else None,  # Only use mask if we have source
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=gen,
                )

            img = out.images[0]
            return {
                "image_base64": _pil_to_b64(img, fmt="PNG"),
                "meta": {
                    "seed": seed,
                    "steps": steps,
                    "guidance": guidance,
                    "size": [width, height],
                    "route": "A",
                    "pipeline": "SDXL-ControlNet-Inpaint",
                },
            }
            
        except torch.cuda.OutOfMemoryError:
            return {
                "error": "GPU out of memory. Try reducing width/height or num_inference_steps",
                "suggestion": "Consider using Route B (Gemini) fallback for this request"
            }
        except Exception as e:
            return {"error": f"Pipeline execution failed: {str(e)}"}
