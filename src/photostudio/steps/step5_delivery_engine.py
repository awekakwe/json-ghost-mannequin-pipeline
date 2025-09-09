#!/usr/bin/env python3
"""
step5_delivery_engine.py – Step 5/6 Package & Publish
======================================================

Produce master + web variants, embed metadata, upload to CDN, emit manifest.

Features:
- Content-addressable naming (SHA256), semantic versioning
- PNG master; WebP (2048/1024/512), optional AVIF, thumb (384)
- sRGB ICC embedding, immutable cache headers
- S3 upload (optional), delivery manifest with checksums & metadata

Outputs:
/step5/
  master/ SKU_variant_vX_<hash>.png
  web/    *.webp, *.avif
  thumbs/ *.webp
  manifests/delivery_manifest.json

Dependencies: Pillow boto3 google-cloud-storage
"""

from __future__ import annotations

import os
import json
import hashlib
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from urllib.parse import urljoin

from PIL import Image, ImageCms
from PIL.ExifTags import TAGS

# Optional cloud storage dependencies
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    HAVE_S3 = True
except ImportError:
    HAVE_S3 = False
    boto3 = None

try:
    from google.cloud import storage as gcs
    HAVE_GCS = True
except ImportError:
    HAVE_GCS = False
    gcs = None

logger = logging.getLogger("photostudio.delivery_engine")

# ---------------------------------------------------------------------------
# Configuration & Data Structures
# ---------------------------------------------------------------------------

@dataclass
class DeliveryConfig:
    """Configuration for the delivery engine."""
    master_format: str = "PNG"
    web_formats: List[str] = None  # ["WebP", "AVIF"]
    web_sizes: List[int] = None    # [2048, 1024, 512]
    thumb_size: int = 384
    quality: int = 90
    enable_avif: bool = False
    enable_progressive: bool = True
    embed_icc: bool = True
    
    def __post_init__(self):
        if self.web_formats is None:
            self.web_formats = ["WebP"]
            if self.enable_avif:
                self.web_formats.append("AVIF")
        if self.web_sizes is None:
            self.web_sizes = [2048, 1024, 512]

@dataclass 
class CloudConfig:
    """Configuration for cloud storage."""
    provider: str = "s3"  # "s3" or "gcs"
    bucket: str = ""
    prefix: str = ""
    public_url_base: str = ""
    # S3 specific
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    # GCS specific
    gcs_credentials_path: Optional[str] = None

@dataclass
class AssetInfo:
    """Information about a generated asset."""
    path: str
    filename: str
    format: str
    size_bytes: int
    dimensions: tuple  # (width, height)
    checksum: str
    url: Optional[str] = None

@dataclass
class DeliveryManifest:
    """Complete delivery manifest for an asset."""
    sku: str
    variant: str
    version: str
    assets: Dict[str, AssetInfo]
    source_info: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    created_at: str

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def compute_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute hash of file contents."""
    hash_algo = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_algo.update(chunk)
            
    return hash_algo.hexdigest()

def get_image_info(image_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract image information."""
    with Image.open(image_path) as img:
        return {
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info,
            "icc_profile": img.info.get("icc_profile") is not None
        }

def ensure_srgb_profile(image: Image.Image) -> Image.Image:
    """Ensure image has sRGB ICC profile."""
    try:
        if not hasattr(image, 'info') or 'icc_profile' not in image.info:
            # Create sRGB profile
            srgb_profile = ImageCms.createProfile('sRGB')
            # Try modern method first, fall back to older method
            try:
                icc_profile = srgb_profile.tobytes()
            except AttributeError:
                # Older PIL versions might not have tobytes()
                icc_profile = srgb_profile.save()
            
            # Apply to image
            image.info['icc_profile'] = icc_profile
    except Exception as e:
        # If ICC profile handling fails, just continue without it
        logger.warning(f"ICC profile handling failed: {e}")
        
    return image

def generate_content_addressable_name(
    content: bytes, 
    sku: str, 
    variant: str, 
    version: str, 
    extension: str
) -> str:
    """Generate content-addressable filename."""
    content_hash = hashlib.sha256(content).hexdigest()[:16]
    return f"{sku}_{variant}_{version}_{content_hash}.{extension.lower()}"

# ---------------------------------------------------------------------------
# Image Processing Pipeline
# ---------------------------------------------------------------------------

class ImageProcessor:
    """Process images for web delivery."""
    
    def __init__(self, config: DeliveryConfig):
        self.config = config
        
    def process_master(self, input_path: Path, sku: str, variant: str, version: str) -> AssetInfo:
        """Process master PNG file."""
        with Image.open(input_path) as img:
            # Ensure RGB mode for master
            if img.mode == "RGBA":
                # Create white background for transparency
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
                
            # Ensure sRGB profile if ICC embedding is enabled
            if self.config.embed_icc:
                img = ensure_srgb_profile(img)
            
            # Generate content-addressable name
            img_bytes = self._image_to_bytes(img, "PNG")
            filename = generate_content_addressable_name(
                img_bytes, sku, variant, version, "png"
            )
            
            return AssetInfo(
                path=filename,
                filename=filename,
                format="PNG", 
                size_bytes=len(img_bytes),
                dimensions=img.size,
                checksum=hashlib.sha256(img_bytes).hexdigest()
            )
    
    def process_web_variants(
        self, 
        input_path: Path, 
        sku: str, 
        variant: str, 
        version: str
    ) -> Dict[str, AssetInfo]:
        """Process web format variants at multiple sizes."""
        variants = {}
        
        with Image.open(input_path) as img:
            # Convert to RGB for web formats
            if img.mode != "RGB":
                if img.mode == "RGBA":
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert("RGB")
            
            # Generate variants for each size and format
            for size in self.config.web_sizes:
                for fmt in self.config.web_formats:
                    resized = self._resize_image(img, size)
                    
                    # Apply ICC profile for web formats
                    if self.config.embed_icc and fmt in ["WebP"]:
                        resized = ensure_srgb_profile(resized)
                    
                    img_bytes = self._image_to_bytes(resized, fmt)
                    filename = f"{sku}_{variant}_{size}.{fmt.lower()}"
                    
                    variant_key = f"{fmt.lower()}_{size}"
                    variants[variant_key] = AssetInfo(
                        path=filename,
                        filename=filename,
                        format=fmt,
                        size_bytes=len(img_bytes),
                        dimensions=resized.size,
                        checksum=hashlib.sha256(img_bytes).hexdigest()
                    )
                    
        return variants
    
    def process_thumbnail(
        self, 
        input_path: Path, 
        sku: str, 
        variant: str
    ) -> AssetInfo:
        """Process thumbnail image."""
        with Image.open(input_path) as img:
            # Convert to RGB
            if img.mode != "RGB":
                if img.mode == "RGBA":
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert("RGB")
            
            # Resize to thumbnail size
            thumb = self._resize_image(img, self.config.thumb_size)
            
            img_bytes = self._image_to_bytes(thumb, "WebP")
            filename = f"{sku}_{variant}_{self.config.thumb_size}.webp"
            
            return AssetInfo(
                path=filename,
                filename=filename,
                format="WebP",
                size_bytes=len(img_bytes),
                dimensions=thumb.size,
                checksum=hashlib.sha256(img_bytes).hexdigest()
            )
    
    def _resize_image(self, image: Image.Image, target_size: int) -> Image.Image:
        """Resize image maintaining aspect ratio."""
        # For square images, just resize directly
        if image.size[0] == image.size[1]:
            return image.resize((target_size, target_size), Image.LANCZOS)
        
        # For non-square, pad to square first, then resize
        max_dim = max(image.size)
        square = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        offset = ((max_dim - image.size[0]) // 2, (max_dim - image.size[1]) // 2)
        square.paste(image, offset)
        
        return square.resize((target_size, target_size), Image.LANCZOS)
    
    def _image_to_bytes(self, image: Image.Image, format: str) -> bytes:
        """Convert image to bytes in specified format."""
        import io
        
        buffer = io.BytesIO()
        
        save_kwargs = {"format": format, "optimize": True}
        
        if format == "WebP":
            save_kwargs["quality"] = self.config.quality
            save_kwargs["method"] = 6  # Best compression
            if self.config.enable_progressive:
                save_kwargs["progressive"] = True
        elif format == "AVIF":
            save_kwargs["quality"] = self.config.quality
        elif format == "PNG":
            save_kwargs["optimize"] = True
            
        image.save(buffer, **save_kwargs)
        return buffer.getvalue()

# ---------------------------------------------------------------------------
# Cloud Storage Providers
# ---------------------------------------------------------------------------

class CloudStorageProvider:
    """Base class for cloud storage providers."""
    
    def upload_file(self, local_path: Path, remote_path: str) -> str:
        """Upload file and return public URL."""
        raise NotImplementedError
    
    def set_cache_headers(self, remote_path: str, max_age: int = 31536000):
        """Set cache headers for immutable content."""
        raise NotImplementedError

class S3Provider(CloudStorageProvider):
    """AWS S3 storage provider."""
    
    def __init__(self, config: CloudConfig):
        if not HAVE_S3:
            raise ImportError("boto3 not installed")
            
        self.config = config
        self.client = boto3.client(
            's3',
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            region_name=config.aws_region
        )
        
    def upload_file(self, local_path: Path, remote_path: str) -> str:
        """Upload file to S3."""
        full_key = f"{self.config.prefix.rstrip('/')}/{remote_path}"
        
        # Determine content type
        content_type = self._get_content_type(local_path.suffix)
        
        try:
            self.client.upload_file(
                str(local_path),
                self.config.bucket,
                full_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'CacheControl': 'public, max-age=31536000, immutable',
                    'ACL': 'public-read'
                }
            )
            
            return urljoin(self.config.public_url_base, full_key)
            
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    def _get_content_type(self, extension: str) -> str:
        """Get MIME type for file extension."""
        ext_map = {
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.avif': 'image/avif',
            '.json': 'application/json'
        }
        return ext_map.get(extension.lower(), 'application/octet-stream')

class GCSProvider(CloudStorageProvider):
    """Google Cloud Storage provider."""
    
    def __init__(self, config: CloudConfig):
        if not HAVE_GCS:
            raise ImportError("google-cloud-storage not installed")
            
        self.config = config
        if config.gcs_credentials_path:
            self.client = gcs.Client.from_service_account_json(config.gcs_credentials_path)
        else:
            self.client = gcs.Client()
        
        self.bucket = self.client.bucket(config.bucket)
        
    def upload_file(self, local_path: Path, remote_path: str) -> str:
        """Upload file to GCS."""
        full_path = f"{self.config.prefix.rstrip('/')}/{remote_path}"
        blob = self.bucket.blob(full_path)
        
        # Set content type
        blob.content_type = self._get_content_type(local_path.suffix)
        
        # Set cache headers
        blob.cache_control = 'public, max-age=31536000, immutable'
        
        blob.upload_from_filename(str(local_path))
        blob.make_public()
        
        return urljoin(self.config.public_url_base, full_path)
    
    def _get_content_type(self, extension: str) -> str:
        """Get MIME type for file extension.""" 
        ext_map = {
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.avif': 'image/avif',
            '.json': 'application/json'
        }
        return ext_map.get(extension.lower(), 'application/octet-stream')

def create_cloud_provider(config: CloudConfig) -> Optional[CloudStorageProvider]:
    """Factory function to create cloud storage provider."""
    if not config.bucket:
        return None
        
    if config.provider == "s3":
        return S3Provider(config)
    elif config.provider == "gcs":
        return GCSProvider(config)
    else:
        raise ValueError(f"Unsupported cloud provider: {config.provider}")

# ---------------------------------------------------------------------------
# Main Delivery Engine
# ---------------------------------------------------------------------------

class DeliveryEngine:
    """Main delivery engine for packaging and publishing assets."""
    
    def __init__(
        self, 
        delivery_config: DeliveryConfig,
        cloud_config: Optional[CloudConfig] = None
    ):
        self.delivery_config = delivery_config
        self.cloud_config = cloud_config
        self.processor = ImageProcessor(delivery_config)
        self.cloud_provider = create_cloud_provider(cloud_config) if cloud_config else None
        
    def process_and_deliver(
        self,
        final_image_path: Path,
        qa_report_path: Path,
        analysis_json_path: Path,
        sku: str,
        variant: str,
        version: str = "v1",
        output_dir: Path = Path("step5")
    ) -> DeliveryManifest:
        """Complete processing and delivery pipeline."""
        
        # Create output directory structure
        output_dir.mkdir(parents=True, exist_ok=True)
        master_dir = output_dir / "master"
        web_dir = output_dir / "web"
        thumbs_dir = output_dir / "thumbs"
        manifests_dir = output_dir / "manifests"
        
        for dir_path in [master_dir, web_dir, thumbs_dir, manifests_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"Processing delivery for {sku}-{variant} {version}")
        
        # Process master image
        logger.info("Processing master PNG...")
        master_info = self.processor.process_master(final_image_path, sku, variant, version)
        master_path = master_dir / master_info.filename
        self._save_processed_image(final_image_path, master_path, "PNG")
        master_info.path = str(master_path)
        
        # Process web variants
        logger.info("Processing web variants...")
        web_variants = self.processor.process_web_variants(final_image_path, sku, variant, version)
        for variant_key, variant_info in web_variants.items():
            web_path = web_dir / variant_info.filename
            self._save_web_variant(final_image_path, web_path, variant_info.format, variant_info.dimensions)
            variant_info.path = str(web_path)
        
        # Process thumbnail
        logger.info("Processing thumbnail...")
        thumb_info = self.processor.process_thumbnail(final_image_path, sku, variant)
        thumb_path = thumbs_dir / thumb_info.filename
        self._save_web_variant(final_image_path, thumb_path, "WebP", (thumb_info.dimensions[0], thumb_info.dimensions[1]))
        thumb_info.path = str(thumb_path)
        
        # Combine all assets
        all_assets = {
            "master": master_info,
            "thumb": thumb_info,
            **web_variants
        }
        
        # Upload to cloud storage if configured
        if self.cloud_provider:
            logger.info("Uploading to cloud storage...")
            for asset_key, asset_info in all_assets.items():
                try:
                    remote_path = f"{sku}/{variant}/{Path(asset_info.path).name}"
                    asset_info.url = self.cloud_provider.upload_file(Path(asset_info.path), remote_path)
                    logger.info(f"Uploaded {asset_key}: {asset_info.url}")
                except Exception as e:
                    logger.error(f"Failed to upload {asset_key}: {e}")
        
        # Load source information
        source_info = self._load_source_info(qa_report_path, analysis_json_path)
        
        # Create delivery manifest
        manifest = DeliveryManifest(
            sku=sku,
            variant=variant,
            version=version,
            assets=all_assets,
            source_info=source_info,
            processing_metadata={
                "pipeline_version": "1.1.0",
                "delivery_engine_version": "1.0.0",
                "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "total_assets": len(all_assets),
                "cloud_provider": self.cloud_config.provider if self.cloud_config else None
            },
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S")
        )
        
        # Save manifest
        manifest_path = manifests_dir / "delivery_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            # Convert dataclass to dict for JSON serialization
            manifest_dict = asdict(manifest)
            json.dump(manifest_dict, f, indent=2)
        
        logger.info(f"Delivery complete! Manifest saved to: {manifest_path}")
        
        return manifest
    
    def _save_processed_image(self, input_path: Path, output_path: Path, format: str):
        """Save processed image in specified format."""
        with Image.open(input_path) as img:
            # Ensure RGB for non-PNG formats
            if format != "PNG" and img.mode != "RGB":
                if img.mode == "RGBA":
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert("RGB")
                    
            # Apply ICC profile if enabled
            if self.delivery_config.embed_icc:
                img = ensure_srgb_profile(img)
            
            save_kwargs = {"format": format, "optimize": True}
            if format == "PNG":
                save_kwargs["optimize"] = True
            
            img.save(output_path, **save_kwargs)
    
    def _save_web_variant(self, input_path: Path, output_path: Path, format: str, dimensions: tuple):
        """Save web variant at specific dimensions."""
        with Image.open(input_path) as img:
            # Convert to RGB
            if img.mode != "RGB":
                if img.mode == "RGBA":
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert("RGB")
            
            # Resize if needed
            if img.size != dimensions:
                img = self.processor._resize_image(img, max(dimensions))
            
            # Apply ICC profile for supported formats
            if self.delivery_config.embed_icc and format == "WebP":
                img = ensure_srgb_profile(img)
            
            save_kwargs = {"format": format, "optimize": True}
            if format == "WebP":
                save_kwargs["quality"] = self.delivery_config.quality
                save_kwargs["method"] = 6
            elif format == "AVIF":
                save_kwargs["quality"] = self.delivery_config.quality
                
            img.save(output_path, **save_kwargs)
    
    def _load_source_info(self, qa_report_path: Path, analysis_json_path: Path) -> Dict[str, Any]:
        """Load source information from QA report and analysis JSON."""
        source_info = {}
        
        # Load QA report
        if qa_report_path.exists():
            with open(qa_report_path, 'r', encoding='utf-8') as f:
                source_info["qa_report"] = json.load(f)
        
        # Load analysis JSON  
        if analysis_json_path.exists():
            with open(analysis_json_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
                source_info["analysis"] = {
                    "primary_color": analysis.get("color_extraction", {}).get("primary_color", {}),
                    "category": analysis.get("garment_categorization", {}).get("primary_classification", {}),
                    "quality_score": analysis.get("input_validation", {}).get("image_quality_score", 0.0),
                    "session_id": analysis.get("processing_metadata", {}).get("session_id", "unknown")
                }
        
        return source_info

# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 5/6: Package & Publish - Generate delivery assets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--final", required=True, help="Path to final.png from Step 4")
    parser.add_argument("--qa", required=True, help="Path to qa_report.json from Step 4")
    parser.add_argument("--json", required=True, help="Path to garment_analysis.json from Step 1")
    parser.add_argument("--sku", required=True, help="Product SKU identifier")
    parser.add_argument("--variant", required=True, help="Product variant (e.g., color)")
    parser.add_argument("--version", default="v1", help="Asset version")
    parser.add_argument("--out", default="./step5", help="Output directory")
    
    # Delivery options
    parser.add_argument("--quality", type=int, default=90, help="Quality for web formats")
    parser.add_argument("--enable-avif", action="store_true", help="Enable AVIF format generation")
    parser.add_argument("--thumb-size", type=int, default=384, help="Thumbnail size")
    
    # Cloud storage options
    parser.add_argument("--s3-bucket", help="S3 bucket name for upload")
    parser.add_argument("--s3-prefix", default="", help="S3 key prefix")
    parser.add_argument("--s3-public-url", help="S3 public URL base (e.g., CloudFront domain)")
    parser.add_argument("--gcs-bucket", help="Google Cloud Storage bucket name")
    parser.add_argument("--gcs-credentials", help="Path to GCS service account JSON")

    args = parser.parse_args()

    # Validate inputs
    final_path = Path(args.final)
    qa_path = Path(args.qa)
    json_path = Path(args.json)
    
    if not final_path.exists():
        raise FileNotFoundError(f"Final image not found: {final_path}")
    if not qa_path.exists():
        raise FileNotFoundError(f"QA report not found: {qa_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Analysis JSON not found: {json_path}")

    # Create delivery configuration
    delivery_config = DeliveryConfig(
        quality=args.quality,
        enable_avif=args.enable_avif,
        thumb_size=args.thumb_size
    )

    # Create cloud configuration if specified
    cloud_config = None
    if args.s3_bucket:
        cloud_config = CloudConfig(
            provider="s3",
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            public_url_base=args.s3_public_url or f"https://{args.s3_bucket}.s3.amazonaws.com",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
    elif args.gcs_bucket:
        cloud_config = CloudConfig(
            provider="gcs",
            bucket=args.gcs_bucket,
            public_url_base=f"https://storage.googleapis.com/{args.gcs_bucket}",
            gcs_credentials_path=args.gcs_credentials
        )

    # Initialize delivery engine
    engine = DeliveryEngine(delivery_config, cloud_config)

    try:
        # Process and deliver
        manifest = engine.process_and_deliver(
            final_image_path=final_path,
            qa_report_path=qa_path,
            analysis_json_path=json_path,
            sku=args.sku,
            variant=args.variant,
            version=args.version,
            output_dir=Path(args.out)
        )

        # Print results
        print(f"✅ Delivery complete! Output saved to: {args.out}")
        print(f"📦 SKU: {args.sku}-{args.variant} ({args.version})")
        print(f"🖼️  Assets generated: {manifest.processing_metadata['total_assets']}")
        
        asset_summary = {}
        for key, asset in manifest.assets.items():
            if asset.format not in asset_summary:
                asset_summary[asset.format] = 0
            asset_summary[asset.format] += 1
        
        for format, count in asset_summary.items():
            print(f"   - {format}: {count} variant(s)")
            
        if cloud_config:
            print(f"☁️  Uploaded to: {cloud_config.provider}://{cloud_config.bucket}")
            
        total_size_mb = sum(asset.size_bytes for asset in manifest.assets.values()) / (1024 * 1024)
        print(f"💾 Total size: {total_size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"Delivery processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
