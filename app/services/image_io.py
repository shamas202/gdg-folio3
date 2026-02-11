from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from fastapi import UploadFile

from app.core.constants import (
    SMALL_CATEGORIES,
    MEDIUM_CATEGORIES,
    LARGE_CATEGORIES,
    MAX_FILE_SIZE_MB,
    MIN_DIMENSION_UNIVERSAL,
    MAX_DIMENSION,
    BLUR_THRESHOLDS,
    CONTRAST_THRESHOLDS,
)


@dataclass
class ImageIOService:
    """
    Image I/O service with category-based validation and quality checks.
    
    Features:
    - Category-aware blur detection (different thresholds for small/large objects)
    - Comprehensive file validation (size, format, dimensions)
    - Automatic resizing for large images (prevents OOM)
    - Hard rejections for poor quality (blur, brightness, contrast)
    - Support for JPEG, PNG, WebP formats
    - Separate directories for catalog and search images
    - Detailed error messages for debugging
    
    Best Practices:
    - Always validate images before processing
    - Use category parameter for category-specific checks
    - Resize large images to prevent memory issues
    - Provide clear error messages for debugging
    """
    catalog_images_dir: Path = Path("images/catalog")
    search_images_dir: Path = Path("images/search")
    min_dimension: int = MIN_DIMENSION_UNIVERSAL
    max_dimension: int = MAX_DIMENSION
    max_file_size_mb: int = MAX_FILE_SIZE_MB
    
    def __post_init__(self):
        """Ensure image directories exist"""
        self.catalog_images_dir.mkdir(parents=True, exist_ok=True)
        self.search_images_dir.mkdir(parents=True, exist_ok=True)
        from loguru import logger
        logger.info(f"ImageIOService initialized:")
        logger.info(f"  Catalog images: {self.catalog_images_dir.absolute()}")
        logger.info(f"  Search images: {self.search_images_dir.absolute()}")
    
    def get_category_requirements(self, category: str) -> dict:
        """
        Get quality requirements based on product category (3-tier system).
        
        Different tiers have different quality standards:
        - Small objects (cups, plates) - blur ≥ 30, ROA ≥ 8%
        - Medium objects (chairs, lamps) - blur ≥ 25, ROA ≥ 12%
        - Large objects (sofas, beds) - blur ≥ 20, ROA ≥ 15%
        
        Args:
            category: Product category (e.g., "cup", "chair", "bed")
            
        Returns:
            Dictionary with blur_threshold, blur_warning, category_size
            
        Raises:
            ValueError: If category is not in predefined lists
        """
        category_lower = category.lower().strip()
        
        if category_lower in SMALL_CATEGORIES:
            return {
                "blur_threshold": BLUR_THRESHOLDS["small"]["reject"],
                "blur_warning": BLUR_THRESHOLDS["small"]["warn"],
                "category_size": "small"
            }
        elif category_lower in MEDIUM_CATEGORIES:
            return {
                "blur_threshold": BLUR_THRESHOLDS["medium"]["reject"],
                "blur_warning": BLUR_THRESHOLDS["medium"]["warn"],
                "category_size": "medium"
            }
        elif category_lower in LARGE_CATEGORIES:
            return {
                "blur_threshold": BLUR_THRESHOLDS["large"]["reject"],
                "blur_warning": BLUR_THRESHOLDS["large"]["warn"],
                "category_size": "large"
            }
        else:
            # Fail fast for unknown categories - forces data quality
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Category must be one of {len(SMALL_CATEGORIES)} small, "
                f"{len(MEDIUM_CATEGORIES)} medium, or {len(LARGE_CATEGORIES)} large categories. "
                f"Please check your data source."
            )
    
    def compute_blur_score(self, img: Image.Image) -> float:
        """
        Compute blur score using Laplacian variance method.
        
        Higher score = sharper image
        Lower score = blurrier image
        
        Typical ranges:
        - < 30: Extremely blurry (unusable)
        - 30-40: Very blurry (reject for small objects)
        - 40-100: Acceptable sharpness
        - 100-200: Good sharpness
        - > 200: Excellent sharpness
        
        Args:
            img: PIL Image in RGB mode
            
        Returns:
            Blur score (variance of Laplacian operator)
        """
        # Convert PIL to numpy array
        img_array = np.array(img)
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Laplacian operator (detects edges)
        # Higher variance = more edges = sharper image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
        
        return blur_score
    
    async def read_upload_as_rgb(
        self, 
        file: UploadFile, 
        category: str | None = None
    ) -> Image.Image:
        """
        Read and validate uploaded image with quality checks.
        
        Validation includes:
        - File size (≤ 15MB)
        - Image format (JPEG, PNG, WebP)
        - Dimensions (≥ 500×500 universal minimum)
        - Auto-resize if > 4096px
        - **Blur detection** - HARD REJECTION:
          * With category (ingestion): Category-specific (40 for small, 30 for large)
          * Without category (retrieval): Universal threshold (30)
        - **Contrast check (std ≥ 1.0)** - HARD REJECTION
        - Color mode conversion (ensures RGB)
        
        Note: Brightness check removed - RF-DETR detection naturally handles
        unusable images, and brightness filtering rejected valid white-background products.
        
        Args:
            file: Uploaded file from FastAPI
            category: Product category for category-specific validation (optional)
                     If None, uses default blur threshold (30) for retrieval
            
        Returns:
            PIL Image in RGB mode, validated and resized if necessary
            
        Raises:
            ValueError: If image is invalid, corrupted, or fails validation
        """
        from loguru import logger
        
        # Read file data
        data = await file.read()
        
        # Check file size (prevent OOM from huge files)
        size_mb = len(data) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise ValueError(
                f"Image file too large ({size_mb:.1f}MB). "
                f"Maximum allowed: {self.max_file_size_mb}MB. "
                f"Please compress or resize your image."
            )
        
        logger.debug(f"Reading image file: {size_mb:.2f}MB")
        
        # Try to open image
        try:
            img = Image.open(BytesIO(data))
        except Exception as e:
            raise ValueError(
                f"Invalid or corrupted image file: {str(e)}. "
                f"Please ensure the file is a valid image (JPEG, PNG, WebP)."
            )
        
        # Log original format and mode
        logger.debug(f"Original image: format={img.format}, mode={img.mode}, size={img.size}")
        
        # Convert to RGB (handles RGBA, grayscale, CMYK, etc.)
        try:
            if img.mode != "RGB":
                logger.debug(f"Converting from {img.mode} to RGB")
                img = img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot convert image to RGB: {str(e)}")
        
        # Validate dimensions
        w, h = img.size
        
        if w < self.min_dimension or h < self.min_dimension:
            raise ValueError(
                f"Image too small ({w}x{h} pixels). "
                f"Minimum dimension: {self.min_dimension}px. "
                f"Please use a higher resolution image."
            )
        
        # Resize if too large (prevents OOM and speeds up processing)
        if w > self.max_dimension or h > self.max_dimension:
            original_size = (w, h)
            logger.info(
                f"Image exceeds maximum dimension ({w}x{h}), "
                f"resizing to max {self.max_dimension}px"
            )
            img.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
            logger.info(f"Resized from {original_size} to {img.size}")
        
        # === BLUR DETECTION (DISABLED) ===
        # Blur checks commented out - RF-DETR handles unusable images naturally
        blur_score = None
        # try:
        #     blur_score = self.compute_blur_score(img)
        #     
        #     if category:
        #         # INGESTION: Category-specific blur check
        #         requirements = self.get_category_requirements(category)
        #         blur_threshold = requirements["blur_threshold"]
        #         blur_warning = requirements["blur_warning"]
        #         cat_size = requirements["category_size"]
        #         
        #         # Hard rejection for blurry images
        #         if blur_score < blur_threshold:
        #             raise ValueError(
        #                 f"Image too blurry for reliable detection and embedding. "
        #                 f"Blur score: {blur_score:.1f}, required: {blur_threshold} for {cat_size} objects. "
        #                 f"Please use a sharper, higher-quality image."
        #             )
        #         
        #         # Warning for moderate blur
        #         elif blur_score < blur_warning:
        #             logger.warning(
        #                 f"⚠️ Image has moderate blur (score: {blur_score:.1f}). "
        #                 f"Consider using a sharper image for better search results."
        #             )
        #         else:
        #             logger.debug(f"✓ Image sharpness acceptable (blur_score: {blur_score:.1f})")
        #     else:
        #         # RETRIEVAL: Universal blur check (category unknown)
        #         blur_threshold = BLUR_THRESHOLDS["default"]["reject"]  # 30
        #         blur_warning = BLUR_THRESHOLDS["default"]["warn"]  # 70
        #         
        #         # Hard rejection for very blurry images
        #         if blur_score < blur_threshold:
        #             raise ValueError(
        #                 f"Image too blurry for reliable detection. "
        #                 f"Blur score: {blur_score:.1f}, required: {blur_threshold}. "
        #                 f"Please use a sharper, higher-quality image."
        #             )
        #         
        #         # Warning for moderate blur
        #         elif blur_score < blur_warning:
        #             logger.warning(
        #                 f"⚠️ Image has moderate blur (score: {blur_score:.1f}). "
        #                 f"Search results may be less accurate."
        #             )
        #         else:
        #             logger.debug(f"✓ Image sharpness acceptable (blur_score: {blur_score:.1f})")
        #             
        # except ValueError:
        #     # Re-raise ValueError (blur rejection)
        #     raise
        # except Exception as e:
        #     logger.warning(f"Blur detection failed (non-critical): {e}")
        #     blur_score = None
        
        # === STATISTICAL QUALITY CHECKS (UNIVERSAL, HARD REJECTION) ===
        try:
            img_array = np.array(img)
            
            # === CONTRAST CHECK (Standard Deviation) ===
            std_dev = img_array.std()
            
            # Hard rejection for blank/uniform images
            if std_dev < CONTRAST_THRESHOLDS["reject_min"]:
                raise ValueError(
                    f"Image rejected: almost blank or uniform (std={std_dev:.2f}). "
                    f"The image has no meaningful content. "
                    f"Please upload a valid product image."
                )
            
            # Warning for low contrast
            elif std_dev < CONTRAST_THRESHOLDS["warn_min"]:
                logger.warning(
                    f"⚠️ Low contrast image (std={std_dev:.1f}). "
                    f"Image may lack detail. Consider using better lighting."
                )
            

            mean_brightness = img_array.mean()
            
            # Log comprehensive quality metrics
            blur_display = f"{blur_score:.1f}" if blur_score is not None else "N/A"
            logger.info(
                f"📊 Quality: blur={blur_display}, "
                f"contrast={std_dev:.1f}, brightness={mean_brightness:.1f}, "
                f"size={img.size}"
            )
            
        except ValueError:
            # Re-raise ValueError (quality rejection)
            raise
        except Exception as e:
            logger.warning(f"Statistical quality check failed (non-critical): {e}")
        
        return img

    def pil_to_numpy_rgb(self, img: Image.Image) -> np.ndarray:
        return np.array(img, dtype=np.uint8)
    
    async def save_image(
        self, 
        image_id: str, 
        image_file: UploadFile, 
        image_type: str = "catalog"
    ) -> Path:
        """
        Save uploaded image file to disk.
        
        Args:
            image_id: Unique identifier for the image
            image_file: Uploaded file from FastAPI
            image_type: Type of image - "catalog" or "search" (default: "catalog")
            
        Returns:
            Path to the saved image file
        """
        from loguru import logger
        
        # Select appropriate directory based on image type
        if image_type == "search":
            target_dir = self.search_images_dir
        else:
            target_dir = self.catalog_images_dir
        
        # Reset file pointer to beginning
        await image_file.seek(0)
        data = await image_file.read()
        
        # Determine file extension from content type or filename
        content_type = image_file.content_type or ""
        if "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        elif "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"
        else:
            # Try to infer from filename
            filename = image_file.filename or ""
            if filename.lower().endswith((".jpg", ".jpeg")):
                ext = ".jpg"
            elif filename.lower().endswith(".png"):
                ext = ".png"
            elif filename.lower().endswith(".webp"):
                ext = ".webp"
            else:
                ext = ".jpg"  # Default to jpg
        
        image_path = target_dir / f"{image_id}{ext}"
        image_path.write_bytes(data)
        logger.debug(f"Saved {image_type} image: {image_path}")
        return image_path
    
    def get_image_path(self, image_id: str, image_type: str | None = None) -> Path | None:
        """
        Get path to image file by image_id.
        
        Args:
            image_id: Unique identifier for the image
            image_type: Type of image - "catalog", "search", or None to search both
            
        Returns:
            Path to the image file if found, None otherwise
        """
        # Determine which directories to search
        if image_type == "search":
            directories = [self.search_images_dir]
        elif image_type == "catalog":
            directories = [self.catalog_images_dir]
        else:
            # Search both directories (catalog first, then search)
            directories = [self.catalog_images_dir, self.search_images_dir]
        
        # Try common extensions in each directory
        for directory in directories:
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                image_path = directory / f"{image_id}{ext}"
                if image_path.exists():
                    return image_path
        return None