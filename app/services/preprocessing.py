from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from loguru import logger

from app.models.schemas import BBox
from app.core.constants import (
    MEDIUM_PADDING_RATIO,
    ROTATION_ANGLES,
    ENABLE_HORIZONTAL_FLIP,
)


@dataclass
class PreprocessingService:
    """
    Image preprocessing service with realistic augmentation.
    
    Features:
    - Multi-crop generation (tight, medium, full) with fixed 15% padding
    - Realistic augmentation (±5° tilts + horizontal flip)
    - Mask application with smart background handling
    - Bbox validation and clamping
    
    Best Practices:
    - Always clamp bboxes before cropping
    - Use augmentation for products with varying camera angles
    - Apply masks to remove background noise
    """
    enable_rotation_aug: bool = True  # Enable augmentation (tilts + flip)
    
    def clamp_bbox(self, bbox: BBox, w: int, h: int) -> BBox:
        x1 = max(0, min(w - 1, bbox.x1))
        y1 = max(0, min(h - 1, bbox.y1))
        x2 = max(0, min(w, bbox.x2))
        y2 = max(0, min(h, bbox.y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bbox after clamping")
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def crop_base(self, img: Image.Image, bbox: BBox) -> tuple[dict[str, Image.Image], dict[str, BBox]]:
        """
        Generate base crops (tight, medium, full) without rotations.
        Returns crops and their corresponding bboxes for mask extraction.
        
        Simplified approach:
        - Tight: Exact bbox crop (pure object)
        - Medium: Bbox + fixed 15% padding (object with immediate context)
        - Full: Entire image (full scene context)
        
        Args:
            img: Input image
            bbox: Bounding box for object
            
        Returns:
            Tuple of (crops_dict, bboxes_dict) where:
            - crops_dict: {"tight": crop, "medium": crop, "full": crop}
            - bboxes_dict: {"tight": bbox, "medium": bbox, "full": bbox}
        """
        w, h = img.size
        
        # Calculate object dimensions
        obj_w = bbox.x2 - bbox.x1
        obj_h = bbox.y2 - bbox.y1
        obj_size = max(obj_w, obj_h)
        
        # Tight crop: Exact bbox (no expansion, no minimum size)
        tight_crop = img.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        tight_bbox = bbox
        
        # Medium crop: Fixed 15% padding for consistent context
        padding_ratio = MEDIUM_PADDING_RATIO  # 0.15 from constants
        pad = int(padding_ratio * obj_size)
        
        mx1 = max(0, bbox.x1 - pad)
        my1 = max(0, bbox.y1 - pad)
        mx2 = min(w, bbox.x2 + pad)
        my2 = min(h, bbox.y2 + pad)
        
        medium_crop = img.crop((mx1, my1, mx2, my2))
        medium_bbox = BBox(x1=mx1, y1=my1, x2=mx2, y2=my2)
        
        logger.debug(
            f"Crops generated: tight={obj_w:.0f}×{obj_h:.0f}px, "
            f"medium={mx2-mx1:.0f}×{my2-my1:.0f}px (15% padding)"
        )
        
        # Full crop: Entire image
        full_crop = img
        full_bbox = BBox(x1=0, y1=0, x2=w, y2=h)
        
        crops = {
            "tight": tight_crop,
            "medium": medium_crop,
            "full": full_crop
        }
        
        bboxes = {
            "tight": tight_bbox,
            "medium": medium_bbox,
            "full": full_bbox
        }
        
        return crops, bboxes
    
    def add_rotated_crops(self, base_crops: dict[str, Image.Image]) -> dict[str, Image.Image]:
        """
        Add realistic augmented crops from base crops.
        
        Generates subtle rotations (±5°) and horizontal flip to match real-world variations:
        - Slight tilts (±5°): Camera angle variations
        - Horizontal flip: Mirror images (left/right viewpoint)
        
        Use this AFTER masking to ensure augmentations have the same masking as base crops.
        
        Args:
            base_crops: Dictionary with "tight", "medium", "full" crops
            
        Returns:
            Dictionary with base crops + augmented variants
        """
        crops = base_crops.copy()
        
        if self.enable_rotation_aug:
            logger.debug("Adding realistic augmentations: ±5° tilts + horizontal flip")
            try:
                # Process tight and medium crops (full stays as-is)
                for crop_name in ["tight", "medium"]:
                    if crop_name not in base_crops:
                        continue
                    
                    base_crop = base_crops[crop_name]
                    
                    # Add subtle rotations (±5°)
                    for angle in ROTATION_ANGLES:  # [-5, 5]
                        rotated = base_crop.rotate(
                            angle,
                            expand=True,  # Expand canvas to fit rotated image
                            resample=Image.Resampling.BILINEAR,
                            fillcolor=(128, 128, 128)  # Gray fill for expanded areas
                        )
                        crops[f"{crop_name}_rot{angle:+d}"] = rotated
                    
                    # Add horizontal flip (mirror image)
                    if ENABLE_HORIZONTAL_FLIP:
                        flipped = base_crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        crops[f"{crop_name}_flip"] = flipped
                
                augmented_count = len([k for k in crops if 'rot' in k or 'flip' in k])
                logger.debug(f"Added {augmented_count} augmented crops (tilts + flips)")
                
            except Exception as e:
                logger.warning(f"Failed to generate augmented crops (non-critical): {e}")
        
        return crops

    def crop_multi(self, img: Image.Image, bbox: BBox) -> dict[str, Image.Image]:
        """
        Generate multiple crops for robust feature extraction.
        
        Crops:
        - tight: Exact bbox crop (focused on object)
        - medium: Bbox with 15% padding (includes context)
        - full: Full image (global context)
        - rotated variants: 90°, 180°, 270° (if rotation_aug enabled)
        
        Args:
            img: Input image
            bbox: Bounding box for object
            
        Returns:
            Dictionary of crop names to crop images
        """
        w, h = img.size
        tight = img.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))

        # Medium crop with 8% padding for context
        pad = int(0.08 * max(bbox.x2 - bbox.x1, bbox.y2 - bbox.y1))
        mx1 = max(0, bbox.x1 - pad)
        my1 = max(0, bbox.y1 - pad)
        mx2 = min(w, bbox.x2 + pad)
        my2 = min(h, bbox.y2 + pad)
        medium = img.crop((mx1, my1, mx2, my2))

        crops = {"tight": tight, "medium": medium, "full": img}
        
        # Add rotation-augmented crops for angle invariance
        # Handles products photographed at different angles (0°, 90°, 180°, 270°)
        if self.enable_rotation_aug:
            logger.debug("Adding rotation-augmented crops for angle invariance")
            try:
                for angle in [90, 180, 270]:
                    # Rotate crops (expand=True maintains full content)
                    crops[f"tight_rot{angle}"] = tight.rotate(
                        angle, expand=True, resample=Image.Resampling.BILINEAR
                    )
                    crops[f"medium_rot{angle}"] = medium.rotate(
                        angle, expand=True, resample=Image.Resampling.BILINEAR
                    )
                logger.debug(f"Added {len([k for k in crops if 'rot' in k])} rotated crops")
            except Exception as e:
                logger.warning(f"Failed to generate rotated crops (non-critical): {e}")
        
        return crops

    def apply_mask_on_crop(
        self, 
        crop: Image.Image, 
        mask: np.ndarray | None,
        bbox: BBox | None = None
    ) -> Image.Image:
        """
        Apply mask to crop image, removing background for cleaner embeddings.
        
        Uses polygon mask to isolate the product from background clutter.
        Background is replaced with pure white (255, 255, 255) for consistency.
        
        Benefits of white background:
        - Removes color bias (same product in different rooms = same background)
        - Product-catalog style (like Amazon/e-commerce)
        - Consistent embeddings across all products
        
        Args:
            crop: The cropped image
            mask: Full image mask (H, W) as boolean array from polygon segmentation
            bbox: Bounding box used for crop (to extract mask region)
            
        Returns:
            Masked crop image with white background
        """
        if mask is None:
            logger.debug("No mask provided, returning original crop")
            return crop
        
        try:
            # Convert crop to numpy
            crop_array = np.array(crop)
            
            if bbox is not None:
                # Extract mask region corresponding to the crop
                crop_mask = mask[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
                
                # Resize mask if needed to match crop size
                if crop_mask.shape[:2] != crop_array.shape[:2]:
                    from PIL import Image as PILImage
                    mask_img = PILImage.fromarray(crop_mask.astype(np.uint8) * 255)
                    mask_img = mask_img.resize(
                        (crop_array.shape[1], crop_array.shape[0]),
                        Image.Resampling.NEAREST
                    )
                    crop_mask = np.array(mask_img) > 128
            else:
                # If no bbox, assume mask matches crop size
                crop_mask = mask
                if crop_mask.shape[:2] != crop_array.shape[:2]:
                    logger.warning(
                        f"Mask size mismatch: {crop_mask.shape} vs crop {crop_array.shape}, "
                        f"returning original crop"
                    )
                    return crop
            
            # Apply mask
            masked_array = crop_array.copy()
            
            # Use pure white background for consistency (product-catalog style)
            if crop_mask.sum() > 0:  # Check if mask is not empty
                # Replace background with white (255, 255, 255)
                # This removes color bias and creates consistent backgrounds
                masked_array[~crop_mask] = [255, 255, 255]  # Pure white RGB
                
                coverage = crop_mask.sum() / crop_mask.size * 100
                logger.debug(f"Applied mask with white background ({coverage:.1f}% foreground coverage)")
            else:
                # If mask is empty, return original (segmentation failed)
                logger.warning("Empty mask, returning original crop")
                return crop
            
            return Image.fromarray(masked_array)
            
        except Exception as e:
            logger.error(f"Error applying mask to crop: {e}")
            # Fallback: return original crop
            return crop