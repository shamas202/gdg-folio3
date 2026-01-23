from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPModel, CLIPVisionModel, CLIPImageProcessor
from loguru import logger


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2 normalize a vector."""
    n = np.linalg.norm(x) + 1e-12
    return x / n


@dataclass
class MultiScaleEmbeddingService:
    """
    Multi-scale embedding service for handling extreme size variations.
    Embeds images at multiple scales and fuses the results.
    
    Features:
    - Handles 10x-100x size variations
    - Multi-scale pyramid (small, medium, large)
    - Weighted fusion with emphasis on middle scale
    - Rotation augmentation support
    - Smart caching for performance
    
    Best Practices:
    - Always normalize embeddings
    - Use proper error handling
    - Log performance metrics
    - Cache projection matrices
    """
    instance_model_name: str
    semantic_model_name: str
    target_dim: int
    scales: list[int] | None = None  # e.g., [224, 384, 512]
    enable_rotation_aug: bool = False  # Enable rotation during embedding

    def __post_init__(self) -> None:
        self._loaded = False
        if self.scales is None:
            # Default scales: small (global), medium (balanced), large (details)
            self.scales = [224, 384, 512]
        
        self._inst_proc = None
        self._inst_model = None
        self._sem_proc = None
        self._sem_model = None
        
        logger.info(f"MultiScaleEmbeddingService initialized with scales: {self.scales}")

    async def load(self) -> None:
        """Load embedding models onto device (GPU preferred, CPU fallback)."""
        if self._loaded:
            logger.debug("MultiScaleEmbeddingService already loaded")
            return

        # Determine device
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            logger.info("Multi-scale embedding models using GPU (CUDA)")
        else:
            self._device = torch.device('cpu')
            logger.warning("Multi-scale embedding models using CPU (GPU not available)")

        try:
            # Load instance tower (ViT for fine-grained features)
            logger.info(f"Loading instance model: {self.instance_model_name}")
            self._inst_proc = AutoImageProcessor.from_pretrained(
                self.instance_model_name, 
                use_fast=True
            )
            self._inst_model = AutoModel.from_pretrained(self.instance_model_name)
            self._inst_model.to(self._device)
            self._inst_model.eval()
            logger.info("Instance model loaded successfully")

            # Load semantic tower (CLIP for semantic features)
            logger.info(f"Loading semantic model: {self.semantic_model_name}")
            
            model_name_lower = self.semantic_model_name.lower()
            if "clip" in model_name_lower:
                # Use CLIPImageProcessor for CLIP models
                self._sem_proc = CLIPImageProcessor.from_pretrained(self.semantic_model_name)
            else:
                self._sem_proc = AutoImageProcessor.from_pretrained(
                    self.semantic_model_name, 
                    use_fast=True
                )
            
            if "clip" in model_name_lower:
                try:
                    # Try loading as CLIPVisionModel first (vision-only, more efficient)
                    self._sem_model = CLIPVisionModel.from_pretrained(self.semantic_model_name)
                    logger.debug("Loaded as CLIPVisionModel")
                except Exception as e:
                    logger.debug(f"CLIPVisionModel failed ({e}), loading full CLIP model")
                    # Fallback: load full CLIP model and extract vision_model
                    clip_model = CLIPModel.from_pretrained(self.semantic_model_name)
                    self._sem_model = clip_model.vision_model
            else:
                # For non-CLIP models, use AutoModel
                self._sem_model = AutoModel.from_pretrained(self.semantic_model_name)
            
            self._sem_model.to(self._device)
            self._sem_model.eval()
            logger.info("Semantic model loaded successfully")
            
            # Using full precision (FP32) for best accuracy
            logger.info("Models loaded in full precision (FP32) for maximum accuracy")
            
            # Warmup: Run inference once to compile/optimize models
            logger.info("Warming up multi-scale embedding models...")
            dummy_img = Image.new('RGB', (224, 224), color='gray')
            try:
                for scale in self.scales:
                    _ = self._embed_one_scale(dummy_img, scale, self._inst_proc, self._inst_model)
                    _ = self._embed_one_scale(dummy_img, scale, self._sem_proc, self._sem_model)
                logger.info("Warmup complete - models ready for inference")
            except Exception as e:
                logger.warning(f"Warmup failed (non-critical): {e}")
            
            self._loaded = True
            logger.success("MultiScaleEmbeddingService loaded and ready")
            
        except Exception as e:
            logger.error(f"Failed to load MultiScaleEmbeddingService: {e}")
            raise

    async def unload(self) -> None:
        """Unload models from memory."""
        logger.info("Unloading MultiScaleEmbeddingService")
        self._inst_proc = None
        self._inst_model = None
        self._sem_proc = None
        self._sem_model = None
        self._loaded = False
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _embed_one_scale(self, img: Image.Image, scale: int, proc, model) -> np.ndarray:
        """
        Embed image at a specific scale.
        
        Args:
            img: Input image
            scale: Target size (will maintain aspect ratio)
            proc: Image processor
            model: Embedding model
            
        Returns:
            Normalized embedding vector
        """
        try:
            # Resize image to target scale while maintaining aspect ratio
            img_resized = self._resize_to_scale(img, scale)
            
            inputs = proc(images=img_resized, return_tensors="pt")
            
            # Move inputs to device
            device = self._device if hasattr(self, '_device') else next(model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                out = model(**inputs)

            # Extract features (handle different model architectures)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vec = out.pooler_output[0].cpu().numpy()
            elif hasattr(out, "pooled_output") and out.pooled_output is not None:
                # CLIPVisionModel pooled output
                vec = out.pooled_output[0].cpu().numpy()
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                # Mean pool over sequence dimension
                lhs = out.last_hidden_state[0].cpu().numpy()
                vec = lhs.mean(axis=0)
            else:
                raise ValueError(f"Unknown model output format: {type(out)}")

            return _l2_normalize(vec.astype(np.float32))
            
        except Exception as e:
            logger.error(f"Error embedding at scale {scale}: {e}")
            raise

    def _resize_to_scale(self, img: Image.Image, target_size: int) -> Image.Image:
        """
        Resize image to target size while maintaining aspect ratio.
        Uses high-quality LANCZOS resampling.
        
        Args:
            img: Input image
            target_size: Target size for longer edge
            
        Returns:
            Resized image
        """
        w, h = img.size
        
        # Calculate new dimensions maintaining aspect ratio
        if w > h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        
        # Ensure minimum size (avoid too small images)
        new_w = max(new_w, 32)
        new_h = max(new_h, 32)
        
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def embed_crops(self, crops: dict[str, Image.Image]) -> list[float]:
        """
        Embed crops at multiple scales for robustness.
        Handles extreme size variations (10x-100x difference).
        
        Args:
            crops: Dictionary of crop images (tight, medium, full, rotated variants)
            
        Returns:
            Fused embedding vector as list
        """
        if not self._loaded:
            raise RuntimeError("MultiScaleEmbeddingService not loaded. Call load() first.")

        # Separate rotated crops from regular crops
        regular_crops = {k: v for k, v in crops.items() if 'rot' not in k}
        rotated_crops = {k: v for k, v in crops.items() if 'rot' in k}
        
        # Embed regular crops at multiple scales
        inst_vecs_all_scales = []
        sem_vecs_all_scales = []
        
        for scale in self.scales:
            inst_vecs_this_scale = []
            sem_vecs_this_scale = []
            
            # Process regular crops
            for crop_name, crop in regular_crops.items():
                try:
                    inst_vec = self._embed_one_scale(crop, scale, self._inst_proc, self._inst_model)
                    sem_vec = self._embed_one_scale(crop, scale, self._sem_proc, self._sem_model)
                    
                    inst_vecs_this_scale.append(inst_vec)
                    sem_vecs_this_scale.append(sem_vec)
                except Exception as e:
                    logger.warning(f"Failed to embed crop '{crop_name}' at scale {scale}: {e}")
                    continue
            
            # If rotation augmentation is enabled, add rotated crops
            if self.enable_rotation_aug and rotated_crops:
                for crop_name, crop in rotated_crops.items():
                    try:
                        inst_vec = self._embed_one_scale(crop, scale, self._inst_proc, self._inst_model)
                        sem_vec = self._embed_one_scale(crop, scale, self._sem_proc, self._sem_model)
                        
                        # Weight rotated crops less than regular crops
                        inst_vecs_this_scale.append(inst_vec * 0.7)
                        sem_vecs_this_scale.append(sem_vec * 0.7)
                    except Exception as e:
                        logger.debug(f"Failed to embed rotated crop '{crop_name}': {e}")
                        continue
            
            if not inst_vecs_this_scale:
                logger.error(f"No valid embeddings at scale {scale}")
                raise RuntimeError(f"Failed to embed any crops at scale {scale}")
            
            # Average across crops for this scale
            inst_scale_avg = _l2_normalize(np.mean(inst_vecs_this_scale, axis=0))
            sem_scale_avg = _l2_normalize(np.mean(sem_vecs_this_scale, axis=0))
            
            inst_vecs_all_scales.append(inst_scale_avg)
            sem_vecs_all_scales.append(sem_scale_avg)
        
        # Average across scales with weighted emphasis on middle scale
        # Small scale captures global structure, large scale captures details
        scale_weights = self._get_scale_weights(len(self.scales))
        
        inst = _l2_normalize(
            np.average(inst_vecs_all_scales, axis=0, weights=scale_weights)
        )
        sem = _l2_normalize(
            np.average(sem_vecs_all_scales, axis=0, weights=scale_weights)
        )

        # Fuse instance and semantic embeddings
        fused = np.concatenate([inst, sem], axis=0)
        fused = _l2_normalize(fused)

        # Project to target dimension
        fused = self._fit_dim(fused, self.target_dim)
        
        logger.debug(
            f"Multi-scale embedding: {len(self.scales)} scales, "
            f"{len(regular_crops)} regular crops, "
            f"{len(rotated_crops)} rotated crops, "
            f"final dim: {len(fused)}"
        )
        
        return fused.astype(np.float32).tolist()

    def _get_scale_weights(self, num_scales: int) -> np.ndarray:
        """
        Get weights for different scales.
        Middle scales get higher weight as they balance detail and context.
        
        Args:
            num_scales: Number of scales used
            
        Returns:
            Weight array (sums to 1.0)
        """
        if num_scales == 1:
            return np.array([1.0])
        elif num_scales == 2:
            return np.array([0.4, 0.6])  # Prefer larger scale
        elif num_scales == 3:
            return np.array([0.25, 0.5, 0.25])  # Prefer middle scale
        else:
            # Gaussian-like weights centered on middle
            weights = np.exp(-((np.arange(num_scales) - num_scales // 2) ** 2) / (num_scales / 2))
            return weights / weights.sum()

    def _fit_dim(self, v: np.ndarray, dim: int) -> np.ndarray:
        """
        Project vector to target dimension using random projection.
        Preserves distances better than truncation or zero-padding.
        Uses cached projection matrix for consistency.
        
        Args:
            v: Input vector
            dim: Target dimension
            
        Returns:
            Projected vector
        """
        if v.shape[0] == dim:
            return v
        
        # Use cached projection matrix for consistency across calls
        cache_key = f"_projection_matrix_{v.shape[0]}_{dim}"
        if not hasattr(self, cache_key):
            # Create random projection matrix with fixed seed for reproducibility
            rng = np.random.RandomState(42)
            proj = rng.randn(v.shape[0], dim).astype(np.float32)
            
            # Normalize columns to preserve norm (approximate isometry)
            proj = proj / np.sqrt(v.shape[0])
            
            setattr(self, cache_key, proj)
            logger.debug(f"Created projection matrix: {v.shape[0]} -> {dim}")
        
        proj_matrix = getattr(self, cache_key)
        
        # Project to target dimension
        projected = v @ proj_matrix
        
        # Normalize to unit vector
        return _l2_normalize(projected)

