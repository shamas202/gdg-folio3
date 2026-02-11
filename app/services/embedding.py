from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPModel, CLIPVisionModel


class EmbeddingService(ABC):
    @abstractmethod
    async def load(self) -> None: ...
    @abstractmethod
    async def unload(self) -> None: ...
    @abstractmethod
    def embed_crops(self, crops: dict[str, Image.Image]) -> list[float]:
        """
        Returns a single fused vector for the object using multi-crop + multi-tower.
        """
        raise NotImplementedError


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return x / n


@dataclass
class HFEmbeddingService(EmbeddingService):
    instance_model_name: str
    semantic_model_name: str
    target_dim: int

    def __post_init__(self) -> None:
        self._loaded = False

        self._inst_proc = None
        self._inst_model = None

        self._sem_proc = None
        self._sem_model = None

    async def load(self) -> None:
        if self._loaded:
            return

        from loguru import logger
        
        # Determine device (GPU preferred, CPU fallback) with CUDA optimizations
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            logger.info("Embedding models using GPU (CUDA)")
            
            # Enable CUDA optimizations (works on Windows, no Triton needed)
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            
            # Enable TF32 for faster matrix operations on Ampere+ GPUs (RTX 3060+)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set float32 matmul precision for speed (trades minimal precision for speed)
            torch.set_float32_matmul_precision('medium')
            
            logger.info("CUDA optimizations enabled (cudnn.benchmark, TF32, optimized matmul)")
        else:
            self._device = torch.device('cpu')
            logger.info("Embedding models using CPU (GPU not available)")

        # Instance tower (ViT)
        logger.info(f"Loading instance model: {self.instance_model_name}")
        self._inst_proc = AutoImageProcessor.from_pretrained(self.instance_model_name, use_fast=True)
        self._inst_model = AutoModel.from_pretrained(self.instance_model_name)
        self._inst_model.to(self._device)
        self._inst_model.eval()
        # Note: torch.compile() removed - using CUDA optimizations instead for Windows compatibility

        # Semantic tower (CLIP vision backbone)
        logger.info(f"Loading semantic model: {self.semantic_model_name}")
        self._sem_proc = AutoImageProcessor.from_pretrained(self.semantic_model_name, use_fast=True)
        
        # Check if it's a CLIP model - if so, use CLIPVisionModel or extract vision_model
        model_name_lower = self.semantic_model_name.lower()
        if "clip" in model_name_lower:
            try:
                # Try loading as CLIPVisionModel first (vision-only, more efficient)
                self._sem_model = CLIPVisionModel.from_pretrained(self.semantic_model_name)
            except Exception:
                # Fallback: load full CLIP model and extract vision_model
                clip_model = CLIPModel.from_pretrained(self.semantic_model_name)
                self._sem_model = clip_model.vision_model
        else:
            # For non-CLIP models, use AutoModel
            self._sem_model = AutoModel.from_pretrained(self.semantic_model_name)
        
        self._sem_model.to(self._device)
        self._sem_model.eval()
        # Note: torch.compile() removed - using CUDA optimizations instead for Windows compatibility
        
        # NOTE: Not using quantization/FP16 as per user request - focus on best accuracy
        logger.info("Models loaded in full precision (FP32) for best accuracy")
        
        # Warmup: Compile models with dummy input for faster first inference
        logger.info("Warming up embedding models...")
        dummy_img = Image.new('RGB', (224, 224), color='gray')
        try:
            _ = self._embed_one(dummy_img, self._inst_proc, self._inst_model)
            _ = self._embed_one(dummy_img, self._sem_proc, self._sem_model)
            logger.info("Warmup complete - models ready for inference")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")
        
        self._loaded = True

    async def unload(self) -> None:
        self._inst_proc = None
        self._inst_model = None
        self._sem_proc = None
        self._sem_model = None
        self._loaded = False

    def _embed_one(self, img: Image.Image, proc, model) -> np.ndarray:
        inputs = proc(images=img, return_tensors="pt")
        
        # Move inputs to device
        device = self._device if hasattr(self, '_device') else next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Use inference_mode() instead of no_grad() for better performance
        with torch.inference_mode():
            out = model(**inputs)

        # Generic pooling strategy:
        # - if model exposes pooler_output, use it
        # - else if it has pooled_output (CLIPVisionModel), use that
        # - else mean-pool last_hidden_state
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            vec = out.pooler_output[0].cpu().numpy()
        elif hasattr(out, "pooled_output") and out.pooled_output is not None:
            # CLIPVisionModel pooled output
            vec = out.pooled_output[0].cpu().numpy()
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            # Mean pool over sequence dimension (CLIPVisionModel, ViT, etc.)
            lhs = out.last_hidden_state[0].cpu().numpy()
            vec = lhs.mean(axis=0)
        else:
            # Fallback: try to get any tensor output
            raise ValueError(f"Unknown model output format: {type(out)}, available attributes: {dir(out)}")

        return _l2_normalize(vec.astype(np.float32))

    def embed_crops(self, crops: dict[str, Image.Image]) -> list[float]:
        if not self._loaded:
            raise RuntimeError("EmbeddingService not loaded")

        # Multi-crop average per tower
        inst_vecs = []
        sem_vecs = []
        for _, crop in crops.items():
            inst_vecs.append(self._embed_one(crop, self._inst_proc, self._inst_model))
            sem_vecs.append(self._embed_one(crop, self._sem_proc, self._sem_model))

        inst = _l2_normalize(np.mean(inst_vecs, axis=0))
        sem = _l2_normalize(np.mean(sem_vecs, axis=0))

        fused = np.concatenate([inst, sem], axis=0)
        fused = _l2_normalize(fused)

        # Ensure Pinecone dimension matches by deterministic projection/pad.
        fused = self._fit_dim(fused, self.target_dim)
        return fused.astype(np.float32).tolist()

    def _fit_dim(self, v: np.ndarray, dim: int) -> np.ndarray:
        """
        Project vector to target dimension using random projection.
        Preserves distances better than truncation or zero-padding.
        Uses cached projection matrix for consistency.
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
        
        proj_matrix = getattr(self, cache_key)
        
        # Project to target dimension
        projected = v @ proj_matrix
        
        # Normalize to unit vector
        return _l2_normalize(projected)