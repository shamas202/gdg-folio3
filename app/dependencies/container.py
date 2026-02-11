from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from fastapi import Request

from app.core.config import Settings

from app.repositories.pinecone_repo import PineconeVectorRepository

from app.services.image_io import ImageIOService
from app.services.preprocessing import PreprocessingService
from app.services.detection import DetectionService, RTDETRDetectionService
from app.services.detection_rfdetr import RFDETRDetectionService
from app.services.segmentation import (
    SegmentationService,
    SAM2SegmentationService,
    SAMLikeSegmentationService,
)
from app.services.segmentation_polygon import PolygonSegmentationService
from app.services.embedding import EmbeddingService, HFEmbeddingService
from app.services.embedding_multiscale import MultiScaleEmbeddingService
from app.services.attributes import AttributeService

from app.services.visual_matcher import VisualCrossEncoder
from app.services.rerank import (
    RerankService,
    VisualCrossEncoderRerankService,
)
from app.services.rerank_advanced import (
    MultiStageRerankService,
    HybridRerankService,
)

from app.services.search_service import SearchService


# ============================
# Dependency Injection Container
# ============================

@dataclass(frozen=True)
class Container:
    settings: Settings

    # Low-level services
    image_io: ImageIOService
    preprocessing: PreprocessingService
    detection: DetectionService
    segmentation: SegmentationService
    embedding: EmbeddingService
    attributes: AttributeService

    # Vector DB
    vectors: PineconeVectorRepository

    # Ranking
    rerank: RerankService

    # High-level orchestration
    search_service: SearchService

    # ----------------------------
    # Factory
    # ----------------------------
    @classmethod
    def from_settings(cls, settings: Settings) -> "Container":
        from loguru import logger
        
        # ---- Image & Preprocessing ----
        image_io = ImageIOService(
            catalog_images_dir=Path(settings.catalog_images_dir),
            search_images_dir=Path(settings.search_images_dir),
            min_dimension=settings.image_min_dimension,
            max_dimension=settings.image_max_dimension,
            max_file_size_mb=settings.image_max_size_mb,
        )
        
        preprocessing = PreprocessingService(
            enable_rotation_aug=settings.enable_rotation_augmentation,
        )

        # ---- Detection (Furniture/Products) ----
        detection_mode = settings.detection_mode.lower()
        
        if detection_mode == "runpod":
            # Use RunPod RF-DETR API (primary)
            detection = RFDETRDetectionService(
                api_url=settings.runpod_api_url,
                status_url=settings.runpod_status_url,
                api_key=settings.runpod_api_key,
                confidence_threshold=settings.runpod_confidence_threshold,
                max_wait_seconds=settings.runpod_max_wait_seconds,
            )
            logger.info("Using RF-DETR via RunPod API for detection")
        else:
            # Use local RT-DETR (fallback)
            try:
                detection = RTDETRDetectionService(model_path=settings.rtdetr_model_path)
                logger.info("Using local RT-DETR for detection")
            except FileNotFoundError as e:
                logger.error(f"Local RT-DETR model not found: {e}")
                logger.warning("Falling back to RunPod RF-DETR API")
                detection = RFDETRDetectionService(
                    api_url=settings.runpod_api_url,
                    status_url=settings.runpod_status_url,
                    api_key=settings.runpod_api_key,
                    confidence_threshold=settings.runpod_confidence_threshold,
                    max_wait_seconds=settings.runpod_max_wait_seconds,
                )

        # ---- Segmentation (Mask-aware) ----
        # Use polygon segmentation if RF-DETR API is used (provides polygon masks)
        # Otherwise use SAM2.1 or GrabCut
        if detection_mode == "runpod":
            segmentation = PolygonSegmentationService()
            logger.info("Using polygon segmentation (from RF-DETR API masks)")
        else:
            # Try to use SAM2.1 (Ultralytics) if available, fallback to GrabCut
            try:
                segmentation = SAM2SegmentationService(
                    model_path=settings.sam2_model_path,
                )
                logger.info("Using SAM2.1 (Ultralytics) for segmentation")
            except (ImportError, FileNotFoundError) as e:
                logger.warning(f"SAM2.1 not available ({e}), falling back to GrabCut segmentation")
                segmentation = SAMLikeSegmentationService()

        # ---- Embeddings (Instance + Semantic) ----
        # Use multi-scale embedding for better robustness
        if settings.enable_multiscale_embedding:
            # Parse embedding scales from config
            try:
                scales = [int(s.strip()) for s in settings.embedding_scales.split(",")]
            except Exception:
                scales = [224, 384, 512]  # Default scales
            
            embedding = MultiScaleEmbeddingService(
                instance_model_name=settings.instance_model_name,
                semantic_model_name=settings.semantic_model_name,
                target_dim=settings.pinecone_dim,
                scales=scales,
                enable_rotation_aug=settings.enable_rotation_augmentation,
            )
            logger.info(f"Using multi-scale embedding with scales: {scales}")
        else:
            # Standard single-scale embedding
            embedding = HFEmbeddingService(
                instance_model_name=settings.instance_model_name,
                semantic_model_name=settings.semantic_model_name,
                target_dim=settings.pinecone_dim,
            )
            logger.info("Using standard single-scale embedding")

        # ---- Attribute extraction & filtering ----
        attributes = AttributeService()

        # ---- Vector Database (Pinecone) ----
        vectors = PineconeVectorRepository(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
            dimension=settings.pinecone_dim,
        )

        # ---- Visual Cross-Encoder Reranker ----
        visual_matcher = VisualCrossEncoder()
        
        # Use multi-stage reranking for large catalogs
        if settings.enable_multistage_rerank:
            rerank = HybridRerankService(
                matcher=visual_matcher
            )
            logger.info("Using hybrid multi-stage reranking (adaptive based on catalog size)")
        else:
            # Standard single-stage reranking
            rerank = VisualCrossEncoderRerankService(
                matcher=visual_matcher
            )
            logger.info("Using standard single-stage reranking")

        # ---- Search Orchestration ----
        search_service = SearchService(
            settings=settings,
            image_io=image_io,
            preprocessing=preprocessing,
            detection=detection,
            segmentation=segmentation,
            embedding=embedding,
            attributes=attributes,
            vectors=vectors,
            rerank=rerank,
        )

        return cls(
            settings=settings,
            image_io=image_io,
            preprocessing=preprocessing,
            detection=detection,
            segmentation=segmentation,
            embedding=embedding,
            attributes=attributes,
            vectors=vectors,
            rerank=rerank,
            search_service=search_service,
        )

    # ----------------------------
    # Lifecycle Hooks
    # ----------------------------
    async def start(self) -> None:
        """
        Called on FastAPI startup.
        """
        await self.vectors.ensure_index()
        await self.embedding.load()

    async def stop(self) -> None:
        """
        Called on FastAPI shutdown.
        """
        await self.embedding.unload()
        
        # Close RunPod HTTP client if using RF-DETR detection
        if hasattr(self.detection, '_close_client'):
            await self.detection._close_client()


# ============================
# FastAPI Dependency
# ============================

def get_container(request: Request) -> Container:
    return request.app.state.container