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
from app.services.embedding import EmbeddingService
from app.services.embedding_gemini import GeminiEmbeddingService
from app.services.attributes import AttributeService

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

        preprocessing = PreprocessingService()

        # ---- Detection (Furniture/Products) ----
        detection_mode = settings.detection_mode.lower()

        if detection_mode == "runpod":
            # Use RF-DETR HTTP endpoint (local GPU server or RunPod)
            detection = RFDETRDetectionService(
                api_url=settings.runpod_api_url,
                status_url=settings.runpod_status_url,
                api_key=settings.runpod_api_key,
                confidence_threshold=settings.runpod_confidence_threshold,
                max_wait_seconds=settings.runpod_max_wait_seconds,
            )
            logger.info(f"Using RF-DETR endpoint for detection: {settings.runpod_api_url}")
        else:
            # Use local RT-DETR .pt file via ultralytics
            try:
                detection = RTDETRDetectionService(model_path=settings.rtdetr_model_path)
                logger.info("Using local RT-DETR for detection")
            except FileNotFoundError as e:
                logger.error(f"Local RT-DETR model not found: {e}")
                logger.warning("Falling back to RF-DETR HTTP endpoint")
                detection = RFDETRDetectionService(
                    api_url=settings.runpod_api_url,
                    status_url=settings.runpod_status_url,
                    api_key=settings.runpod_api_key,
                    confidence_threshold=settings.runpod_confidence_threshold,
                    max_wait_seconds=settings.runpod_max_wait_seconds,
                )

        # ---- Segmentation ----
        if detection_mode == "runpod":
            segmentation = PolygonSegmentationService()
            logger.info("Using polygon segmentation (RF-DETR polygon masks)")
        else:
            try:
                segmentation = SAM2SegmentationService(model_path=settings.sam2_model_path)
                logger.info("Using SAM2.1 for segmentation")
            except (ImportError, FileNotFoundError) as e:
                logger.warning(f"SAM2.1 not available ({e}), falling back to GrabCut")
                segmentation = SAMLikeSegmentationService()

        # ---- Embeddings (Gemini multimodal API) ----
        embedding = GeminiEmbeddingService(
            api_key=settings.google_api_key,
            model=settings.gemini_embedding_model,
            output_dimensionality=settings.pinecone_dim,
        )
        logger.info(
            f"Using GeminiEmbeddingService: model={settings.gemini_embedding_model}, "
            f"dim={settings.pinecone_dim}"
        )

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
            search_service=search_service,
        )

    # ----------------------------
    # Lifecycle Hooks
    # ----------------------------
    async def start(self) -> None:
        """Called on FastAPI startup."""
        await self.vectors.ensure_index()
        await self.embedding.load()

    async def stop(self) -> None:
        """Called on FastAPI shutdown."""
        await self.embedding.unload()

        # Close RF-DETR HTTP client if applicable
        if hasattr(self.detection, '_close_client'):
            await self.detection._close_client()


# ============================
# FastAPI Dependency
# ============================

def get_container(request: Request) -> Container:
    return request.app.state.container
