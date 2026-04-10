from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from fastapi import Request

from app.core.config import Settings
from app.repositories.pinecone_repo import PineconeVectorRepository
from app.services.image_io import ImageIOService
from app.services.preprocessing import PreprocessingService
from app.services.detection import DetectionService
from app.services.segmentation import SegmentationService, BBoxSegmentationService
from app.services.embedding import EmbeddingService
from app.services.embedding_gemini import GeminiEmbeddingService
from app.services.search_service import SearchService


@dataclass(frozen=True)
class Container:
    settings: Settings
    image_io: ImageIOService
    preprocessing: PreprocessingService
    detection: DetectionService
    segmentation: SegmentationService
    embedding: EmbeddingService
    vectors: PineconeVectorRepository
    search_service: SearchService

    @classmethod
    def from_settings(cls, settings: Settings) -> "Container":
        from loguru import logger

        image_io = ImageIOService(
            catalog_images_dir=Path(settings.catalog_images_dir),
            search_images_dir=Path(settings.search_images_dir),
            min_dimension=settings.image_min_dimension,
            max_dimension=settings.image_max_dimension,
            max_file_size_mb=settings.image_max_size_mb,
        )

        preprocessing = PreprocessingService()

        from app.services.detection_yolo import YOLODetectionService
        detection = YOLODetectionService(
            confidence_threshold=settings.detection_confidence_threshold,
        )
        logger.info(
            f"Using YOLO11n for detection "
            f"(conf={settings.detection_confidence_threshold})"
        )

        segmentation = BBoxSegmentationService()

        embedding = GeminiEmbeddingService(
            api_key=settings.google_api_key,
            model=settings.gemini_embedding_model,
            output_dimensionality=settings.pinecone_dim,
        )
        logger.info(
            f"Using GeminiEmbeddingService: model={settings.gemini_embedding_model}, "
            f"dim={settings.pinecone_dim}"
        )

        vectors = PineconeVectorRepository(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
            dimension=settings.pinecone_dim,
        )

        search_service = SearchService(
            settings=settings,
            image_io=image_io,
            preprocessing=preprocessing,
            detection=detection,
            embedding=embedding,
            vectors=vectors,
        )

        return cls(
            settings=settings,
            image_io=image_io,
            preprocessing=preprocessing,
            detection=detection,
            segmentation=segmentation,
            embedding=embedding,
            vectors=vectors,
            search_service=search_service,
        )

    async def start(self) -> None:
        await self.vectors.ensure_index()
        await self.embedding.load()

    async def stop(self) -> None:
        await self.embedding.unload()


def get_container(request: Request) -> Container:
    return request.app.state.container
