from __future__ import annotations
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Load .env file from backend directory (parent of app/core)
    _env_file_path = Path(__file__).parent.parent.parent / ".env"
    
    model_config = SettingsConfigDict(
        env_file=str(_env_file_path) if _env_file_path.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    app_name: str = Field(default="Interior Visual Search", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Test mode: Skip embedding and Pinecone upsert (for pipeline testing)
    test_mode: bool = Field(default=False, alias="TEST_MODE")

    # Pinecone
    pinecone_api_key: str = Field(alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="interior-products-gemini", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")
    pinecone_dim: int = Field(default=3072, alias="PINECONE_DIM")

    # Gemini Embedding
    google_api_key: str = Field(alias="GOOGLE_API_KEY")
    gemini_embedding_model: str = Field(
        default="models/gemini-embedding-2-preview",
        alias="GEMINI_EMBEDDING_MODEL",
    )

    # Pinecone reranking (optional)
    enable_pinecone_rerank: bool = Field(default=False, alias="ENABLE_PINECONE_RERANK")
    pinecone_rerank_model: str = Field(default="bge-reranker-v2-m3", alias="PINECONE_RERANK_MODEL")

    # Local RT-DETR Detection Model (fallback)
    rtdetr_model_path: str = Field(
        default=r"D:\image_image_search\backend\app\models\rtdetr-x.pt",
        alias="RTDETR_MODEL_PATH"
    )

    # SAM2.1 Segmentation Model (Ultralytics)
    sam2_model_path: str = Field(
        default=r"D:\image_image_search\backend\app\models\sam2.1_l.pt",
        alias="SAM2_MODEL_PATH"
    )

    # RF-DETR HTTP endpoint (local GPU server or RunPod)
    runpod_api_url: str = Field(
        default="http://localhost:8000/predict",
        alias="RUNPOD_API_URL"
    )
    runpod_status_url: str = Field(default="", alias="RUNPOD_STATUS_URL")
    runpod_api_key: str = Field(default="", alias="RUNPOD_API_KEY")
    runpod_confidence_threshold: float = Field(default=0.25, alias="RUNPOD_CONFIDENCE_THRESHOLD")
    runpod_max_wait_seconds: int = Field(default=60, alias="RUNPOD_MAX_WAIT_SECONDS")

    # Detection mode: "runpod" (local RF-DETR server) or "local" (RT-DETR .pt file)
    detection_mode: str = Field(default="runpod", alias="DETECTION_MODE")

    # Image validation
    image_min_dimension: int = Field(default=500, alias="IMAGE_MIN_DIMENSION")
    image_max_dimension: int = Field(default=4096, alias="IMAGE_MAX_DIMENSION")
    image_max_size_mb: int = Field(default=15, alias="IMAGE_MAX_SIZE_MB")

    # Image storage directories
    catalog_images_dir: str = Field(default="images/catalog", alias="CATALOG_IMAGES_DIR")
    search_images_dir: str = Field(default="images/search", alias="SEARCH_IMAGES_DIR")

    # Search settings
    search_candidate_multiplier: int = Field(default=15, alias="SEARCH_CANDIDATE_MULTIPLIER")
    search_deduplicate_skus: bool = Field(default=True, alias="SEARCH_DEDUPLICATE_SKUS")
    max_candidate_k: int = Field(default=5000, alias="MAX_CANDIDATE_K")

