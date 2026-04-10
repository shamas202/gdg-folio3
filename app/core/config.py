from __future__ import annotations
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    _env_file_path = Path(__file__).parent.parent.parent / ".env"

    model_config = SettingsConfigDict(
        env_file=str(_env_file_path) if _env_file_path.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="Interior Visual Search", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

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

    # YOLO11n detection confidence threshold
    detection_confidence_threshold: float = Field(
        default=0.15, alias="DETECTION_CONFIDENCE_THRESHOLD"
    )

    # Image validation
    image_min_dimension: int = Field(default=300, alias="IMAGE_MIN_DIMENSION")
    image_max_dimension: int = Field(default=4096, alias="IMAGE_MAX_DIMENSION")
    image_max_size_mb: int = Field(default=10, alias="IMAGE_MAX_SIZE_MB")

    # Image storage directories
    catalog_images_dir: str = Field(default="images/catalog", alias="CATALOG_IMAGES_DIR")
    search_images_dir: str = Field(default="images/search", alias="SEARCH_IMAGES_DIR")

    # Search settings
    search_candidate_multiplier: int = Field(default=15, alias="SEARCH_CANDIDATE_MULTIPLIER")
    max_candidate_k: int = Field(default=5000, alias="MAX_CANDIDATE_K")
