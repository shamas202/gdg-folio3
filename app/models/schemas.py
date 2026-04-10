from __future__ import annotations

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class SearchHit(BaseModel):
    pinecone_id: str
    score: float
    image_url: str | None = None
    product_url: str | None = None
    product_name: str | None = None   # Simplified demo metadata
    name_english: str | None = None
    name_arabic: str | None = None
    category: str | None = None
    price_amount: int | None = None
    price_unit: str | None = None
    is_active: bool | None = None
    store_id: int | None = None
    countries: list[str] | None = None
    store: str | None = None


class SearchResponse(BaseModel):
    query_category: str | None
    hits: list[SearchHit] = Field(default_factory=list)
    message: str | None = None


class CatalogUpsertResponse(BaseModel):
    pinecone_id: str
    upserted: bool
    message: str | None = None


class CatalogAddResponse(BaseModel):
    pinecone_id: str
    success: bool
    message: str


class DetectedObject(BaseModel):
    """A detected object with bounding box and category"""
    category: str
    bbox: BBox
    score: float
    object_id: int  # Index in the detection list


class DetectionResponse(BaseModel):
    """Response containing all detected objects in an image"""
    objects: list[DetectedObject] = Field(default_factory=list)
    image_width: int
    image_height: int


class SegmentedObject(BaseModel):
    """A detected object with segmentation mask"""
    category: str
    bbox: BBox
    score: float
    object_id: int
    mask_base64: str  # Base64 encoded mask image (PNG)
    mask_polygon: list[list[float]] | None = None  # Polygon points for fast search


class DetectionSegmentationResponse(BaseModel):
    """Response containing all detected and segmented objects"""
    objects: list[SegmentedObject] = Field(default_factory=list)
    image_width: int
    image_height: int