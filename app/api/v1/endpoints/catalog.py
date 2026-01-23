from __future__ import annotations

import json
from fastapi import APIRouter, Depends, File, Form, UploadFile
from loguru import logger

from app.dependencies.container import Container, get_container
from app.models.schemas import CatalogUpsertResponse
from app.core.errors import BadRequest

router = APIRouter()


@router.post("/catalog/upsert", response_model=CatalogUpsertResponse)
async def upsert_catalog_item(
    pinecone_id: str = Form(...),
    assigned_category: str = Form(...),
    image: UploadFile = File(...),
    metadata_json: str = Form(...),
    container: Container = Depends(get_container),
) -> CatalogUpsertResponse:
    """
    Upsert a product to the catalog with new data structure.
    
    Uses only AI detection (no manual bounding boxes).
    Category is taken from assigned_category field.
    
    Args:
        pinecone_id: Unique product identifier (used as Pinecone vector ID)
        assigned_category: Product category for detection and Pinecone namespace
        image: Product image file
        metadata_json: JSON string containing all product metadata
    """
    # Normalize category to lowercase for consistency
    normalized_category = assigned_category.lower().strip()
    
    # Parse metadata JSON
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as e:
        raise BadRequest(f"Invalid metadata_json: {str(e)}")
    
    # Validate metadata is a dict
    if not isinstance(metadata, dict):
        raise BadRequest("metadata_json must be a JSON object")
    
    logger.info(f"Catalog upsert: ID={pinecone_id}, category={normalized_category}")
    
    svc = container.search_service
    return await svc.upsert_catalog_image(
        pinecone_id=pinecone_id,
        assigned_category=normalized_category,
        image_file=image,
        metadata=metadata,
    )