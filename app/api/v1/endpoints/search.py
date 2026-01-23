from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from loguru import logger

from app.dependencies.container import Container, get_container
from app.models.schemas import BBox, SearchResponse

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    image: UploadFile = File(...),
    assigned_category: str | None = Form(default=None),
    top_k: int = Form(default=50),
    # User-selected bbox and mask (from object detection view)
    bbox_x1: int | None = Form(default=None),
    bbox_y1: int | None = Form(default=None),
    bbox_x2: int | None = Form(default=None),
    bbox_y2: int | None = Form(default=None),
    mask_polygon: str | None = Form(default=None),  # JSON string: "[[x1,y1],[x2,y2],...]"
    container: Container = Depends(get_container),
) -> SearchResponse:
    from app.core.errors import BadRequest
    from app.models.schemas import BBox
    import json
    
    # Validate image file
    if not image.filename:
        raise BadRequest("Image file is required")
    
    # Validate top_k
    if top_k <= 0:
        raise BadRequest(f"top_k must be greater than 0, got {top_k}")
    if top_k > 100:
        raise BadRequest(f"top_k cannot exceed 100, got {top_k}")

    # Normalize empty category string to None
    if assigned_category is not None and assigned_category.strip() == "":
        assigned_category = None
    
    svc = container.search_service
    
    # Check if bbox provided (user clicked specific object from detection view)
    has_bbox = (bbox_x1 is not None and bbox_y1 is not None and 
                bbox_x2 is not None and bbox_y2 is not None)
    
    if has_bbox:
        # FAST PATH: User selected specific object, skip detection
        logger.info(
            f"Search request with bbox - category: {assigned_category}, "
            f"bbox: ({bbox_x1},{bbox_y1},{bbox_x2},{bbox_y2}), top_k: {top_k}"
        )
        
        bbox = BBox(x1=bbox_x1, y1=bbox_y1, x2=bbox_x2, y2=bbox_y2)
        
        # Parse mask polygon if provided
        mask_polygon_list = None
        if mask_polygon:
            try:
                mask_polygon_list = json.loads(mask_polygon)
            except json.JSONDecodeError:
                logger.warning("Invalid mask_polygon JSON, proceeding without mask")
        
        result = await svc.search_with_bbox(
            room_image_file=image,
            bbox=bbox,
            mask_polygon=mask_polygon_list,
            category=assigned_category,
            top_k=top_k,
        )
    else:
        # FULL PATH: No bbox provided, run full detection
        logger.info(
            f"Search request (full detection) - category: {assigned_category}, "
            f"top_k: {top_k}, image_filename: {image.filename}"
        )
        
        result = await svc.search_room_image(
            room_image_file=image,
            assigned_category=assigned_category,
            top_k=top_k,
        )
    
    return result