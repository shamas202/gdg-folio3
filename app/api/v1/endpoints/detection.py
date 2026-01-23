from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile
from loguru import logger

from app.dependencies.container import Container, get_container
from app.models.schemas import DetectionResponse, DetectionSegmentationResponse, DetectedObject, SegmentedObject, BBox
from app.core.errors import BadRequest

router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    image: UploadFile = File(...),
    container: Container = Depends(get_container),
) -> DetectionResponse:
    """
    Detect all objects in an uploaded image.
    Returns bounding boxes and categories for all detected objects.
    """
    from app.utils.timing import timed
    
    # Validate image file
    if not image.filename:
        raise BadRequest("Image file is required")
    
    logger.info(f"Detection request for image: {image.filename}")
    
    with timed("Image load"):
        img = await container.search_service.image_io.read_upload_as_rgb(image)
    
    w, h = img.size
    logger.debug(f"Image size: {w}x{h}")
    
    # Run detection (no category hint - detect all objects)
    with timed("Detection"):
        detections = await container.detection.detect(img, category_hint=None)
    
    logger.info(f"Detected {len(detections)} objects")
    
    # Convert to response format
    objects = []
    for idx, det in enumerate(detections):
        objects.append(
            DetectedObject(
                category=det.category,
                bbox=BBox(x1=det.bbox[0], y1=det.bbox[1], x2=det.bbox[2], y2=det.bbox[3]),
                score=det.score,
                object_id=idx,
            )
        )
    
    return DetectionResponse(
        objects=objects,
        image_width=w,
        image_height=h,
    )


@router.post("/detect-and-segment", response_model=DetectionSegmentationResponse)
async def detect_and_segment_objects(
    image: UploadFile = File(...),
    container: Container = Depends(get_container),
) -> DetectionSegmentationResponse:
    """
    Detect all objects in an uploaded image and segment them using SAM 2.1.
    Returns bounding boxes, categories, and segmentation masks for all detected objects.
    """
    from app.utils.timing import timed
    import base64
    import io
    import numpy as np
    from PIL import Image
    
    # Validate image file
    if not image.filename:
        raise BadRequest("Image file is required")
    
    logger.info(f"Detection and segmentation request for image: {image.filename}")
    
    with timed("Image load"):
        img = await container.image_io.read_upload_as_rgb(image)
    
    w, h = img.size
    logger.debug(f"Image size: {w}x{h}")
    
    # Run detection (no category hint - detect all objects)
    with timed("Detection"):
        detections = await container.detection.detect(img, category_hint=None)
    
    logger.info(f"Detected {len(detections)} objects")
    
    if not detections:
        logger.warning("No objects detected")
        return DetectionSegmentationResponse(
            objects=[],
            image_width=w,
            image_height=h,
        )
    
    # Segment each detected object
    segmented_objects = []
    for idx, det in enumerate(detections):
        try:
            bbox = BBox(x1=det.bbox[0], y1=det.bbox[1], x2=det.bbox[2], y2=det.bbox[3])
            
            # Clamp bbox to image bounds
            try:
                bbox = container.preprocessing.clamp_bbox(bbox, w, h)
            except ValueError as e:
                logger.warning(f"Invalid bbox for object {idx}: {e}, skipping")
                continue
            
            # Extract polygon mask if available (from RF-DETR API)
            mask_polygon = getattr(det, 'mask_polygon', None)
            
            # Run segmentation with polygon if available
            with timed(f"Segmentation {idx}"):
                if hasattr(container.segmentation, 'segment') and mask_polygon:
                    try:
                        segment = await container.segmentation.segment(img, bbox, mask_polygon=mask_polygon)
                        logger.debug(f"Used polygon segmentation for object {idx}")
                    except TypeError:
                        # Fallback if segmentation doesn't support polygon parameter
                        segment = await container.segmentation.segment(img, bbox)
                        logger.debug(f"Fallback to SAM2/GrabCut for object {idx}")
                else:
                    segment = await container.segmentation.segment(img, bbox)
            
            # Convert mask to base64 PNG
            # Create a binary mask image (0 = background, 255 = foreground)
            mask_array = segment.mask.astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_array, mode='L')
            
            # Convert to base64
            buffer = io.BytesIO()
            mask_img.save(buffer, format='PNG')
            mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            segmented_objects.append(
                SegmentedObject(
                    category=det.category,
                    bbox=bbox,
                    score=det.score,
                    object_id=idx,
                    mask_base64=mask_base64,
                    mask_polygon=det.mask_polygon,  # Include polygon for fast search
                )
            )
            
            logger.debug(f"Segmented object {idx}: {det.category} (score: {det.score:.3f})")
            
        except Exception as e:
            logger.error(f"Error segmenting object {idx}: {e}")
            # Continue with other objects even if one fails
            continue
    
    logger.info(f"Successfully segmented {len(segmented_objects)} objects")
    
    return DetectionSegmentationResponse(
        objects=segmented_objects,
        image_width=w,
        image_height=h,
    )

