from __future__ import annotations

from dataclasses import dataclass
import time
import base64
from io import BytesIO

import requests
from PIL import Image
from loguru import logger

from app.models.domain import Detection


@dataclass
class RFDETRDetectionService:
    """
    RF-DETR detection via RunPod API.
    
    Provides both detection and segmentation in a single API call.
    Returns bounding boxes and polygon masks for detected objects.
    """
    api_url: str
    status_url: str
    api_key: str
    confidence_threshold: float = 0.10
    max_wait_seconds: int = 60
    
    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string for API"""
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    async def detect(
        self, 
        img: Image.Image, 
        category_hint: str | None = None
    ) -> list[Detection]:
        """
        Detect objects using RF-DETR via RunPod API.
        
        Args:
            img: Input image (PIL Image)
            category_hint: Optional category filter (e.g., "bed", "sofa")
        
        Returns:
            List of Detection objects sorted by confidence
        """
        try:
            # Convert image to base64
            logger.debug("Converting image to base64 for RF-DETR API")
            image_b64 = self._image_to_base64(img)
            
            # Prepare request
            payload = {
                "input": {
                    "image": image_b64,
                    "threshold": self.confidence_threshold
                }
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Submit job
            logger.info(f"Submitting RF-DETR detection job (category_hint: {category_hint})")
            response = requests.post(
                self.api_url, 
                json=payload, 
                headers=headers, 
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            job_id = result.get("id")
            
            if not job_id:
                logger.error("RF-DETR API: No job ID returned")
                return []
            
            logger.debug(f"RF-DETR job submitted: {job_id}")
            
            # Poll for results
            status_url = f"{self.status_url}/{job_id}"
            start_time = time.time()
            poll_count = 0
            
            while time.time() - start_time < self.max_wait_seconds:
                poll_count += 1
                status_response = requests.get(status_url, headers=headers, timeout=30)
                status_data = status_response.json()
                status = status_data.get("status")
                
                if status == "COMPLETED":
                    elapsed = time.time() - start_time
                    logger.info(f"RF-DETR job completed in {elapsed:.2f}s ({poll_count} polls)")
                    
                    output = status_data.get("output", {})
                    detections = self._parse_detections(output, category_hint)
                    logger.info(f"RF-DETR detected {len(detections)} objects")
                    return detections
                
                elif status == "FAILED":
                    error = status_data.get("error", "Unknown error")
                    logger.error(f"RF-DETR API job failed: {error}")
                    return []
                
                elif status == "IN_QUEUE" or status == "IN_PROGRESS":
                    # Still processing, wait and retry
                    time.sleep(0.5)
                else:
                    logger.warning(f"RF-DETR API unknown status: {status}")
                    time.sleep(0.5)
            
            logger.error(f"RF-DETR API timeout after {self.max_wait_seconds}s")
            return []
            
        except requests.exceptions.RequestException as e:
            logger.error(f"RF-DETR API request error: {e}")
            return []
        except Exception as e:
            logger.error(f"RF-DETR API unexpected error: {e}")
            return []
    
    def _parse_detections(
        self, 
        output: dict, 
        category_hint: str | None
    ) -> list[Detection]:
        """
        Parse RF-DETR API response into Detection objects.
        
        API Response Format:
        {
            "result": {
                "detected_objects": [
                    {
                        "label": "bed",
                        "confidence": 0.95,
                        "bbox_from_mask": [x1, y1, x2, y2],
                        "mask_polygon": [[x1, y1], [x2, y2], ...]
                    }
                ]
            }
        }
        """
        detections = []
        
        result = output.get("result", {})
        detected_objects = result.get("detected_objects", [])
        
        logger.debug(f"Parsing {len(detected_objects)} detected objects from API")
        
        for det in detected_objects:
            conf = det.get("confidence", 0)
            class_name = det.get("label", "unknown")
            bbox = det.get("bbox_from_mask", [0, 0, 0, 0])
            mask_polygon = det.get("mask_polygon", None)
            
            # Skip low-confidence detections
            if conf < self.confidence_threshold:
                continue
            
            # Validate bbox
            if len(bbox) != 4:
                logger.warning(f"Invalid bbox format: {bbox}")
                continue
            
            x1, y1, x2, y2 = bbox
            
            # Create Detection object with mask_polygon
            detection = Detection(
                category=class_name.lower().strip(),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                score=float(conf),
                mask_polygon=mask_polygon
            )
            
            detections.append(detection)
        
        # Filter by category_hint if provided
        if category_hint:
            hint_lower = category_hint.lower().strip()
            matching = [d for d in detections if d.category == hint_lower]
            
            if matching:
                logger.info(
                    f"Filtered {len(matching)}/{len(detections)} detections "
                    f"matching category '{category_hint}'"
                )
                detections = matching
            else:
                available_cats = list(set(d.category for d in detections))
                logger.warning(
                    f"No detection matching category '{category_hint}'. "
                    f"Available: {available_cats}"
                )
        
        # Sort by confidence (descending)
        detections.sort(key=lambda d: d.score, reverse=True)
        
        if detections:
            best = detections[0]
            logger.info(
                f"Best detection: {best.category} "
                f"(score: {best.score:.3f}, bbox: {best.bbox})"
            )
        
        return detections
