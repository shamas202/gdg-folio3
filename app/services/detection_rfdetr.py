from __future__ import annotations

from dataclasses import dataclass
import asyncio
import time
import base64
from io import BytesIO
from typing import Optional

import httpx
from PIL import Image
from loguru import logger

from app.models.domain import Detection


@dataclass
class RFDETRDetectionService:
    """
    RF-DETR detection via RunPod API (fully async for maximum throughput).
    
    Provides both detection and segmentation in a single API call.
    Returns bounding boxes and polygon masks for detected objects.
    
    Uses httpx.AsyncClient for non-blocking async operations, allowing
    multiple RunPod jobs to run concurrently without blocking the event loop.
    """
    api_url: str
    status_url: str
    api_key: str
    confidence_threshold: float = 0.10
    max_wait_seconds: int = 60
    
    def __post_init__(self):
        """Initialize shared async HTTP client for connection pooling."""
        # Client will be created on first use and reused for better performance
        # Not a dataclass field to avoid serialization issues
        self._client: Optional[httpx.AsyncClient] = None
    
    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string for API"""
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create shared async HTTP client for connection pooling."""
        if self._client is None:
            # Create client with connection pooling and reasonable timeouts
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                http2=False,  # RunPod API uses HTTP/1.1
            )
        return self._client
    
    async def _close_client(self) -> None:
        """Close the shared HTTP client (call on shutdown)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def submit_job(self, img: Image.Image) -> Optional[str]:
        """
        Submit a RunPod job and return job_id immediately (non-blocking).
        This allows rapid submission of multiple jobs without waiting.
        
        Args:
            img: Input image (PIL Image)
        
        Returns:
            Job ID if successful, None otherwise
        """
        client = await self._get_client()
        
        try:
            image_b64 = self._image_to_base64(img)
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
            
            response = await client.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            job_id = result.get("id")
            
            if job_id:
                logger.debug(f"RF-DETR job submitted: {job_id}")
            return job_id
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to submit RF-DETR job (HTTP {e.response.status_code}): {e}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Failed to submit RF-DETR job (request error): {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to submit RF-DETR job (unexpected error): {e}")
            return None
    
    async def poll_job_status(self, job_id: str) -> Optional[dict]:
        """
        Poll a single job's status (non-blocking).
        
        Args:
            job_id: RunPod job ID
        
        Returns:
            Status data dict or None if failed
        """
        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            status_url = f"{self.status_url}/{job_id}"
            response = await client.get(status_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.debug(f"Failed to poll job {job_id} (HTTP {e.response.status_code}): {e}")
            return None
        except httpx.RequestError as e:
            logger.debug(f"Failed to poll job {job_id} (request error): {e}")
            return None
        except Exception as e:
            logger.debug(f"Failed to poll job {job_id} (unexpected error): {e}")
            return None
    
    async def wait_for_job(self, job_id: str) -> Optional[dict]:
        """
        Wait for a job to complete with efficient concurrent polling.
        
        Args:
            job_id: RunPod job ID
        
        Returns:
            Completed job data dict or None if failed/timed out
        """
        start_time = time.time()
        poll_count = 0
        poll_interval = 0.3  # Start with faster polling (0.3s instead of 0.5s)
        
        while time.time() - start_time < self.max_wait_seconds:
            poll_count += 1
            status_data = await self.poll_job_status(job_id)
            
            if not status_data:
                # If polling failed, wait a bit and retry
                await asyncio.sleep(poll_interval)
                continue
                
            status = status_data.get("status")
            
            if status == "COMPLETED":
                elapsed = time.time() - start_time
                logger.debug(f"Job {job_id} completed in {elapsed:.2f}s ({poll_count} polls)")
                return status_data
            elif status == "FAILED":
                error = status_data.get("error", "Unknown error")
                logger.error(f"Job {job_id} failed: {error}")
                return None
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                await asyncio.sleep(poll_interval)
                # Adaptive polling: increase interval if job is taking longer
                if poll_count > 5:
                    poll_interval = min(0.8, poll_interval * 1.1)  # Max 0.8s interval
            else:
                logger.warning(f"Job {job_id} unknown status: {status}")
                await asyncio.sleep(poll_interval)
        
        logger.error(f"Job {job_id} timeout after {self.max_wait_seconds}s")
        return None
    
    async def detect_batch(
        self, 
        images: list[Image.Image], 
        category_hints: list[str | None] | None = None
    ) -> list[list[Detection]]:
        """
        Submit and process multiple jobs concurrently for maximum RunPod utilization.
        
        This is the key optimization: 
        1. Submit all jobs rapidly (no waiting between submissions)
        2. Poll all jobs concurrently (maximum parallelism)
        3. Parse results in parallel
        
        Args:
            images: List of images to process
            category_hints: Optional list of category hints (one per image)
        
        Returns:
            List of detection lists (one per image)
        """
        if category_hints is None:
            category_hints = [None] * len(images)
        
        if len(images) != len(category_hints):
            raise ValueError(f"Images ({len(images)}) and category_hints ({len(category_hints)}) must have same length")
        
        if not images:
            return []
        
        # Phase 1: Submit all jobs rapidly (no waiting between submissions)
        logger.info(f"Submitting {len(images)} RF-DETR jobs concurrently...")
        submit_tasks = [self.submit_job(img) for img in images]
        job_ids = await asyncio.gather(*submit_tasks, return_exceptions=True)
        
        # Filter out failed submissions
        valid_jobs = []
        for i, (job_id, img, hint) in enumerate(zip(job_ids, images, category_hints)):
            if isinstance(job_id, Exception) or job_id is None:
                logger.warning(f"Failed to submit job for image {i}")
                continue
            valid_jobs.append((job_id, img, hint))
        
        logger.info(f"Successfully submitted {len(valid_jobs)}/{len(images)} jobs")
        
        if not valid_jobs:
            return [[]] * len(images)
        
        # Phase 2: Poll all jobs concurrently
        logger.info(f"Polling {len(valid_jobs)} jobs concurrently...")
        poll_tasks = [self.wait_for_job(job_id) for job_id, _, _ in valid_jobs]
        results = await asyncio.gather(*poll_tasks, return_exceptions=True)
        
        # Phase 3: Parse results and map back to original images
        all_detections = []
        result_idx = 0
        
        for i, (job_id, hint) in enumerate(zip(job_ids, category_hints)):
            if isinstance(job_id, Exception) or job_id is None:
                # Failed submission - no result to process
                all_detections.append([])
            else:
                # Get result for this job (results array corresponds to valid_jobs)
                if result_idx < len(results):
                    result = results[result_idx]
                    result_idx += 1
                    
                    if isinstance(result, Exception) or result is None:
                        logger.warning(f"Job {job_id} failed or timed out")
                        all_detections.append([])
                        continue
                    
                    output = result.get("output", {})
                    detections = self._parse_detections(output, hint)
                    all_detections.append(detections)
                else:
                    # Should not happen, but handle gracefully
                    logger.error(f"Result index mismatch for job {job_id}")
                    all_detections.append([])
        
        completed = sum(1 for dets in all_detections if dets)
        logger.info(f"Batch processing complete: {completed}/{len(images)} jobs succeeded")
        
        return all_detections
    
    async def detect(
        self, 
        img: Image.Image, 
        category_hint: str | None = None
    ) -> list[Detection]:
        """
        Detect objects using RF-DETR via RunPod API (fully async, non-blocking).
        
        Args:
            img: Input image (PIL Image)
            category_hint: Optional category filter (e.g., "bed", "sofa")
        
        Returns:
            List of Detection objects sorted by confidence
        """
        client = await self._get_client()
        
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
            
            # Submit job (async, non-blocking)
            logger.info(f"Submitting RF-DETR detection job (category_hint: {category_hint})")
            response = await client.post(
                self.api_url, 
                json=payload, 
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            job_id = result.get("id")
            
            if not job_id:
                logger.error("RF-DETR API: No job ID returned")
                return []
            
            logger.debug(f"RF-DETR job submitted: {job_id}")
            
            # Poll for results (async, non-blocking)
            # Use optimized polling with faster initial interval
            status_url = f"{self.status_url}/{job_id}"
            start_time = time.time()
            poll_count = 0
            poll_interval = 0.3  # Start with faster polling (0.3s instead of 0.5s)
            
            while time.time() - start_time < self.max_wait_seconds:
                poll_count += 1
                
                # Async status check (non-blocking)
                status_response = await client.get(status_url, headers=headers)
                status_response.raise_for_status()
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
                    # Still processing, wait and retry (async sleep - non-blocking)
                    await asyncio.sleep(poll_interval)
                    # Adaptive polling: increase interval if job is taking longer
                    if poll_count > 5:
                        poll_interval = min(0.8, poll_interval * 1.1)  # Max 0.8s interval
                else:
                    logger.warning(f"RF-DETR API unknown status: {status}")
                    await asyncio.sleep(poll_interval)
            
            logger.error(f"RF-DETR API timeout after {self.max_wait_seconds}s")
            return []
            
        except httpx.HTTPStatusError as e:
            logger.error(f"RF-DETR API HTTP error {e.response.status_code}: {e}")
            return []
        except httpx.RequestError as e:
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
