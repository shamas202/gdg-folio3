from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4
import json

from fastapi import UploadFile

from app.core.errors import BadRequest
from app.core.constants import SMALL_CATEGORIES, MEDIUM_CATEGORIES, LARGE_CATEGORIES, ROA_THRESHOLDS
from app.core.config import Settings
from app.models.schemas import BBox, CatalogUpsertResponse, SearchHit, SearchResponse
from app.services.image_io import ImageIOService
from app.services.preprocessing import PreprocessingService
from app.services.detection import DetectionService
from app.services.segmentation import SegmentationService
from app.services.embedding import EmbeddingService
from app.services.attributes import AttributeService
from app.services.rerank import RerankService
from app.repositories.pinecone_repo import PineconeVectorRepository


@dataclass
class SearchService:
    settings: Settings
    image_io: ImageIOService
    preprocessing: PreprocessingService
    detection: DetectionService
    segmentation: SegmentationService
    embedding: EmbeddingService
    attributes: AttributeService
    vectors: PineconeVectorRepository
    rerank: RerankService

    async def upsert_catalog_image(
        self,
        *,
         pinecone_id: str,
        assigned_category: str,
        image_file: UploadFile,
        metadata: dict[str, Any],
    ) -> CatalogUpsertResponse:
        from loguru import logger
        from app.utils.timing import timed
        
        logger.info(f"Upserting catalog item: ID={pinecone_id}, category={assigned_category}")
        
        # Stage 1: Image quality validation (with category for strict checks)
        with timed("Image load"):
            img = await self.image_io.read_upload_as_rgb(
                image_file,
                category=assigned_category  # Category-specific blur validation
            )
        
        w, h = img.size
        image_area = w * h
        logger.debug(f"Catalog image size: {w}x{h}, area: {image_area:,} pixels")

        # Run detection using assigned_category as hint
        mask_polygon = None
        
        with timed("Detection"):
            detections = await self.detection.detect(img, category_hint=assigned_category)
        
        logger.info(f"Detected {len(detections)} objects")
        
        # STRICT: Reject if no detection (no fallback, no manual bbox)
        if not detections:
            raise BadRequest(
                f"No objects detected in image for category '{assigned_category}'. "
                f"Cannot ingest product without detection. "
                f"Please ensure the image contains a clear view of the product with good lighting."
            )
        
        # Filter detections to only include those matching assigned_category
        matching_detections = [d for d in detections if d.category == assigned_category]
        
        if not matching_detections:
            # No detection matches the assigned category
            available_categories = list(set(d.category for d in detections))
            raise BadRequest(
                f"No '{assigned_category}' detected in image. "
                f"RF-DETR found: {available_categories}. "
                f"Please verify the 'assigned_category' in your CSV matches the actual product."
            )
        
        # Select detection with LARGEST BBOX AREA from matching detections
        det = max(
            matching_detections,
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
        )
        
        if len(matching_detections) > 1:
            logger.info(
                f"📦 Multiple '{assigned_category}' detected ({len(matching_detections)}). "
                f"Selected LARGEST: {(det.bbox[2]-det.bbox[0]):.0f}×{(det.bbox[3]-det.bbox[1]):.0f}px "
                f"(area: {(det.bbox[2]-det.bbox[0])*(det.bbox[3]-det.bbox[1]):,.0f}px, score: {det.score:.3f})"
            )
        else:
            logger.debug(
                f"Single '{assigned_category}' detected: "
                f"{(det.bbox[2]-det.bbox[0]):.0f}×{(det.bbox[3]-det.bbox[1]):.0f}px "
                f"(score: {det.score:.3f})"
            )
        
        bbox = BBox(x1=det.bbox[0], y1=det.bbox[1], x2=det.bbox[2], y2=det.bbox[3])
        mask_polygon = getattr(det, 'mask_polygon', None)
        
        logger.info(
            f"Using detected bbox: {bbox} "
            f"(category: {det.category}, score: {det.score:.3f})"
        )
        
        # === STAGE 2: ROA (Ratio of Area) VALIDATION ===
        # Only for ingestion - ensures proper product framing
        bbox_width = bbox.x2 - bbox.x1
        bbox_height = bbox.y2 - bbox.y1
        bbox_area = bbox_width * bbox_height
        roa = bbox_area / image_area
        
        logger.info(
            f"Object size: {bbox_width:.0f}×{bbox_height:.0f}px, "
            f"ROA: {roa:.1%} (bbox area / image area)"
        )
        
        # Category-based ROA thresholds (3-tier system)
        category_lower = assigned_category.lower().strip()
        
        if category_lower in SMALL_CATEGORIES:
            min_roa = ROA_THRESHOLDS["small"]["min"]  # 0.06 (6%)
            cat_size = "small"
        elif category_lower in MEDIUM_CATEGORIES:
            min_roa = ROA_THRESHOLDS["medium"]["min"]  # 0.08 (8%)
            cat_size = "medium"
        elif category_lower in LARGE_CATEGORIES:
            min_roa = ROA_THRESHOLDS["large"]["min"]  # 0.12 (12%)
            cat_size = "large"
        else:
            # Should never happen (get_category_requirements would have failed)
            raise BadRequest(f"Unknown category '{assigned_category}'")
        
        # Validate ROA - reject if object too small/distant
        if roa < min_roa:
            raise BadRequest(
                f"Object too small or distant in image (ROA: {roa:.1%}). "
                f"The {assigned_category} occupies only {roa:.1%} of the image. "
                f"Minimum required for {cat_size} objects: {min_roa:.0%}. "
                f"Please take a closer photo with the product as the main subject."
            )
        
        logger.info(f"✓ ROA validation passed: {roa:.1%} ≥ {min_roa:.0%} for {cat_size} objects")
        
        if not mask_polygon:
            raise BadRequest(
                "No polygon mask from detection. Polygon-based segmentation is required."
            )
        
        logger.info(f"Polygon mask available with {len(mask_polygon)} points")
        
        # Segment the product using polygon from RF-DETR (no fallback)
        with timed("Segmentation"):
            segment = await self.segmentation.segment(img, bbox, mask_polygon=mask_polygon)
        
        with timed("Crop generation and Masking"):
            # Generate ALL base crops (tight, medium, full) and their bboxes
            base_crops, base_bboxes = self.preprocessing.crop_base(img, bbox)
            
            # === CATEGORY-BASED CROP SELECTION (3-TIER) ===
            if category_lower in SMALL_CATEGORIES:
                # Small objects: tight only (background is noise)
                crops_to_process = ["tight"]
                logger.info(
                    f"📦 Small object '{assigned_category}': Generating TIGHT crop only "
                    f"(4 crops, background is noise)"
                )
            elif category_lower in MEDIUM_CATEGORIES:
                # Medium objects: tight only (background mostly noise)
                crops_to_process = ["tight"]
                logger.info(
                    f"📦 Medium object '{assigned_category}': Generating TIGHT crop only "
                    f"(4 crops, standard product shot)"
                )
            elif category_lower in LARGE_CATEGORIES:
                # Large objects: all crops (context helps matching)
                crops_to_process = ["tight", "medium", "full"]
                logger.info(
                    f"📦 Large object '{assigned_category}': Generating ALL crops "
                    f"(9 crops: tight + medium + full, lifestyle context)"
                )
            else:
                # Should never happen (already validated in ROA section)
                raise BadRequest(f"Unknown category '{assigned_category}'")
            
            # Filter crops to only those needed for this category
            filtered_crops = {k: v for k, v in base_crops.items() if k in crops_to_process}
            filtered_bboxes = {k: v for k, v in base_bboxes.items() if k in crops_to_process}
            
            # === MASKING: TIGHT CROP ONLY ===
            # Always mask tight crop for pure object features
            filtered_crops["tight"] = self.preprocessing.apply_mask_on_crop(
                filtered_crops["tight"], 
                segment.mask, 
                bbox=filtered_bboxes["tight"]
            )
            # Medium and full crops remain unmasked (natural context preserved)
            
            logger.debug(
                f"Masking: tight=masked, "
                f"{'medium+full=unmasked' if len(crops_to_process) > 1 else 'no other crops'}"
            )

            # Generate augmentations (±5° rotations + horizontal flip)
            crops = self.preprocessing.add_rotated_crops(filtered_crops)
            
            logger.info(
                f"✅ Generated {len(crops)} total crops "
                f"({len(filtered_crops)} base + {len(crops) - len(filtered_crops)} augmented)"
            )

        # DEBUG: Save all crops to debug directory for first 3 items
        try:
            from pathlib import Path
            debug_dir = Path("debug_crop_image")
            debug_dir.mkdir(exist_ok=True)
            
            # Count existing items to limit to first 3
            existing_files = list(debug_dir.glob("*_tight.jpg"))
            if len(existing_files) < 3:
                import time
                timestamp = int(time.time() * 1000)
                
                for crop_name, crop_img in crops.items():
                    debug_path = debug_dir / f"{pinecone_id}_{timestamp}_{crop_name}.jpg"
                    crop_img.save(debug_path, quality=95)
                
                logger.info(f"💾 Saved {len(crops)} debug crops to {debug_dir}/ for product {pinecone_id}")
        except Exception as e:
            logger.warning(f"Failed to save debug crops (non-critical): {e}")

        # === TEST MODE: Skip embedding and Pinecone upsert ===
        if self.settings.test_mode:
            logger.info(
                f"🧪 TEST MODE: Skipping embedding and Pinecone upsert. "
                f"Validation, detection, segmentation, and cropping completed successfully."
            )
            return CatalogUpsertResponse(
                pinecone_id=pinecone_id,
                upserted=False,  # Not actually upserted
                message=f"TEST MODE: Pipeline validated successfully ({len(crops)} crops generated)"
            )
        
        # === PRODUCTION MODE: Continue with embedding and upsert ===
        with timed("Embedding"):
            vector = self.embedding.embed_crops(crops)
        
        # Use pinecone_id directly as vector_id (no UUID generation)
        vector_id = pinecone_id

        
        
        # Upsert to Pinecone using assigned_category as namespace
        with timed("Vector upsert"):
            self.vectors.upsert(
                vector_id=vector_id,
                vector=vector,
                metadata=metadata,
                namespace=assigned_category
            )
        
        logger.info(
            f"Successfully upserted: ID={pinecone_id}, namespace={assigned_category}"
        )

        return CatalogUpsertResponse(
            pinecone_id=pinecone_id,
            upserted=True,
            message=f"Successfully upserted product {pinecone_id} to namespace {assigned_category}"
        )

    async def search_with_bbox(
        self,
        *,
        room_image_file: UploadFile,
        bbox: BBox,
        mask_polygon: list | None,
        category: str | None,
        top_k: int,
    ) -> SearchResponse:
        """
        Search using user-selected bbox from object detection view.
        Skips detection phase for faster, more accurate search.
        
        This is called when user clicks a specific detected object,
        allowing us to reuse the bbox and mask from the initial detection.
        """
        from loguru import logger
        from app.utils.timing import timed
        import numpy as np
        
        logger.info(
            f"Fast search with provided bbox: category={category}, "
            f"bbox=({bbox.x1},{bbox.y1},{bbox.x2},{bbox.y2}), top_k={top_k}"
        )

        # Stage 1: Image quality validation (basic checks only)
        with timed("Image load"):
            img = await self.image_io.read_upload_as_rgb(room_image_file)
        
        w, h = img.size
        query_category = category.lower().strip() if category else None
        
        # Validate and clamp bbox
        try:
            bbox = self.preprocessing.clamp_bbox(bbox, w, h)
        except ValueError as e:
            raise BadRequest(f"Invalid bounding box: {str(e)}")
        
        logger.info(f"Using provided bbox: {bbox} (category: {query_category})")
        
        # Segmentation using provided mask or generate if not provided
        if not mask_polygon:
            logger.warning("No mask polygon provided, segmentation may be less accurate")
            # This shouldn't happen if frontend sends mask, but handle gracefully
            raise BadRequest(
                "Mask polygon required for bbox-based search. "
                "Please use full detection if mask not available."
            )
        
        with timed("Segmentation"):
            segment = await self.segmentation.segment(img, bbox, mask_polygon=mask_polygon)
        
        # NOTE: ROA validation SKIPPED for retrieval (user-friendly)
        
        # Generate crops based on detected category
        with timed("Crop generation and Masking"):
            base_crops, base_bboxes = self.preprocessing.crop_base(img, bbox)
            
            # Category-based crop selection (3-tier)
            if query_category and query_category in SMALL_CATEGORIES:
                crops_to_process = ["tight"]
                logger.info(f"📦 Small object '{query_category}': 4 crops")
            elif query_category and query_category in MEDIUM_CATEGORIES:
                crops_to_process = ["tight"]
                logger.info(f"📦 Medium object '{query_category}': 4 crops")
            elif query_category and query_category in LARGE_CATEGORIES:
                crops_to_process = ["tight", "medium", "full"]
                logger.info(f"📦 Large object '{query_category}': 9 crops")
            else:
                # Unknown category: default to tight only
                crops_to_process = ["tight"]
                logger.warning(f"Unknown category '{query_category}', using 4 crops")
            
            filtered_crops = {k: v for k, v in base_crops.items() if k in crops_to_process}
            filtered_bboxes = {k: v for k, v in base_bboxes.items() if k in crops_to_process}
            
            # Mask tight crop only
            filtered_crops["tight"] = self.preprocessing.apply_mask_on_crop(
                filtered_crops["tight"], 
                segment.mask, 
                bbox=filtered_bboxes["tight"]
            )
            
            # Generate augmentations
            crops = self.preprocessing.add_rotated_crops(filtered_crops)
            
            logger.info(f"✅ Generated {len(crops)} crops")
        
        # DEBUG: Save all crops to debug directory for visualization
        try:
            from pathlib import Path
            debug_dir = Path("debug_crops_search")
            debug_dir.mkdir(exist_ok=True)
            
            import time
            timestamp = int(time.time() * 1000)
            
            for crop_name, crop_img in crops.items():
                # Use category + timestamp for filename
                safe_category = query_category.replace("/", "-") if query_category else "unknown"
                debug_path = debug_dir / f"{safe_category}_{timestamp}_{crop_name}.jpg"
                crop_img.save(debug_path, quality=95)
            
            logger.info(f"💾 Saved {len(crops)} search crops to {debug_dir}/ ({query_category}_{timestamp})")
        except Exception as e:
            logger.warning(f"Failed to save debug crops (non-critical): {e}")
        
        # Embedding and search (same as search_room_image)
        with timed("Embedding"):
            query_vector = self.embedding.embed_crops(crops)
        
        # Estimate catalog size
        catalog_size = await self._estimate_catalog_size(query_category)
        
        if catalog_size > 50000:
            candidate_k = min(max(top_k * 40, 1000), 5000)
        elif catalog_size > 10000:
            candidate_k = min(max(top_k * 30, 500), 2000)
        else:
            candidate_k = min(max(top_k * 15, 100), 1000)
        
        logger.info(f"Retrieving {candidate_k} candidates (catalog: {catalog_size:,})")

        with timed("Vector search"):
            candidates = self.vectors.query(
                vector=query_vector,
                top_k=candidate_k,
                category=query_category,
            )
        
        logger.info(f"Retrieved {len(candidates)} candidates")

        if not candidates:
            message = "No products found in the catalog"
            if query_category:
                message += f" for category '{query_category}'"
            return SearchResponse(
                query_category=query_category,
                hits=[],
                message=message
            )

        with timed("Reranking"):
            rerank_k = top_k * 3
            reranked = self.rerank.rerank(
                query_vector=query_vector,
                candidates=candidates,
                top_k=rerank_k,
                exact_first=True,
            )
        
        if not reranked:
            return SearchResponse(
                query_category=query_category,
                hits=[],
                message="No relevant products found"
            )

        # Deduplicate and return top_k
        seen_ids = {}
        for c in reranked:
            product_id = c.get("id")
            if product_id and product_id not in seen_ids:
                seen_ids[product_id] = c
            elif product_id and c["final_score"] > seen_ids[product_id]["final_score"]:
                seen_ids[product_id] = c

        unique_reranked = sorted(
            seen_ids.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )[:top_k]
        
        logger.info(f"After deduplication: {len(unique_reranked)} unique products")

        hits = []
        for candidate in unique_reranked:
            metadata = candidate.get("metadata", {})
            hits.append(
                SearchHit(
                    pinecone_id=candidate["id"],
                    score=candidate["final_score"],
                    image_url=metadata.get("image_url", ""),
                    product_url=metadata.get("product_url", ""),
                    name_english=metadata.get("name_english", ""),
                    name_arabic=metadata.get("name_arabic", ""),
                    category=metadata.get("category", ""),
                    price_amount=metadata.get("price_amount", 0),
                    price_unit=metadata.get("price_unit", ""),
                )
            )
        
        return SearchResponse(
            query_category=query_category,
            hits=hits,
        )

    async def search_room_image(
        self,
        *,
        room_image_file: UploadFile,
        assigned_category: str | None,
        top_k: int,
    ) -> SearchResponse:
        from loguru import logger
        from app.utils.timing import timed
        import numpy as np
        
        logger.info(f"Search request: category={assigned_category}, top_k={top_k}")

        # Stage 1: Image quality validation (without category - universal checks)
        with timed("Image load"):
            img = await self.image_io.read_upload_as_rgb(
                room_image_file
                # No category - uses default blur threshold (30)
            )
        
        w, h = img.size
        logger.debug(f"Image size: {w}x{h}")

        # Run detection (REQUIRED - no manual bbox, no fallback)
        with timed("Detection"):
                detections = await self.detection.detect(
                    img, 
                category_hint=assigned_category.lower().strip() if assigned_category else None
                )
            
        logger.info(f"Detected {len(detections)} objects")
            
        # STRICT: Reject if no detection
        if not detections:
            raise BadRequest(
                "Could not detect any objects in the image. "
                "Please ensure the image shows a clear view of the product "
                "with good lighting and contrast."
            )
        
        # Select detection with LARGEST BBOX AREA
        # For retrieval: prefer assigned_category if provided, but be flexible
        if assigned_category:
            # Filter by assigned category
            matching_detections = [d for d in detections if d.category == assigned_category]
            
            if matching_detections:
                # Found matching category - pick largest
                det = max(
                    matching_detections,
                    key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
                )
                if len(matching_detections) > 1:
                    logger.info(
                        f"📦 Multiple '{assigned_category}' detected ({len(matching_detections)}). "
                        f"Selected LARGEST: {(det.bbox[2]-det.bbox[0]):.0f}×{(det.bbox[3]-det.bbox[1]):.0f}px "
                        f"(score: {det.score:.3f})"
                    )
            else:
                # No match - be user-friendly, pick largest overall
                det = max(
                    detections,
                    key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
                )
                available_categories = list(set(d.category for d in detections))
                logger.warning(
                    f"⚠️ No '{assigned_category}' detected. Found: {available_categories}. "
                    f"Using largest object: '{det.category}' for search."
                )
        else:
            # No category hint - pick largest object
            det = max(
                detections,
                key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
            )
            logger.info(
                f"📦 No category hint. Selected LARGEST: '{det.category}' "
                f"{(det.bbox[2]-det.bbox[0]):.0f}×{(det.bbox[3]-det.bbox[1]):.0f}px "
                f"(score: {det.score:.3f})"
            )
        
        bbox = BBox(x1=det.bbox[0], y1=det.bbox[1], x2=det.bbox[2], y2=det.bbox[3])
        mask_polygon = getattr(det, 'mask_polygon', None)
        query_category = det.category.lower().strip() if det.category else None
        
        logger.info(f"Using detected bbox and polygon mask from RF-DETR")
        if mask_polygon:
            logger.info(f"Polygon mask available with {len(mask_polygon)} points")

        try:
            bbox = self.preprocessing.clamp_bbox(bbox, w, h)
        except ValueError as e:
            raise BadRequest(f"Invalid bounding box: {str(e)}")
        
        # Segment using polygon mask from RF-DETR (required, no fallback)
        if not mask_polygon:
            raise BadRequest(
                "No polygon mask from detection. Polygon-based segmentation is required."
            )
        
        with timed("Segmentation"):
            segment = await self.segmentation.segment(img, bbox, mask_polygon=mask_polygon)
            logger.debug("Used polygon-based segmentation")

        # NOTE: ROA validation SKIPPED for retrieval (user-friendly)
        # User's room photo framing shouldn't reject their search
        
        # Generate crops with same pattern as ingestion
        with timed("Crop generation and Masking"):
            # Generate ALL base crops (tight, medium, full) and their bboxes
            base_crops, base_bboxes = self.preprocessing.crop_base(img, bbox)
            
            # === CATEGORY-BASED CROP SELECTION (3-TIER, same as ingestion) ===
            if query_category and query_category in SMALL_CATEGORIES:
                # Small objects: tight only (background is noise)
                crops_to_process = ["tight"]
                logger.info(
                    f"📦 Small object '{query_category}': Generating TIGHT crop only "
                    f"(4 crops)"
                )
            elif query_category and query_category in MEDIUM_CATEGORIES:
                # Medium objects: tight only (background mostly noise)
                crops_to_process = ["tight"]
                logger.info(
                    f"📦 Medium object '{query_category}': Generating TIGHT crop only "
                    f"(4 crops)"
                )
            elif query_category and query_category in LARGE_CATEGORIES:
                # Large objects: all crops (context helps matching)
                crops_to_process = ["tight", "medium", "full"]
                logger.info(
                    f"📦 Large object '{query_category}': Generating ALL crops "
                    f"(9 crops: tight + medium + full)"
                )
            else:
                # Unknown category from detection: default to tight only (safe fallback)
                crops_to_process = ["tight"]
                logger.warning(
                    f"Unknown detected category '{query_category}', "
                    f"using tight only (4 crops)"
                )
            
            # Filter crops to only those needed
            filtered_crops = {k: v for k, v in base_crops.items() if k in crops_to_process}
            filtered_bboxes = {k: v for k, v in base_bboxes.items() if k in crops_to_process}
            
            # === MASKING: TIGHT CROP ONLY ===
            # Always mask tight crop for pure object features
            filtered_crops["tight"] = self.preprocessing.apply_mask_on_crop(
                filtered_crops["tight"], 
                segment.mask, 
                bbox=filtered_bboxes["tight"]
            )
            # Medium and full crops remain unmasked (natural context preserved)
            
            logger.debug(
                f"Masking: tight=masked, "
                f"{'medium+full=unmasked' if len(crops_to_process) > 1 else 'no other crops'}"
            )

            # Generate augmentations (±5° rotations + horizontal flip)
            crops = self.preprocessing.add_rotated_crops(filtered_crops)
            
            logger.info(
                f"✅ Generated {len(crops)} total crops "
                f"({len(filtered_crops)} base + {len(crops) - len(filtered_crops)} augmented)"
            )
        
        # Validate crops are not empty
        for crop_name, crop_img in crops.items():
            if crop_img.size[0] == 0 or crop_img.size[1] == 0:
                raise BadRequest(
                    f"Invalid crop '{crop_name}': crop has zero size. "
                    f"Please check your bounding box coordinates."
                )

        # DEBUG: Save all crops to debug directory for visualization
        try:
            from pathlib import Path
            debug_dir = Path("debug_crops_search")
            debug_dir.mkdir(exist_ok=True)
            
            import time
            timestamp = int(time.time() * 1000)
            
            for crop_name, crop_img in crops.items():
                debug_path = debug_dir / f"{timestamp}_{crop_name}.jpg"
                crop_img.save(debug_path, quality=95)
            
            logger.info(f"💾 Saved {len(crops)} debug crops to {debug_dir}/ with timestamp {timestamp}")
        except Exception as e:
            logger.warning(f"Failed to save debug crops (non-critical): {e}")

        with timed("Embedding"):
            query_vector = self.embedding.embed_crops(crops)
        
        logger.debug(f"Query vector norm: {np.linalg.norm(query_vector):.3f}")

        # Adaptive candidate count based on catalog size for optimal recall
        # For large catalogs (50K+), retrieve more candidates to ensure relevant items aren't missed
        catalog_size = await self._estimate_catalog_size(query_category)
        
        if catalog_size > 50000:
            # Very large catalog (50K+): Retrieve more candidates
            candidate_multiplier = 40
            candidate_k = min(max(top_k * candidate_multiplier, 1000), 5000)
        elif catalog_size > 10000:
            # Large catalog (10K-50K): Moderate increase
            candidate_multiplier = 30
            candidate_k = min(max(top_k * candidate_multiplier, 500), 2000)
        else:
            # Small/Medium catalog (<10K): Standard retrieval
            candidate_multiplier = 15
            candidate_k = min(max(top_k * candidate_multiplier, 100), 1000)
        
        logger.info(
            f"Catalog size estimate: {catalog_size:,} products, "
            f"retrieving {candidate_k} candidates (multiplier: {candidate_multiplier}x)"
        )

        with timed("Vector search"):
            candidates = self.vectors.query(
                vector=query_vector,
                top_k=candidate_k,
                category=query_category,  # Use category-based namespace for fast search
            )
        
        logger.info(f"Retrieved {len(candidates)} candidates from Pinecone (namespace: {query_category or 'default'})")

        # Check if no candidates were found
        if not candidates:
            message = "No products found in the catalog"
            if query_category:
                message += f" matching the category '{query_category}'"
            message += ". Please add products to the catalog first."
            logger.warning(message)
            return SearchResponse(
                query_category=query_category,
                hits=[],
                message=message
            )

        with timed("Reranking"):
            # Get extra results for deduplication
            rerank_k = top_k * 3
            reranked = self.rerank.rerank(
                query_vector=query_vector,
                candidates=candidates,
                top_k=rerank_k,
                exact_first=True,
            )
        
        logger.info(f"Reranked to {len(reranked)} results")
        if reranked:
            logger.debug(
                f"Score range: {reranked[0]['final_score']:.3f} (best) to "
                f"{reranked[-1]['final_score']:.3f} (worst)"
            )

        # Check if reranking resulted in no hits
        if not reranked:
            message = "No relevant products found"
            if query_category:
                message += f" for category '{query_category}'"
            message += ". Try adjusting your search criteria or use a different image."
            logger.warning(message)
            return SearchResponse(
                query_category=query_category,
                hits=[],
                message=message
            )

        # Deduplicate by pinecone_id (since each product has unique pinecone_id, 
        # this effectively removes duplicate vector entries if any exist)
        seen_ids = {}
        for c in reranked:
            # Get pinecone_id from the vector id (stored in 'id' field)
            product_id = c.get("id")
            if not product_id:
                logger.warning("Skipping result without ID")
                continue
                
            if product_id not in seen_ids:
                seen_ids[product_id] = c
            elif c["final_score"] > seen_ids[product_id]["final_score"]:
                # Replace with higher scoring instance
                seen_ids[product_id] = c

        # Take top K unique products
        unique_reranked = sorted(
            seen_ids.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )[:top_k]
        
        logger.info(f"After deduplication: {len(unique_reranked)} unique products")

        # Build response hits with new metadata structure
        hits = []
        for c in unique_reranked:
            metadata = c.get("metadata", {})
            
            hits.append(
                SearchHit(
                    pinecone_id=c.get("id", "unknown"),
                    score=float(c["final_score"]),
                    image_url=metadata.get("image_url"),
                    product_url=metadata.get("product_url"),
                    name_english=metadata.get("name_english"),
                    name_arabic=metadata.get("name_arabic"),
                    category=metadata.get("category"),
                    price_amount=metadata.get("price_amount"),
                    price_unit=metadata.get("price_unit"),
                    is_active=metadata.get("is_active"),
                    store_id=metadata.get("store_id"),
                    countries=metadata.get("countries"),
                    store=metadata.get("store"),
                )
            )

        return SearchResponse(query_category=query_category, hits=hits)
    
    async def _estimate_catalog_size(self, category: str | None) -> int:
        """
        Estimate catalog size for the given category to optimize candidate retrieval.
        
        For large catalogs (50K+), we need to retrieve more candidates to ensure
        relevant items aren't missed during ANN search.
        
        Args:
            category: Category to estimate size for (None = all categories)
            
        Returns:
            Approximate number of products in the catalog
        """
        from loguru import logger
        
        try:
            # Get index stats from Pinecone
            stats = self.vectors.get_stats(category=category)
            vector_count = stats.get('total_vector_count', 1000)
            
            logger.debug(
                f"Catalog size for category '{category or 'all'}': {vector_count:,} vectors"
            )
            
            return vector_count
            
        except Exception as e:
            logger.warning(f"Failed to get catalog size (using default): {e}")
            # Fallback: assume medium catalog
            return 5000