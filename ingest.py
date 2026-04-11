#!/usr/bin/env python3
"""
Simplified demo catalog ingestion using YOLO11n + Gemini embeddings.

CSV Format (Required Columns):
  image_url    - URL to product image
  product_name - Display name stored in Pinecone metadata
  category     - Must be a supported YOLO category (see CATEGORY_TO_YOLO below)

Supported categories:
  chair, couch, sofa, bed, dining-table, tv, clock, wall-clock, vase, laptop, tennis-racket

How it works:
  1. Download image from image_url
  2. Run YOLO11n detection (local, CPU, ~2.6MB model)
  3. Select highest-confidence bbox matching category
  4. Tight bbox crop (no masking, no padding)
  5. Gemini multimodal embedding (3072-dim)
  6. Upsert to Pinecone (id = md5(image_url)[:16])
  
  Pinecone metadata stored: { image_url, product_name, category }

Usage:
    python ingest.py --csv data/demo.csv
    python ingest.py --csv data/demo.csv --max-workers 4 --confidence 0.15
    python ingest.py --csv data/demo.csv --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from loguru import logger
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Category → YOLO class name mapping
# Add entries here to support more categories.
# YOLO11n uses COCO class names exactly as returned by model.names[cls_id].
# ---------------------------------------------------------------------------
CATEGORY_TO_YOLO: dict[str, str] = {
    "chair": "chair",
    "couch": "couch",
    "sofa": "couch",
    "bed": "bed",
    "dining-table": "dining table",
    "dining table": "dining table",
    "tv": "tv",
    "clock": "clock",
    "wall-clock": "clock",
    "vase": "vase",
    "laptop": "laptop",
    "tennis-racket": "tennis racket",
}

SUPPORTED_CATEGORIES = set(CATEGORY_TO_YOLO.keys())

DEFAULT_MAX_WORKERS = 4
DEFAULT_CONFIDENCE = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pinecone_id(image_url: str) -> str:
    """Deterministic 16-char hex ID from image URL (idempotent on re-runs)."""
    return hashlib.md5(image_url.encode()).hexdigest()[:16]


def download_image(image_url: str, timeout: int = 20) -> Image.Image:
    """Download image from URL and return as RGB PIL Image."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(image_url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return img


def detect_best_bbox(
    model: Any,
    img: Image.Image,
    yolo_class: str,
    confidence_threshold: float,
) -> tuple[int, int, int, int] | None:
    """
    Run YOLO11n inference and return the (x1, y1, x2, y2) bbox of the
    highest-confidence detection matching yolo_class, or None if not found.
    """
    results = model(img, verbose=False)
    if not results:
        return None

    best_bbox: tuple[int, int, int, int] | None = None
    best_conf = -1.0

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id].lower()
            conf = float(box.conf[0].item())

            if cls_name != yolo_class.lower():
                continue
            if conf < confidence_threshold:
                continue
            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                best_bbox = (int(x1), int(y1), int(x2), int(y2))

    if best_bbox:
        logger.debug(
            f"YOLO detected '{yolo_class}' bbox={best_bbox} conf={best_conf:.3f}"
        )
    return best_bbox


def tight_crop(img: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Exact bbox crop — no padding, no mask, no white fill."""
    x1, y1, x2, y2 = bbox
    w, h = img.size
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return img.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Per-row pipeline (runs in a thread pool worker)
# ---------------------------------------------------------------------------

def ingest_row(
    row: dict[str, str],
    model: Any,
    embedding_service: Any,
    vector_repo: Any,
    confidence_threshold: float,
    dry_run: bool,
) -> dict[str, str]:
    """
    Full pipeline for one CSV row.
    Returns {"status": "ok"|"fail", "pinecone_id": ..., "image_url": ..., "error": ...}.
    """
    image_url = row["image_url"].strip()
    product_name = row["product_name"].strip()
    category = row["category"].strip().lower()
    pinecone_id = make_pinecone_id(image_url)

    if category not in SUPPORTED_CATEGORIES:
        return {
            "status": "fail",
            "pinecone_id": pinecone_id,
            "image_url": image_url,
            "error": (
                f"Unsupported category '{category}'. "
                f"Supported: {sorted(SUPPORTED_CATEGORIES)}"
            ),
        }

    yolo_class = CATEGORY_TO_YOLO[category]

    try:
        # 1. Download
        img = download_image(image_url)
        logger.debug(f"[{pinecone_id}] Downloaded {img.size[0]}×{img.size[1]}px")

        # 2. Detect
        bbox = detect_best_bbox(model, img, yolo_class, confidence_threshold)
        if bbox is None:
            raise ValueError(
                f"No '{yolo_class}' detected at confidence ≥ {confidence_threshold:.0%}"
            )

        # 3. Tight bbox crop (no masking)
        crop = tight_crop(img, bbox)
        logger.debug(f"[{pinecone_id}] Crop: {crop.size[0]}×{crop.size[1]}px")

        # 4. Embed (single tight crop)
        vector = embedding_service.embed_crops({"tight": crop})

        # 5. Upsert to Pinecone
        metadata: dict[str, Any] = {
            "image_url": image_url,
            "product_name": product_name,
            "category": category,
        }
        if not dry_run:
            vector_repo.upsert(
                vector_id=pinecone_id,
                vector=vector,
                metadata=metadata,
                namespace=category,
            )

        logger.info(f"[{pinecone_id}] ✅ '{product_name}' ({category})")
        return {"status": "ok", "pinecone_id": pinecone_id, "image_url": image_url, "error": ""}

    except Exception as exc:
        logger.error(f"[{pinecone_id}] ❌ {exc}")
        return {
            "status": "fail",
            "pinecone_id": pinecone_id,
            "image_url": image_url,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo catalog ingestion using YOLO11n detection + Gemini embeddings."
    )
    parser.add_argument("--csv", required=True,
                        help="Path to CSV with columns: image_url, product_name, category")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"Concurrent workers (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                        help=f"YOLO confidence threshold (default: {DEFAULT_CONFIDENCE})")
    parser.add_argument("--failures-csv", default="failed_ingestions.csv",
                        help="Path to write failed rows (default: failed_ingestions.csv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run full pipeline but skip Pinecone upsert")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    logger.add("ingestion.log", level="DEBUG", rotation="10 MB")

    # ------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.error("CSV is empty")
        sys.exit(1)

    required_cols = {"image_url", "product_name", "category"}
    missing = required_cols - {c.lower() for c in rows[0].keys()}
    if missing:
        logger.error(f"CSV missing required columns: {missing}")
        sys.exit(1)

    logger.info(f"Loaded {len(rows)} rows from {csv_path}")

    # ------------------------------------------------------------------
    # Initialize services
    # ------------------------------------------------------------------
    logger.info("Loading YOLO11n (downloads ~2.6MB on first run)...")
    from ultralytics import YOLO as UltralyticsYOLO
    yolo_model = UltralyticsYOLO("yolo11n.pt")
    logger.success("YOLO11n ready")

    from app.core.config import Settings
    from app.services.embedding_gemini import GeminiEmbeddingService
    from app.repositories.pinecone_repo import PineconeVectorRepository

    settings = Settings()

    embedding_service = GeminiEmbeddingService(
        api_key=settings.google_api_key,
        model=settings.gemini_embedding_model,
        output_dimensionality=settings.pinecone_dim,
    )
    await embedding_service.load()
    logger.success("Gemini embedding service ready")

    pinecone_api_key = os.environ.get("PINECONE_API_KEY_OVERRIDE") or settings.pinecone_api_key
    pinecone_index = os.environ.get("PINECONE_INDEX_OVERRIDE") or settings.pinecone_index_name
    logger.info(f"Pinecone index: {pinecone_index}")

    vector_repo = PineconeVectorRepository(
        api_key=pinecone_api_key,
        index_name=pinecone_index,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        dimension=settings.pinecone_dim,
    )
    await vector_repo.ensure_index()
    logger.success(f"Pinecone ready — index: {pinecone_index}")

    if args.dry_run:
        logger.warning("DRY-RUN mode — Pinecone upsert will be skipped")

    # ------------------------------------------------------------------
    # Process rows concurrently
    # ------------------------------------------------------------------
    results: list[dict] = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                ingest_row,
                row,
                yolo_model,
                embedding_service,
                vector_repo,
                args.confidence,
                args.dry_run,
            ): row
            for row in rows
        }

        with tqdm(total=len(rows), desc="Ingesting", unit="product") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                icon = "✅" if result["status"] == "ok" else "❌"
                pbar.set_postfix(last=f"{icon} {result['pinecone_id']}")
                pbar.update(1)

    elapsed = time.time() - start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "fail"]

    print("\n" + "=" * 55)
    print("INGESTION SUMMARY")
    print("=" * 55)
    print(f"Total:    {len(results)}")
    print(f"✅ OK:    {len(ok)}")
    print(f"❌ Failed: {len(failed)}")
    if results:
        print(f"Rate:     {len(ok)/len(results)*100:.1f}%")
    print(f"Time:     {elapsed:.1f}s")
    print("=" * 55)

    if failed:
        failures_path = Path(args.failures_csv)
        with open(failures_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["pinecone_id", "image_url", "error"])
            writer.writeheader()
            writer.writerows(
                {"pinecone_id": r["pinecone_id"], "image_url": r["image_url"], "error": r["error"]}
                for r in failed
            )
        logger.info(f"Failed rows saved to: {failures_path}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    asyncio.run(main())
