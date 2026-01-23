#!/usr/bin/env python3
"""
Batch catalog ingestion from CSV file with new data structure.

CSV Format (Required Columns):
- pinecone_id: Unique product identifier (used as Pinecone vector ID)
- image_url: URL to product image
- assigned_category: Category for detection and Pinecone namespace
- name_english: English product name
- name_arabic: Arabic product name
- price_amount: Product price (will be converted to int)
- price_unit: Currency code (SAR, USD, etc.)
- is_active: Product active status (boolean)
- store_id: Store identifier (will be converted to int)
- countries: Available countries (e.g., "[SA,AE]" → ["SA", "AE"])
- store: Store name
- product_url: URL to product page

Usage:
    python ingest.py --csv data/products.csv --batch-size 10 --max-workers 6
    python ingest.py --csv data/products.csv --api-url http://localhost:8000
    
Performance:
    --max-workers: Controls concurrent RunPod API calls (default: 6)
    --batch-size: Controls batch size for progress tracking (default: 10)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

# Import categories from centralized constants (single source of truth)
from app.core.constants import SMALL_CATEGORIES, MEDIUM_CATEGORIES, LARGE_CATEGORIES

# All supported categories (46 total: 10 small + 19 medium + 17 large)
ALL_CATEGORIES = set(SMALL_CATEGORIES + MEDIUM_CATEGORIES + LARGE_CATEGORIES)

# Metadata columns that will be stored in Pinecone
# Note: assigned_category from CSV will be stored as "category" in metadata
METADATA_COLS = [
    "image_url",
    "product_url",
    "name_english",
    "name_arabic",
    "category",  # This will be populated from assigned_category
    "price_amount",
    "price_unit",
    "is_active",
    "store_id",
    "countries",
    "store"
]


def prepare_metadata(row: pd.Series) -> dict[str, Any]:
    """
    Prepare metadata dict from row with correct data types.
    Handles NaN values and type conversions per Pinecone requirements.
    
    Note: Maps 'assigned_category' from CSV to 'category' in metadata.
    """
    metadata = {}

    for col in METADATA_COLS:
        col_lower = col.lower()
        
        # Special handling: map 'category' metadata key to 'assigned_category' CSV column
        csv_col = 'assigned_category' if col_lower == 'category' else col_lower
        
        if csv_col in row.index:
            val = row[csv_col]

            # Handle NaN values with appropriate defaults
            if pd.isna(val):
                if col_lower in ['store_id', 'price_amount']:
                    metadata[col] = 0
                elif col_lower == 'is_active':
                    metadata[col] = False
                elif col_lower == 'countries':
                    metadata[col] = []
                else:
                    metadata[col] = ""
                continue

            # Type conversions based on field
            if col_lower == 'store_id':
                # Convert to integer
                try:
                    metadata[col] = int(float(val))
                except (ValueError, TypeError):
                    metadata[col] = 0

            elif col_lower == 'price_amount':
                # Convert to integer (599, not 599.00)
                try:
                    metadata[col] = int(float(val))
                except (ValueError, TypeError):
                    metadata[col] = 0

            elif col_lower == 'is_active':
                # Convert to boolean
                if isinstance(val, str):
                    metadata[col] = val.lower() in ['true', '1', 'yes']
                elif isinstance(val, bool):
                    metadata[col] = val
                else:
                    metadata[col] = bool(val)

            elif col_lower == 'countries':
                # Convert to list of strings: ["SA", "QA"]
                if isinstance(val, list):
                    metadata[col] = val
                elif isinstance(val, str):
                    # Parse "[SA]" or "[SA,QA]" → ["SA"] or ["SA", "QA"]
                    countries_str = val.strip('[]')
                    if countries_str:
                        metadata[col] = [c.strip() for c in countries_str.split(',')]
                    else:
                        metadata[col] = []
                else:
                    metadata[col] = []

            else:
                # All other fields as strings
                metadata[col] = str(val)

    return metadata


class CatalogIngester:
    """Batch catalog ingestion from CSV with new data structure."""
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        timeout: float = 300.0,
        max_retries: int = 3,
        max_workers: int = 6,
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max_workers
        
        # Statistics
        self.total = 0
        self.success = 0
        self.failed = 0
        self.skipped_categories = 0  # Rows skipped due to invalid categories
        self.failed_items: list[dict[str, Any]] = []
    
    def read_csv(self, csv_path: str) -> pd.DataFrame:
        """Read CSV file and validate required columns."""
        path = Path(csv_path)
        
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.info(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Validate required columns
        required_cols = ['pinecone_id', 'image_url', 'assigned_category']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate metadata columns (warn if missing)
        missing_metadata = [col.lower() for col in METADATA_COLS if col.lower() not in df.columns]
        if missing_metadata:
            logger.warning(f"Missing optional metadata columns: {missing_metadata}")
        
        # === VALIDATE CATEGORIES IN CSV ===
        # Filter out rows with invalid categories (skip instead of failing)
        invalid_categories = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            cat = str(row['assigned_category']).lower().strip()
            if cat not in ALL_CATEGORIES:
                invalid_categories.append((idx, cat, row.get('pinecone_id', 'unknown')))
            else:
                valid_indices.append(idx)
        
        # Filter dataframe to only include valid categories
        df_filtered = df.loc[valid_indices].copy()
        
        # Track skipped categories count
        self.skipped_categories = len(invalid_categories)
        
        if invalid_categories:
            logger.warning(
                f"⚠️  Found {len(invalid_categories)} rows with invalid categories (will be skipped):"
            )
            # Group by category for cleaner output
            invalid_by_cat = {}
            for idx, cat, pid in invalid_categories:
                if cat not in invalid_by_cat:
                    invalid_by_cat[cat] = []
                invalid_by_cat[cat].append((idx, pid))
            
            for cat, items in sorted(invalid_by_cat.items()):
                logger.warning(f"  '{cat}': {len(items)} rows (e.g., ID: {items[0][1]})")
            
            logger.warning(
                f"  → Skipping {len(invalid_categories)} rows, processing {len(df_filtered)} valid rows"
            )
        else:
            logger.info(
                f"✓ All categories valid ({len(df_filtered)} products, "
                f"{len(ALL_CATEGORIES)} supported categories)"
            )
        
        logger.info(f"Loaded {len(df_filtered)} products from CSV (after filtering invalid categories)")
        return df_filtered
    
    async def download_image(self, image_url: str) -> bytes:
        """Download image from URL with validation and browser-like headers."""
        try:
            logger.debug(f"Downloading image from: {image_url}")
            
            # Add browser-like headers to avoid 403 Forbidden errors from CDNs
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(image_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Check file size
            size_mb = len(response.content) / (1024 * 1024)
            if size_mb > 15:
                raise ValueError(f"Image too large: {size_mb:.1f}MB (max 15MB)")
            
            logger.debug(f"Downloaded {size_mb:.2f}MB")
            return response.content
            
        except Exception as e:
            raise ValueError(f"Failed to download image from {image_url}: {str(e)}")
    
    async def upsert_catalog_item(
        self,
        pinecone_id: str,
        assigned_category: str,
        image_bytes: bytes,
        metadata: dict[str, Any],
        client: httpx.AsyncClient,
    ) -> dict:
        """Upload product to catalog with new data structure."""
        # Prepare form data
        files = {
            "image": ("image.jpg", image_bytes, "image/jpeg"),
        }
        
        # Send only required fields to API
        data = {
            "pinecone_id": pinecone_id,
            "assigned_category": assigned_category,
            # Metadata is sent as JSON string
            "metadata_json": json.dumps(metadata),
        }
        
        try:
            response = await client.post(
                f"{self.api_base_url}/api/v1/catalog/upsert",
                files=files,
                data=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError:
            # Re-raise HTTPStatusError to preserve status code for retry logic
            raise
        except Exception as e:
            raise ValueError(f"Failed to upsert: {str(e)}")
    
    async def process_item(
        self,
        row: pd.Series,
        client: httpx.AsyncClient,
    ) -> bool:
        """Process a single item with retries."""
        # Keep pinecone_id as string (supports both numeric and alphanumeric IDs)
        try:
            pinecone_id = str(row['pinecone_id']).strip()
            if not pinecone_id:
                raise ValueError("Empty pinecone_id")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid pinecone_id: {row.get('pinecone_id')} - {e}")
            return False
        
        image_url = str(row['image_url'])
        assigned_category = str(row['assigned_category'])
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Step 1: Prepare metadata with correct types
                metadata = prepare_metadata(row)
                
                # Step 2: Download image (no retry on download failures)
                logger.debug(f"[{pinecone_id}] Downloading image")
                try:
                    image_bytes = await self.download_image(image_url)
                except Exception as download_error:
                    # Immediately fail on download errors (404, network issues, etc.) - no retry
                    logger.error(
                        f"[{pinecone_id}] ❌ Download failed (no retry): {download_error}"
                    )
                    self.failed_items.append({
                        "pinecone_id": pinecone_id,
                        "assigned_category": assigned_category,
                        "image_url": image_url,
                        "error_type": "DOWNLOAD",
                        "error": f"Download failed: {str(download_error)}",
                    })
                    return False
                
                # Step 3: Upload to catalog
                logger.debug(f"[{pinecone_id}] Uploading to catalog (category: {assigned_category})")
                result = await self.upsert_catalog_item(
                    pinecone_id=pinecone_id,
                    assigned_category=assigned_category,
                    image_bytes=image_bytes,
                    metadata=metadata,
                    client=client,
                )
                
                logger.success(
                    f"[{pinecone_id}] ✅ Successfully processed (category: {assigned_category})"
                )
                return True
                
            except httpx.HTTPStatusError as e:
                # Check if it's a validation error (400) - no need to retry
                if e.response.status_code == 400:
                    error_detail = e.response.text
                    logger.error(
                        f"[{pinecone_id}] ❌ Validation failed (no retry): {error_detail}"
                    )
                    
                    # Categorize error type for analysis
                    error_msg = error_detail.lower()
                    if "blur" in error_msg:
                        error_type = "BLUR"
                    elif "roa" in error_msg or "distant" in error_msg or "too small" in error_msg:
                        error_type = "ROA"
                    elif "brightness" in error_msg:
                        error_type = "BRIGHTNESS"
                    elif "contrast" in error_msg or "blank" in error_msg or "uniform" in error_msg:
                        error_type = "CONTRAST"
                    elif "dimension" in error_msg or "resolution" in error_msg:
                        error_type = "DIMENSION"
                    elif "detect" in error_msg:
                        error_type = "NO_DETECTION"
                    else:
                        error_type = "OTHER"
                    
                    self.failed_items.append({
                        "pinecone_id": pinecone_id,
                        "assigned_category": assigned_category,
                        "image_url": image_url,
                        "error_type": error_type,
                        "error": error_detail,
                    })
                    return False
                
                # For other HTTP errors (5xx, network issues), retry
                if attempt < self.max_retries:
                    logger.warning(f"[{pinecone_id}] Attempt {attempt} failed: {e}. Retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"[{pinecone_id}] ❌ Failed after {self.max_retries} attempts: {e}")
                    self.failed_items.append({
                        "pinecone_id": pinecone_id,
                        "assigned_category": assigned_category,
                        "image_url": image_url,
                        "error_type": "HTTP_ERROR",
                        "error": str(e),
                    })
                    return False
            
            except Exception as e:
                # For non-HTTP errors, retry
                if attempt < self.max_retries:
                    logger.warning(f"[{pinecone_id}] Attempt {attempt} failed: {e}. Retrying...")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"[{pinecone_id}] ❌ Failed after {self.max_retries} attempts: {e}")
                    self.failed_items.append({
                        "pinecone_id": pinecone_id,
                        "assigned_category": assigned_category,
                        "image_url": image_url,
                        "error_type": "UNKNOWN",
                        "error": str(e),
                    })
                    return False
        
        return False
    
    async def process_item_with_semaphore(
        self,
        row: pd.Series,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> bool:
        """Process a single item with semaphore to limit concurrency."""
        async with semaphore:
            return await self.process_item(row, client)
    
    async def process_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 10,
    ) -> None:
        """
        Process all items with controlled concurrency.
        Uses semaphore to limit concurrent RunPod API calls to max_workers.
        """
        self.total = len(df)
        
        # Create semaphore to limit concurrent workers (prevents RunPod API overload)
        semaphore = asyncio.Semaphore(self.max_workers)
        
        logger.info(
            f"Starting ingestion with {self.max_workers} concurrent workers "
            f"({self.total} products total)"
        )
        
        async with httpx.AsyncClient() as client:
            # Process in batches for progress tracking
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                
                logger.info(
                    f"Processing batch {i // batch_size + 1} "
                    f"({len(batch)} items, {self.max_workers} workers)"
                )
                
                # Process batch concurrently with semaphore limit
                tasks = [
                    self.process_item_with_semaphore(row, client, semaphore) 
                    for _, row in batch.iterrows()
                ]
                results = await asyncio.gather(*tasks)
                
                # Update statistics
                self.success += sum(results)
                self.failed += len(results) - sum(results)
                
                logger.info(
                    f"Batch complete. Success: {sum(results)}/{len(results)}, "
                    f"Total: {self.success}/{self.total} "
                    f"({self.success / self.total * 100:.1f}% success rate)"
                )
    
    def print_summary(self) -> None:
        """Print ingestion summary with detailed failure analysis."""
        print("\n" + "=" * 60)
        print("INGESTION SUMMARY")
        print("=" * 60)
        print(f"Total products:     {self.total}")
        if self.skipped_categories > 0:
            print(f"⚠️  Skipped (invalid category): {self.skipped_categories}")
        print(f"✅ Successfully ingested: {self.success}")
        print(f"❌ Failed:          {self.failed}")
        if self.total > 0:
            print(f"Success rate:       {self.success / self.total * 100:.1f}%")
        print("=" * 60)
        
        if self.failed_items:
            # Analyze failure reasons by error type
            error_types = {}
            for item in self.failed_items:
                etype = item.get('error_type', 'OTHER')
                error_types[etype] = error_types.get(etype, 0) + 1
            
            print("\n📊 Failure Breakdown by Type:")
            print("-" * 60)
            for etype, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                pct = count / len(self.failed_items) * 100
                print(f"  {etype:15} {count:4} ({pct:5.1f}%)")
            print("-" * 60)
            
            print("\n❌ Failed Items (first 10):")
            print("-" * 60)
            for item in self.failed_items[:10]:  # Show first 10
                print(f"  ID: {item['pinecone_id']}")
                print(f"  Category: {item.get('assigned_category', 'N/A')}")
                print(f"  Type: {item.get('error_type', 'UNKNOWN')}")
                print(f"  URL: {item['image_url'][:60]}...")
                print(f"  Error: {item['error'][:100]}...")
                print("-" * 60)
            
            if len(self.failed_items) > 10:
                print(f"  ... and {len(self.failed_items) - 10} more")
            
            # Save failed items to file
            failed_df = pd.DataFrame(self.failed_items)
            failed_path = "failed_ingestions.csv"
            failed_df.to_csv(failed_path, index=False)
            print(f"\n💾 Failed items saved to: {failed_path}")
            print(f"   Columns: pinecone_id, assigned_category, image_url, error_type, error")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch catalog ingestion from CSV with new data structure."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV file with required columns: pinecone_id, image_url, assigned_category, and metadata columns",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Backend API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of items to process concurrently (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="API timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per item (default: 3)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Maximum concurrent workers for RunPod API calls (default: 6)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    logger.add("ingestion.log", level="DEBUG", rotation="10 MB")
    
    # Create ingester
    ingester = CatalogIngester(
        api_base_url=args.api_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
    )
    
    try:
        # Read CSV
        df = ingester.read_csv(args.csv)
        
        # Process all items
        logger.info(f"Starting batch ingestion ({len(df)} products)")
        await ingester.process_batch(df, batch_size=args.batch_size)
        
        # Print summary
        ingester.print_summary()
        
        # Exit with error code if any failures
        sys.exit(0 if ingester.failed == 0 else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
