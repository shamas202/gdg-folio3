# Interior Visual Search

A lightweight visual search system for interior product catalogs. Upload a room photo, click an object, and find similar products — powered by YOLO11n detection and Google Gemini embeddings.

---

## Architecture

```
Room photo → YOLO11n detect → tight bbox crop → Gemini embed (3072-dim) → Pinecone search → results
```

| Component | Technology |
|-----------|-----------|
| Detection | YOLO11n (CPU, no GPU required) |
| Embedding | `gemini-embedding-2-preview` (3072-dim) |
| Vector store | Pinecone (serverless, cosine similarity) |
| Backend | FastAPI (Python 3.11+) |
| Frontend | Next.js 14 + Tailwind CSS |

---

## Supported Categories

YOLO11n COCO classes mapped to search namespaces:

| Input category | YOLO class |
|----------------|-----------|
| `chair` | chair |
| `couch` / `sofa` | couch |
| `bed` | bed |
| `dining-table` | dining table |
| `tv` | tv |
| `clock` / `wall-clock` | clock |
| `vase` | vase |
| `laptop` | laptop |
| `tennis-racket` | tennis racket |

---

## Prerequisites

- Python 3.11+
- Node.js 18+
- [Pinecone](https://pinecone.io) account (serverless index)
- [Google AI Studio](https://aistudio.google.com) API key (for Gemini embeddings)

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/saaib8/image-recommendation.git
cd image-recommendation
```

### 2. Backend setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys (see Configuration section below)
```

### 4. Start backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

### 5. Start frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:3000`  
Backend API docs: `http://localhost:8001/docs`

---

## Configuration

All settings are loaded from `.env`. Copy `.env.example` to get started.

```env
# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=interior-products-gemini   # must be 3072-dim, cosine
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_DIM=3072

# Google Gemini
GOOGLE_API_KEY=your_google_api_key
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2-preview

# Detection
DETECTION_CONFIDENCE_THRESHOLD=0.15   # YOLO11n confidence (0.0–1.0)

# Image validation
IMAGE_MIN_DIMENSION=300    # minimum px on each side
IMAGE_MAX_DIMENSION=4096
IMAGE_MAX_SIZE_MB=10

# Search
SEARCH_CANDIDATE_MULTIPLIER=15
```

### Pinecone index setup

Create a serverless index in Pinecone dashboard with:
- **Dimensions**: 3072
- **Metric**: cosine
- **Cloud / Region**: aws / us-east-1 (or your preferred region)

---

## Catalog Ingestion

### Option A — via UI (single product)

1. Open `http://localhost:3000/add`
2. Paste a product image URL
3. Enter product name and select category
4. Click **Add to catalog**

The backend downloads the image, runs YOLO detection, crops, embeds with Gemini, and upserts to Pinecone. `pinecone_id` is `md5(image_url)[:16]` — re-adding the same URL is idempotent.

### Option B — bulk via CSV

Prepare a CSV with these columns:

```csv
image_url,product_name,category
https://example.com/chair.jpg,Oak Accent Chair,chair
https://example.com/sofa.jpg,Grey Linen Sofa,couch
```

Run the ingestion script:

```bash
# Basic
python ingest.py --csv data/products.csv

# With options
python ingest.py \
  --csv data/products.csv \
  --max-workers 4 \
  --confidence 0.15 \
  --log-level INFO \
  --failures-csv failed_ingestions.csv
```

**Parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | required | Path to input CSV |
| `--max-workers` | `4` | Concurrent threads |
| `--confidence` | `0.15` | YOLO detection confidence threshold |
| `--failures-csv` | `failed_ingestions.csv` | Output file for failed rows |
| `--dry-run` | off | Parse & validate only, skip embed/upsert |
| `--log-level` | `INFO` | DEBUG / INFO / WARNING / ERROR |

Failed rows are written to `--failures-csv` so you can inspect and re-run them without reprocessing the full dataset.

---

## API Endpoints

### `POST /api/v1/detect-and-segment`
Detect all objects in a room image.
- **Body**: `multipart/form-data` with `image` (file)
- **Returns**: list of detected objects with bounding boxes

### `POST /api/v1/search`
Search for similar products using a query image.
- **Body**: `multipart/form-data`
  - `image` — image file (required)
  - `assigned_category` — category hint (optional)
  - `top_k` — number of results, default `50`
  - `bbox_x1/y1/x2/y2` — pre-computed bbox to skip detection (optional)
- **Returns**: ranked list of matching products with `product_name`, `category`, `image_url`, `score`

### `POST /api/v1/catalog/add`
Add a single product to the catalog via its image URL.
- **Body**: JSON
  ```json
  {
    "image_url": "https://example.com/product.jpg",
    "product_name": "Oak Accent Chair",
    "category": "chair"
  }
  ```
- **Returns**: `{ "pinecone_id": "...", "success": true, "message": "..." }`

### `GET /api/v1/health`
Returns service health status.

---

## Project Structure

```
image-recommendation/
├── app/
│   ├── api/v1/endpoints/
│   │   ├── catalog.py        # POST /catalog/add
│   │   ├── detection.py      # POST /detect-and-segment
│   │   ├── health.py
│   │   └── search.py         # POST /search
│   ├── core/
│   │   ├── config.py         # Settings (loaded from .env)
│   │   ├── constants.py      # Image quality thresholds
│   │   └── errors.py
│   ├── dependencies/
│   │   └── container.py      # Dependency injection
│   ├── models/
│   │   ├── domain.py         # Detection, Segment dataclasses
│   │   └── schemas.py        # Pydantic request/response models
│   ├── repositories/
│   │   └── pinecone_repo.py  # Upsert & query
│   ├── services/
│   │   ├── detection_yolo.py     # YOLO11n detection
│   │   ├── embedding_gemini.py   # Gemini embedding
│   │   ├── image_io.py           # Image loading & validation
│   │   ├── preprocessing.py      # Tight bbox crop
│   │   ├── search_service.py     # Search orchestration
│   │   └── segmentation.py       # BBoxSegmentationService
│   └── main.py
├── frontend/
│   ├── app/
│   │   ├── add/page.tsx      # Add product page
│   │   ├── results/page.tsx  # Search results page
│   │   └── page.tsx          # Home / search page
│   ├── components/
│   │   ├── AddProduct.tsx
│   │   ├── Navigation.tsx
│   │   ├── SearchInterface.tsx
│   │   └── SearchResults.tsx
│   └── lib/
│       └── api.ts            # Typed API client
├── data/
│   └── demo.csv              # 5-row sample for testing
├── ingest.py                 # Bulk CSV ingestion script
├── requirements.txt
├── .env.example
└── README.md
```

---

## Pipeline Details

### Search flow
```
1. User uploads room photo
2. YOLO11n detects all objects (CPU, ~200ms)
3. User clicks on a detected object
4. Tight bbox crop — no masking, no padding
5. Gemini embed_crops() → 3072-dim vector (L2-normalized)
6. Pinecone query in the matching category namespace
7. Top-K results returned with product_name, category, image_url, score
```

### Ingestion flow
```
1. CSV row → download image from image_url
2. Validate: ≥300px each side, contrast std ≥ 1.0
3. YOLO11n detect (category-hint filtered)
4. Pick largest matching bbox; fall back to full image if none detected
5. Tight crop → Gemini embed → L2-normalize
6. Pinecone upsert: namespace = category, id = md5(image_url)[:16]
7. Failures logged to CSV for review
```

---

## Image Quality Requirements

| Check | Threshold |
|-------|-----------|
| Minimum dimension | 300px (width and height) |
| Maximum dimension | 4096px (auto-resized) |
| Maximum file size | 10 MB |
| Contrast (std dev) | ≥ 1.0 (rejects blank/uniform images) |
