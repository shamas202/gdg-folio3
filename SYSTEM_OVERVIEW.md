# Interior Visual Search — Complete System Overview

## 1. What Is This System?

A visual product search engine for interior design e-commerce. A user uploads a photo of a room, the system detects furniture objects in the image, the user clicks on any detected object, and the system returns a ranked list of visually similar products from the catalog.

The system is fully local — no GPU required, no RunPod, no external detection API. Everything runs on CPU using lightweight models.

---

## 2. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Object Detection | YOLO11n (ultralytics) | Detect furniture in room photos — CPU, ~2.6 MB model |
| Embedding | Google Gemini `gemini-embedding-2-preview` | 3072-dimensional multimodal image vectors |
| Vector Database | Pinecone (serverless) | Cosine similarity search, namespaced by category |
| Backend | FastAPI (Python 3.11+) | Async REST API, CORS-enabled |
| Frontend | Next.js 14 + Tailwind CSS | React UI with drag-and-drop upload and live detection overlay |
| Image Processing | Pillow + NumPy | Image loading, validation, bbox cropping |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (Next.js)                   │
│   Upload → Detection Overlay → Click Object → Results   │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (localhost:3000 → 8001)
┌────────────────────────▼────────────────────────────────┐
│                   FastAPI Backend                        │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ /detect-and │  │   /search    │  │ /catalog/add  │  │
│  │  -segment   │  │              │  │               │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘  │
│         │                │                   │          │
│  ┌──────▼──────────────────────────────────▼──────┐    │
│  │            YOLO11n Detection Service            │    │
│  │         (YOLODetectionService — CPU)            │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │          PreprocessingService (tight crop)       │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │       GeminiEmbeddingService (3072-dim)          │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │       PineconeVectorRepository (upsert/query)   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Supported Product Categories

YOLO11n is trained on the COCO dataset (80 classes). The system maps a subset of those classes to internal category names used as Pinecone namespaces.

| User/CSV Category | YOLO COCO Class | Pinecone Namespace |
|-------------------|-----------------|--------------------|
| `chair` | chair | chair |
| `couch` / `sofa` | couch | couch / sofa |
| `bed` | bed | bed |
| `dining-table` | dining table | dining-table |
| `tv` | tv | tv |
| `clock` / `wall-clock` | clock | clock / wall-clock |
| `vase` | vase | vase |
| `laptop` | laptop | laptop |
| `tennis-racket` | tennis racket | tennis-racket |

Any object not in this list is silently filtered out, even if YOLO detects it.

---

## 5. Ingestion Pipeline (Adding Products to Catalog)

Products are added to the Pinecone vector database. There are two ways to do this.

### 5A. Bulk Ingestion via CSV Script

**File:** `ingest.py`

**CSV Format (required columns):**
```csv
image_url,product_name,category
https://example.com/chair.jpg,Oak Accent Chair,chair
https://example.com/sofa.jpg,Grey Linen Sofa,couch
```

**Pipeline steps for each row:**

```
Step 1 — Validate category
   → Check if category is in SUPPORTED_CATEGORIES
   → If not: mark as failed, skip

Step 2 — Download image
   → HTTP GET with browser-like User-Agent headers
   → 20s timeout
   → Convert to RGB PIL Image

Step 3 — YOLO detection
   → Run YOLO11n inference on full image
   → Filter results to the matching YOLO class name
   → Filter by confidence threshold (default: 0.15)
   → Select the highest-confidence bbox
   → If no bbox found: mark as failed, skip

Step 4 — Tight bbox crop
   → Crop image exactly to (x1, y1, x2, y2)
   → No padding, no masking, no white fill
   → Reject crop if smaller than 50×50px

Step 5 — Gemini embedding
   → Convert crop to JPEG bytes
   → Call Gemini API: embed_content(model, image, task=SEMANTIC_SIMILARITY)
   → Receive 3072-dimensional vector
   → L2-normalize the vector

Step 6 — Pinecone upsert
   → pinecone_id = md5(image_url)[:16]   ← deterministic, idempotent
   → Metadata stored: { image_url, product_name, category }
   → Upsert into namespace = category
   → (skipped if --dry-run flag is used)
```

**Command:**
```bash
python ingest.py --csv data/products.csv
python ingest.py --csv data/products.csv --max-workers 4 --confidence 0.15 --dry-run
```

**CLI Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | required | Input CSV path |
| `--max-workers` | 4 | Concurrent threads |
| `--confidence` | 0.15 | YOLO detection confidence threshold |
| `--failures-csv` | `failed_ingestions.csv` | Output path for failed rows |
| `--dry-run` | off | Skip Pinecone upsert |
| `--log-level` | INFO | DEBUG / INFO / WARNING / ERROR |

**Failure handling:**
- Every failed row is written to `--failures-csv` with `pinecone_id`, `image_url`, and `error`
- Failed rows can be re-ingested by fixing the issue and re-running with a filtered CSV
- Full debug log is always written to `ingestion.log`

---

### 5B. Single Product Ingestion via UI

**Page:** `http://localhost:3000/add`

**Endpoint:** `POST /api/v1/catalog/add` (JSON body)

```json
{
  "image_url": "https://example.com/chair.jpg",
  "product_name": "Oak Accent Chair",
  "category": "chair"
}
```

**Pipeline steps:**
```
Step 1 — Validate input
   → product_name must not be empty
   → category must be in SUPPORTED_CATEGORIES
   → image_url must be a valid HTTP/HTTPS URL (validated by Pydantic HttpUrl)

Step 2 — Download image
   → httpx.AsyncClient (async, 15s timeout, follows redirects)
   → Raise HTTP 400 if download fails

Step 3 — Validate image
   → Minimum dimension: 300×300px
   → Contrast check: std deviation ≥ 1.0 (rejects blank/uniform images)

Step 4 — YOLO detection
   → category_hint passed → only matching class returned
   → Pick largest bbox area among matching detections
   → Fallback: use full image if no object detected

Step 5 — Tight crop + Gemini embed + Pinecone upsert
   → Same as bulk ingestion steps 4–6 above
   → pinecone_id = md5(image_url)[:16]
   → image_url stored in metadata (appears in search results)

Response:
{
  "pinecone_id": "a1b2c3d4e5f6g7h8",
  "success": true,
  "message": "'Oak Accent Chair' added. Detected 'chair' at 82% confidence."
}
```

---

## 6. Search Pipeline (Finding Similar Products)

### Step 1 — Upload room photo (frontend)

User drags and drops or clicks to upload an image.

- Frontend validates: file type (image/*), minimum 400×400px
- Images larger than 1920px are client-side resized before upload
- The image is sent to the detection endpoint automatically on upload

### Step 2 — Object detection (`POST /api/v1/detect-and-segment`)

```
Request: multipart/form-data { image: File }

Backend:
  → Load + validate image (ImageIOService)
  → YOLO11n detects all objects (no category filter)
  → For each detection:
       → BBoxSegmentationService creates a rectangular mask
       → Mask encoded as base64 PNG
  → Returns list of detected objects with:
       { category, bbox, score, object_id, mask_base64 }
```

### Step 3 — Detection overlay (frontend)

- Detected objects rendered as clickable bounding boxes over the room photo
- Each box is labeled with category name
- User hovers to preview the detected region, clicks to trigger search

### Step 4 — Search by object (`POST /api/v1/search`)

When user clicks a detected object, the frontend sends the bbox coordinates (already known from Step 2) — this is the **fast path** that skips re-detection.

```
Request: multipart/form-data {
  image: File,
  assigned_category: "chair",
  bbox_x1: 120, bbox_y1: 80, bbox_x2: 340, bbox_y2: 310,
  top_k: 50
}

Backend (fast path — bbox provided):
  → Load image
  → Clamp bbox to image bounds
  → Tight crop: img.crop(x1, y1, x2, y2)
  → Gemini embed: embed_crops({"tight": crop}) → 3072-dim vector
  → Pinecone query: namespace = category, top_k = 50
  → Deduplicate by pinecone_id (keep highest score)
  → Return ranked hits

Backend (slow path — no bbox provided):
  → Load image
  → YOLO detect (with optional category_hint)
  → Pick best detection (largest matching bbox, fallback to largest overall)
  → Continue from crop step above
```

### Step 5 — Results display (frontend)

- Results shown in right panel (30% of screen width)
- Each card shows: product image, product name, category, match score (%)
- Click any card to open full-size image modal with match percentage
- Top result tagged as "Best Match"

---

## 7. Image Quality Checks

### During ingestion (both bulk and UI)

| Check | Threshold | Action |
|-------|-----------|--------|
| Minimum image size | 300×300px | Reject — raise 400 |
| Maximum image size | 4096px | Auto-resize (thumbnail) |
| Maximum file size | 10 MB | Reject — raise 400 |
| Contrast (std dev) | ≥ 1.0 | Reject — blank/uniform image |
| Contrast (std dev) | ≥ 10.0 | Warn in logs (low contrast) |
| Crop minimum size | 50×50px | Reject — degenerate bbox |

### During search (user upload)

- Same size, format, and contrast checks apply
- No ROA (ratio-of-area) check — user room photos vary in framing
- Frontend also validates minimum 400×400px before sending to backend

---

## 8. Pinecone Data Model

### Index configuration
- **Dimensions:** 3072
- **Metric:** cosine
- **Type:** serverless (AWS us-east-1 by default)

### Namespaces
Each category is stored in its own Pinecone namespace. This means search queries only scan vectors in the relevant namespace — faster and more accurate.

```
Namespaces:
  chair | couch | sofa | bed | dining-table |
  tv | clock | wall-clock | vase | laptop | tennis-racket
```

### Vector record structure
```json
{
  "id": "a1b2c3d4e5f67890",        ← md5(image_url)[:16]
  "values": [0.012, -0.034, ...],   ← 3072-dim L2-normalized float32
  "metadata": {
    "image_url": "https://...",
    "product_name": "Oak Accent Chair",
    "category": "chair"
  }
}
```

### Idempotency
`pinecone_id = md5(image_url)[:16]` — the same URL always produces the same ID. Re-ingesting the same product overwrites the existing vector (Pinecone upsert). This means the ingestion script is safe to re-run.

---

## 9. Gemini Embedding Details

**Model:** `models/gemini-embedding-2-preview`
**Output:** 3072 dimensions
**Task type:** `SEMANTIC_SIMILARITY`

**How it works:**
- Crop is converted to JPEG bytes (quality 95)
- Sent to Gemini API as an image Part
- Returned 3072-dim float32 vector is L2-normalized
- If multiple crops are sent (e.g. tight + medium), vectors are averaged then re-normalized

**Retry logic:**
- Up to 4 retries on rate-limit / quota / 429 / 503 / timeout errors
- Exponential backoff starting at 1.0s, capped at 30s
- Non-retryable errors (bad request, auth) fail immediately

---

## 10. Dependency Injection (Container Pattern)

All services are constructed once at startup and injected via FastAPI's `Depends` mechanism.

```
Container (frozen dataclass)
├── settings          ← Settings (loaded from .env)
├── image_io          ← ImageIOService
├── preprocessing     ← PreprocessingService
├── detection         ← YOLODetectionService (YOLO11n)
├── segmentation      ← BBoxSegmentationService
├── embedding         ← GeminiEmbeddingService
├── vectors           ← PineconeVectorRepository
└── search_service    ← SearchService (orchestrator)
```

**Startup sequence (lifespan):**
```
1. Container.from_settings(settings)
   → Instantiates all services
   → YOLO11n model loaded (downloads yolo11n.pt on first run)
2. container.start()
   → vectors.ensure_index()  ← creates Pinecone index if it doesn't exist
   → embedding.load()        ← initializes Gemini API client
3. app.state.container = container
```

---

## 11. API Endpoints Reference

### `POST /api/v1/detect-and-segment`
Detect all supported objects in an image and return bounding boxes + masks.

- **Input:** `multipart/form-data { image: File }`
- **Output:**
  ```json
  {
    "objects": [
      {
        "object_id": 0,
        "category": "chair",
        "score": 0.87,
        "bbox": { "x1": 120, "y1": 80, "x2": 340, "y2": 310 },
        "mask_base64": "iVBORw...",
        "mask_polygon": null
      }
    ],
    "image_width": 1280,
    "image_height": 720
  }
  ```

### `POST /api/v1/search`
Search for similar products using a room photo.

- **Input:** `multipart/form-data`
  - `image` (required) — room photo file
  - `assigned_category` (optional) — filter to specific category
  - `top_k` (optional, default 50) — number of results
  - `bbox_x1/y1/x2/y2` (optional) — pre-computed bbox (skips re-detection)
- **Output:**
  ```json
  {
    "query_category": "chair",
    "hits": [
      {
        "pinecone_id": "a1b2c3d4e5f67890",
        "score": 0.94,
        "product_name": "Oak Accent Chair",
        "category": "chair",
        "image_url": "https://..."
      }
    ],
    "message": null
  }
  ```

### `POST /api/v1/catalog/add`
Add a single product to the catalog via image URL.

- **Input:** `application/json`
  ```json
  { "image_url": "...", "product_name": "...", "category": "chair" }
  ```
- **Output:**
  ```json
  { "pinecone_id": "...", "success": true, "message": "..." }
  ```

### `GET /api/v1/health`
Returns service status.

---

## 12. Configuration (`.env`)

```env
# Application
APP_NAME=Interior Visual Search
APP_ENV=dev
LOG_LEVEL=INFO

# Pinecone
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=interior-products-gemini
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_DIM=3072

# Google Gemini
GOOGLE_API_KEY=your_key_here
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2-preview

# Detection
DETECTION_CONFIDENCE_THRESHOLD=0.15

# Image validation
IMAGE_MIN_DIMENSION=300
IMAGE_MAX_DIMENSION=4096
IMAGE_MAX_SIZE_MB=10

# Search
SEARCH_CANDIDATE_MULTIPLIER=15

# Override for testing (optional)
PINECONE_API_KEY_OVERRIDE=
PINECONE_INDEX_OVERRIDE=
```

---

## 13. Frontend Pages and Components

### Pages
| Route | File | Description |
|-------|------|-------------|
| `/` | `app/page.tsx` | Home — room photo upload and visual search |
| `/add` | `app/add/page.tsx` | Add product to catalog via image URL |
| `/results` | `app/results/page.tsx` | Full-page search results grid |

### Key Components
| Component | Description |
|-----------|-------------|
| `SearchInterface.tsx` | Main search UI — upload, detection overlay, results panel (70/30 split) |
| `ObjectDetectionView.tsx` | Renders bbox overlays on room photo, handles click events |
| `SearchResults.tsx` | Grid view of product cards with match score badges |
| `AddProduct.tsx` | URL input + live preview + category dropdown + submit |
| `Navigation.tsx` | Top nav bar with Search and Add Product links |

### Frontend flow (step by step)
```
1. User opens localhost:3000
2. Drag-and-drop or click to upload room photo
3. Client-side validation (type, minimum 400×400px)
4. If image > 1920px: client-side resize before upload
5. POST to /api/v1/detect-and-segment
6. Bounding boxes rendered as clickable overlays on the image
7. User clicks a box
8. POST to /api/v1/search (bbox + category passed — fast path)
9. Results appear in right panel
10. User clicks a product card → full-size image modal
```

---

## 14. Project File Structure

```
image-recommendation/
│
├── app/                              # FastAPI backend
│   ├── main.py                       # App factory, CORS, error handlers
│   ├── api/v1/
│   │   ├── routes.py                 # Router registration
│   │   └── endpoints/
│   │       ├── health.py             # GET /health
│   │       ├── search.py             # POST /search
│   │       ├── detection.py          # POST /detect-and-segment
│   │       └── catalog.py            # POST /catalog/add
│   ├── core/
│   │   ├── config.py                 # Settings (pydantic-settings, .env)
│   │   ├── constants.py              # Image quality thresholds
│   │   ├── errors.py                 # BadRequest, DependencyError
│   │   ├── lifespan.py               # Startup/shutdown (container init)
│   │   └── logging.py                # Loguru configuration
│   ├── dependencies/
│   │   └── container.py              # DI container, get_container()
│   ├── models/
│   │   ├── domain.py                 # Detection, Segment dataclasses
│   │   └── schemas.py                # Pydantic request/response schemas
│   ├── repositories/
│   │   └── pinecone_repo.py          # upsert(), query(), ensure_index()
│   ├── services/
│   │   ├── detection.py              # DetectionService ABC
│   │   ├── detection_yolo.py         # YOLODetectionService (YOLO11n)
│   │   ├── embedding.py              # EmbeddingService ABC
│   │   ├── embedding_gemini.py       # GeminiEmbeddingService
│   │   ├── image_io.py               # Image load, validate, resize
│   │   ├── preprocessing.py          # clamp_bbox(), tight_crop()
│   │   ├── search_service.py         # Search orchestration (fast + slow path)
│   │   └── segmentation.py           # BBoxSegmentationService
│   └── utils/
│       ├── timing.py                 # @timed context manager for logging
│       └── hashing.py
│
├── frontend/                         # Next.js frontend
│   ├── app/
│   │   ├── layout.tsx                # Root layout with Navigation
│   │   ├── page.tsx                  # / — home search page
│   │   ├── add/page.tsx              # /add — catalog add page
│   │   └── results/page.tsx          # /results — full results grid
│   ├── components/
│   │   ├── Navigation.tsx
│   │   ├── SearchInterface.tsx
│   │   ├── ObjectDetectionView.tsx
│   │   ├── SearchResults.tsx
│   │   └── AddProduct.tsx
│   └── lib/
│       ├── api.ts                    # Typed fetch wrappers for all endpoints
│       └── imageUtils.ts             # Client-side image validation + resize
│
├── data/
│   └── demo.csv                      # 5-row sample CSV for testing ingestion
│
├── ingest.py                         # Bulk CSV ingestion script
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables (not committed)
├── .env.example                      # Template with all keys documented
└── README.md                         # Quick start guide
```

---

## 15. Running the System

### Prerequisites
- Python 3.11+
- Node.js 18+
- Pinecone account (serverless index, 3072-dim, cosine)
- Google AI Studio API key

### Backend
```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — add PINECONE_API_KEY and GOOGLE_API_KEY

# 4. Start server (port 8001)
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# API docs available at:
# http://localhost:8001/docs
```

### Frontend
```bash
cd frontend
npm install
npm run dev

# App available at:
# http://localhost:3000
```

### First run notes
- YOLO11n model (`yolo11n.pt`, ~2.6 MB) downloads automatically from Ultralytics on first startup
- On macOS, an SSL warning may appear during download — it retries and succeeds automatically
- The Pinecone index is created automatically if it does not exist yet

---

