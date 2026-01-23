# 🔍 Image Search - Visual Product Discovery System

A production-ready visual search system for e-commerce catalogs, powered by RF-DETR detection, polygon segmentation, and multi-scale embeddings.

## 🌟 Features

- **3-Tier Category System**: Optimized processing for small (cups), medium (chairs), and large (sofas) products
- **Advanced Detection**: RF-DETR with polygon-based segmentation for precise object extraction
- **Multi-Scale Embeddings**: Dual-tower architecture (ViT + CLIP) for robust visual matching
- **Smart Validation**: Category-aware quality checks (blur, ROA, contrast)
- **Fast Search**: Bbox+mask passing eliminates redundant detection (50-70% faster)
- **White Background Masking**: Consistent embeddings across products
- **Visual Reranking**: Cross-encoder for fine-grained similarity scoring

---

## 📋 Prerequisites

### System Requirements
- Python 3.11+
- Node.js 18+ (for frontend)
- 8GB+ RAM (for embedding models)
- Internet connection (for RunPod API)

### External Services
- **RunPod Account**: For RF-DETR detection API ([runpod.ai](https://runpod.ai))
- **Pinecone Account**: For vector storage ([pinecone.io](https://pinecone.io))

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/saaib8/image-recommendation.git
cd image-recommendation
```

### 2. Backend Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use uv (faster)
pip install uv
uv pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy template and edit with your API keys
cp ENV_TEMPLATE.txt .env

# Required configurations in .env:
# - RUNPOD_API_URL, RUNPOD_API_KEY (for detection)
# - PINECONE_API_KEY, PINECONE_INDEX_NAME (for vector storage)
```

### 4. Create Pinecone Index

```bash
# In Pinecone dashboard, create index with:
# - Name: interior-products (or your choice)
# - Dimensions: 1024
# - Metric: cosine
# - Cloud: aws, Region: us-east-1
```

### 5. Start Backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend available at: `http://localhost:8000`

### 6. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend available at: `http://localhost:3000`

---

## 📦 Catalog Ingestion

### Prepare CSV

Your CSV must have these columns:
- `pinecone_id`: Unique product ID
- `assigned_category`: Product category (see Categories section)
- `image_url`: URL to product image
- `name_english`, `name_arabic`: Product names
- `price_amount`, `price_unit`: Pricing info
- `product_url`, `store_id`, `countries`, `store`: Metadata

### Run Ingestion

```bash
# Basic ingestion
.venv/bin/python3 ingest.py --csv data/products.csv

# With options
.venv/bin/python3 ingest.py \
  --csv data/products.csv \
  --batch-size 10 \
  --max-workers 6 \
  --log-level INFO \
  --api-url http://localhost:8000
```

### Parameters

- `--csv`: Path to CSV file (required)
- `--batch-size`: Batch size for progress tracking (default: 10)
- `--max-workers`: Concurrent async workers (default: 6)
- `--log-level`: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--api-url`: Backend URL (default: http://localhost:8000)

### Monitor Progress

```bash
# Watch ingestion log
tail -f ingestion.log

# Check failures
cat failed_ingestions.csv
```

---

## 📊 3-Tier Category System

### SMALL (10 categories) - 6% ROA, Blur ≥30, 4 crops
```
cup, plate, cooking-pot, coffee-maker, cooking-appliance, 
food-processor, candle, vase, flower, statue-and-antique
```

### MEDIUM (23 categories) - 8% ROA, Blur ≥25, 4 crops
```
chair, office-chair, side-table, console, center-table, 
service-table, tv-table, storage-box, shelve, pillow,
lighting, lampshade, wall-lighting, outdoor-lighting,
chandelier, pendant-lighting, floor-stand, decorative-hanger,
wall-clock, art-canvas, laundry-basket, serving-utensil-and-tray,
flower-pot-and-plant
```

### LARGE (13 categories) - 12% ROA, Blur ≥20, 9 crops
```
3-seater-sofa, 2-seater-sofa, l-shape-sofa, sofa, chaise-lounge,
bed, bedspread, mattresses, comforter, dressing-table,
office-table, dining-table, carpet
```

---

## 🎯 Validation Rules

### Image Quality (All Categories)
- **File size**: ≤ 15MB
- **Format**: JPEG, PNG, WebP
- **Dimensions**: ≥ 500×500 pixels (both width and height)
- **Auto-resize**: Images > 4096px resized automatically
- **Contrast**: Standard deviation ≥ 1.0

### Category-Specific (Ingestion Only)
- **Blur Detection**: 30 (small), 25 (medium), 20 (large) - Laplacian variance
- **ROA (Ratio of Area)**: 6% (small), 8% (medium), 12% (large)
- **Category Match**: Detected category must match assigned category

### Retrieval (User Search)
- **More lenient**: No ROA check, universal blur threshold (25)
- **Flexible**: Accepts any detected category

---

## 🔧 Configuration

### Test Mode

For testing validation without embedding generation:

```bash
# In .env
TEST_MODE=true   # Skip embedding & Pinecone (fast testing)
TEST_MODE=false  # Full production mode (default)
```

### Detection Settings

```bash
DETECTION_MODE=runpod              # Use RunPod API (recommended)
RUNPOD_API_URL=your_endpoint_url
RUNPOD_API_KEY=your_api_key
RUNPOD_CONFIDENCE_THRESHOLD=0.10   # Detection confidence (default: 0.10)
```

### Embedding Settings

```bash
ENABLE_MULTISCALE_EMBEDDING=true
EMBEDDING_SCALES=224,384,512
INSTANCE_MODEL_NAME=google/vit-base-patch16-224-in21k
SEMANTIC_MODEL_NAME=openai/clip-vit-base-patch32
```

### Search Settings

```bash
SEARCH_CANDIDATE_MULTIPLIER=15
MAX_CANDIDATE_K=5000
ENABLE_MULTISTAGE_RERANK=true
```

---

## 📡 API Endpoints

### Detection

**POST** `/api/v1/detect-and-segment`
- Upload image → Returns all detected objects with masks
- Used by frontend to show detection overlay

### Search

**POST** `/api/v1/search`
- Upload image (+ optional bbox/mask) → Returns similar products
- Supports both full detection and bbox-based search

Parameters:
- `image`: Image file (required)
- `assigned_category`: Category hint (optional)
- `top_k`: Number of results (default: 50)
- `bbox_x1, bbox_y1, bbox_x2, bbox_y2`: User-selected bbox (optional)
- `mask_polygon`: JSON polygon points (optional)

### Catalog

**POST** `/api/v1/catalog/upsert`
- Upload product image → Processes and indexes to Pinecone
- Used by ingestion script

---

## 🏗️ Project Structure

```
image-search-back/
├── app/
│   ├── api/v1/endpoints/     # FastAPI endpoints
│   ├── core/                 # Config, constants, errors
│   │   └── constants.py      # 3-tier categories & thresholds
│   ├── models/               # Data models & schemas
│   ├── services/             # Business logic
│   │   ├── detection_rfdetr.py      # RF-DETR detection
│   │   ├── segmentation_polygon.py  # Polygon segmentation
│   │   ├── preprocessing.py         # Cropping & masking
│   │   ├── embedding_multiscale.py  # Multi-scale embeddings
│   │   ├── search_service.py        # Search orchestration
│   │   └── rerank.py               # Visual reranking
│   ├── repositories/         # Pinecone integration
│   └── main.py              # FastAPI app entry point
├── frontend/                # Next.js React frontend
│   ├── app/                # Pages
│   ├── components/         # React components
│   └── lib/               # API client & utilities
├── data/                  # CSV files for ingestion
├── ingest.py             # Batch ingestion script
├── requirements.txt      # Python dependencies
├── ENV_TEMPLATE.txt      # Environment template
└── README.md            # This file
```

---

## 🔬 Pipeline Flow

### Ingestion (Catalog Building)
```
1. CSV → Download images
2. Quality validation (blur ≥30/25/20, ROA ≥6/8/12%)
3. RF-DETR detection → Category match required
4. Polygon segmentation
5. Crop generation (4 or 9 based on tier)
6. White background masking (tight crops)
7. Multi-scale embedding (1024-dim vector)
8. Pinecone upsert (category namespaces)
```

### Search (User Query)
```
1. Upload room photo
2. RF-DETR detects all objects
3. User clicks object → Pass bbox+mask
4. Skip re-detection (fast path!)
5. Crop & mask selected object
6. Generate embedding
7. Pinecone similarity search
8. Visual cross-encoder reranking
9. Return top 50 results
```

---

## 📈 Performance

### Ingestion Speed
- **Small/Medium objects**: ~8-12 seconds per product
- **Large objects**: ~15-20 seconds per product
- **With 6 workers**: ~2 seconds effective rate
- **4,000 products**: ~2-3 hours

### Search Speed
- **With bbox (object click)**: ~5-8 seconds
- **Without bbox (full detection)**: ~10-15 seconds
- **Bottleneck**: Pinecone vector retrieval (bandwidth dependent)

### Expected Success Rates
- **High-performing categories** (bed, chair, sofa): 70-80%
- **Medium categories** (cups, tables): 50-60%
- **Challenging categories** (textiles, lighting): 20-40%
- **Overall**: 55-65%

---

## 🐛 Troubleshooting

### Ingestion Failures

Check `failed_ingestions.csv` for breakdown:
- **NO_DETECTION**: Category mismatch or poor image quality
- **ROA**: Object too small in frame
- **BLUR**: Image too blurry
- **DOWNLOAD**: Network errors, 403 forbidden

### Slow Search

If vector search > 30 seconds:
- Check Pinecone region (closer = faster)
- Verify bandwidth to Pinecone
- Catalog size affects retrieval time

### Detection Issues

Common category confusions:
- Sofa sizes (2-seater ↔ 3-seater)
- Lighting types (lampshade ↔ pendant-lighting)
- Small tables (side-table ↔ console)

---

## 🔑 Key Configuration Files

- **ENV_TEMPLATE.txt**: All environment variables with explanations
- **app/core/constants.py**: Categories and thresholds
- **ingest.py**: Batch ingestion script
- **requirements.txt**: Python dependencies

---

## 📚 Additional Documentation

- Check `ENV_TEMPLATE.txt` for detailed configuration options
- Review `app/core/constants.py` for threshold tuning
- See `failed_ingestions.csv` after ingestion for quality insights

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 License

[Your License Here]

---

## 👥 Authors

- Built with ❤️ for visual product discovery

---

## 🙏 Acknowledgments

- RF-DETR for furniture detection
- OpenAI CLIP for semantic understanding
- Google ViT for instance-level features
- Pinecone for vector search infrastructure

