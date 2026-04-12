# Product Visual Search

Upload an input image, pick an object, and get **visually similar products** from your catalog. The stack uses **YOLO11n** (detect + crop), **Google Gemini** embeddings (3072-d), and **Pinecone** (cosine similarity).

**Flow:** `room image → detect objects → crop → embed → vector search → ranked results`

| Layer | Tech |
|--------|------|
| Detection | YOLO11n (CPU-friendly) |
| Embeddings | `gemini-embedding-2-preview` |
| Vector DB | Pinecone (serverless, cosine, 3072 dims) |
| API | FastAPI |
| UI | Next.js + Tailwind |

**Categories** (YOLO/COCO–mapped): chair, couch/sofa, bed, dining-table, tv, clock, vase, laptop, tennis-racket.

---

## Prerequisites

- Python 3.11+
- Node.js 18+
- [Pinecone](https://www.pinecone.io/) account — index **3072** dimensions, **cosine** metric
- [Google AI Studio](https://aistudio.google.com/) API key for Gemini embeddings

---

## Run locally

### 1. Backend

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Fill PINECONE_* , GOOGLE_API_KEY , etc.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

API docs: **http://localhost:8001/docs**

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
# Default API URL matches backend on 8001; change NEXT_PUBLIC_API_URL if needed
npm run dev
```

App: **http://localhost:3000**

---

## Environment (`.env`)

Copy `.env.example` → `.env`. Minimum:

- `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` (3072-d, cosine index)
- `GOOGLE_API_KEY`
- Optional: `DETECTION_CONFIDENCE_THRESHOLD` (default in `.env.example` is `0.35`), image limits, `SEARCH_CANDIDATE_MULTIPLIER`

---

## Catalog data

- **Bulk CSV:** `image_url`, `product_name`, `category` — then:

  ```bash
  python ingest.py --csv data/products.csv
  ```

  Use `--dry-run` to validate only; `--failures-csv` logs failed rows (default `failed_ingestions.csv`).

- **Single product:** use the UI at `/add` or `POST /api/v1/catalog/add` (see `/docs`).

---

## API (summary)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/detect-and-segment` | Detect objects in an uploaded image |
| `POST` | `/api/v1/search` | Similar products (image + optional bbox / category) |
| `POST` | `/api/v1/catalog/add` | Add one product by image URL |

Details and schemas: **http://localhost:8001/docs**
