// Backend FastAPI runs on port 8001
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

// =====================
// Types
// =====================

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface SegmentedObject {
  object_id: number;
  category: string;
  score: number;
  bbox: BBox;
  mask_base64: string;
  mask_polygon: number[][] | null;
}

export interface DetectionSegmentationResponse {
  objects: SegmentedObject[];
  image_width: number;
  image_height: number;
}

export interface SearchHit {
  pinecone_id: string;
  score: number;
  image_url: string | null;
  product_name: string | null;
  name_english: string | null;
  name_arabic: string | null;
  category: string | null;
  // Legacy fields (may be null for demo-ingested products)
  product_url: string | null;
  price_amount: number | null;
  price_unit: string | null;
  is_active: boolean | null;
  store_id: number | null;
  countries: string[] | null;
  store: string | null;
}

export interface SearchResponse {
  query_category: string | null;
  hits: SearchHit[];
  message: string | null;
}

export interface SearchOptions {
  category?: string | null;
  bbox?: BBox;
  mask_polygon?: number[][] | null;
  top_k?: number;
}

export interface CatalogAddResponse {
  pinecone_id: string;
  success: boolean;
  message: string;
}

export const SUPPORTED_CATEGORIES = [
  "chair",
  "couch",
  "sofa",
  "bed",
  "dining-table",
  "tv",
  "clock",
  "wall-clock",
  "vase",
  "laptop",
  "tennis-racket",
] as const;

export type SupportedCategory = (typeof SUPPORTED_CATEGORIES)[number];

// =====================
// Error class
// =====================

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const json = await res.json();
      detail = json?.detail ?? JSON.stringify(json);
    } catch {
      detail = await res.text();
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

// =====================
// API functions
// =====================

/**
 * POST /api/v1/detect-and-segment
 * Detects objects in a room image using YOLO11n and returns bbox masks.
 */
export async function detectAndSegmentObjects(
  imageFile: File,
): Promise<DetectionSegmentationResponse> {
  const form = new FormData();
  form.append("image", imageFile, imageFile.name);

  const res = await fetch(`${API_BASE}/api/v1/detect-and-segment`, {
    method: "POST",
    body: form,
  });

  return handleResponse<DetectionSegmentationResponse>(res);
}

/**
 * POST /api/v1/search
 * Search for similar products using a query image.
 * If bbox is provided (from a previous detection), detection is skipped.
 */
export async function searchProducts(
  imageFile: File,
  options: SearchOptions = {},
): Promise<SearchResponse> {
  const form = new FormData();
  form.append("image", imageFile, imageFile.name);

  if (options.category) {
    form.append("assigned_category", options.category);
  }

  form.append("top_k", String(options.top_k ?? 50));

  if (options.bbox) {
    form.append("bbox_x1", String(options.bbox.x1));
    form.append("bbox_y1", String(options.bbox.y1));
    form.append("bbox_x2", String(options.bbox.x2));
    form.append("bbox_y2", String(options.bbox.y2));
  }

  if (options.mask_polygon) {
    form.append("mask_polygon", JSON.stringify(options.mask_polygon));
  }

  const res = await fetch(`${API_BASE}/api/v1/search`, {
    method: "POST",
    body: form,
  });

  return handleResponse<SearchResponse>(res);
}

/**
 * POST /api/v1/catalog/add
 * Add a product to the catalog via its image URL.
 * Backend downloads the image, runs YOLO → tight crop → Gemini embed → Pinecone upsert.
 */
export async function addProduct(
  imageUrl: string,
  productName: string,
  category: string,
): Promise<CatalogAddResponse> {
  const res = await fetch(`${API_BASE}/api/v1/catalog/add`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_url: imageUrl,
      product_name: productName,
      category,
    }),
  });

  return handleResponse<CatalogAddResponse>(res);
}
