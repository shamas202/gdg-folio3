"use client";

import { useState } from "react";
import {
  CheckCircle,
  ImageIcon,
  Loader2,
  Link as LinkIcon,
  Tag,
  Type,
  Upload,
  XCircle,
} from "lucide-react";
import { addProduct, ApiError, SUPPORTED_CATEGORIES } from "@/lib/api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const CATEGORY_LABELS: Record<string, string> = {
  chair: "Chair",
  couch: "Couch",
  sofa: "Sofa",
  bed: "Bed",
  "dining-table": "Dining Table",
  tv: "TV",
  clock: "Clock",
  "wall-clock": "Wall Clock",
  vase: "Vase",
  laptop: "Laptop",
  "tennis-racket": "Tennis Racket",
};

type Status =
  | { type: "idle" }
  | { type: "loading" }
  | { type: "success"; pinecone_id: string; message: string }
  | { type: "error"; message: string };

function isValidUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function AddProduct() {
  const [imageUrl, setImageUrl] = useState("");
  const [productName, setProductName] = useState("");
  const [category, setCategory] = useState("");
  const [previewError, setPreviewError] = useState(false);
  const [status, setStatus] = useState<Status>({ type: "idle" });

  const urlValid = isValidUrl(imageUrl);
  const isValid = urlValid && productName.trim() && category;
  const isLoading = status.type === "loading";

  const handleUrlChange = (value: string) => {
    setImageUrl(value);
    setPreviewError(false);
    if (status.type !== "idle") setStatus({ type: "idle" });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isValid || isLoading) return;

    setStatus({ type: "loading" });
    try {
      const res = await addProduct(imageUrl.trim(), productName.trim(), category);
      setStatus({ type: "success", pinecone_id: res.pinecone_id, message: res.message });
    } catch (err) {
      const msg =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : "Unexpected error";
      setStatus({ type: "error", message: msg });
    }
  };

  return (
    <div className="max-w-xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-neutral-900">Add Product</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Provide an image URL and we&apos;ll detect the object, embed it with Gemini, and store it in the catalog.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">

        {/* ---- Image URL ---- */}
        <div>
          <label
            htmlFor="image_url"
            className="block text-sm font-medium text-neutral-700 mb-1.5"
          >
            Image URL
          </label>
          <div className="relative">
            <LinkIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-400 pointer-events-none" />
            <input
              id="image_url"
              type="url"
              value={imageUrl}
              onChange={(e) => handleUrlChange(e.target.value)}
              placeholder="https://example.com/product.jpg"
              className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-neutral-300 text-sm text-neutral-900 placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-primary-400 focus:border-transparent"
              disabled={isLoading}
            />
          </div>
        </div>

        {/* ---- Image preview ---- */}
        {urlValid && (
          <div className="rounded-xl overflow-hidden border border-neutral-200 bg-neutral-50">
            {previewError ? (
              <div className="flex flex-col items-center justify-center gap-2 py-10 text-neutral-400">
                <ImageIcon className="w-8 h-8" />
                <p className="text-xs">Preview unavailable</p>
              </div>
            ) : (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                key={imageUrl}
                src={imageUrl}
                alt="Product preview"
                className="w-full max-h-64 object-contain"
                onError={() => setPreviewError(true)}
              />
            )}
          </div>
        )}

        {/* ---- Product name ---- */}
        <div>
          <label
            htmlFor="product_name"
            className="block text-sm font-medium text-neutral-700 mb-1.5"
          >
            Product name
          </label>
          <div className="relative">
            <Type className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-400 pointer-events-none" />
            <input
              id="product_name"
              type="text"
              value={productName}
              onChange={(e) => setProductName(e.target.value)}
              placeholder="e.g. Oak Accent Chair"
              className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-neutral-300 text-sm text-neutral-900 placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-primary-400 focus:border-transparent"
              disabled={isLoading}
            />
          </div>
        </div>

        {/* ---- Category ---- */}
        <div>
          <label
            htmlFor="category"
            className="block text-sm font-medium text-neutral-700 mb-1.5"
          >
            Category
          </label>
          <div className="relative">
            <Tag className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-400 pointer-events-none" />
            <select
              id="category"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-neutral-300 text-sm text-neutral-900 focus:outline-none focus:ring-2 focus:ring-primary-400 focus:border-transparent bg-white appearance-none"
              disabled={isLoading}
            >
              <option value="" disabled>
                Select a category…
              </option>
              {SUPPORTED_CATEGORIES.map((cat) => (
                <option key={cat} value={cat}>
                  {CATEGORY_LABELS[cat] ?? cat}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* ---- Status feedback ---- */}
        {status.type === "success" && (
          <div className="flex items-start gap-3 rounded-lg border border-green-200 bg-green-50 p-4">
            <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 shrink-0" />
            <div className="text-sm">
              <p className="font-semibold text-green-800">Added to catalog</p>
              <p className="text-green-700 mt-0.5">{status.message}</p>
              <p className="text-green-600 mt-1 font-mono text-xs">
                ID: {status.pinecone_id}
              </p>
            </div>
          </div>
        )}

        {status.type === "error" && (
          <div className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4">
            <XCircle className="w-5 h-5 text-red-500 mt-0.5 shrink-0" />
            <div className="text-sm">
              <p className="font-semibold text-red-700">Failed to add product</p>
              <p className="text-red-600 mt-0.5">{status.message}</p>
            </div>
          </div>
        )}

        {/* ---- Submit ---- */}
        <button
          type="submit"
          disabled={!isValid || isLoading}
          className="w-full flex items-center justify-center gap-2 py-3 px-6 rounded-xl bg-primary-600 text-white font-semibold text-sm hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-400 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Processing…
            </>
          ) : (
            <>
              <Upload className="w-4 h-4" />
              Add to catalog
            </>
          )}
        </button>

        {isLoading && (
          <p className="text-center text-xs text-neutral-400">
            Downloading → detecting → embedding → upserting to Pinecone…
          </p>
        )}
      </form>
    </div>
  );
}
