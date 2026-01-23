"use client";

import { useState, useRef } from "react";
import { Upload, X, CheckCircle, Loader2, AlertCircle, FileJson } from "lucide-react";
import { upsertCatalogItem } from "@/lib/api";
import { cn } from "@/lib/utils";

export function CatalogManager() {
  const [skuId, setSkuId] = useState("");
  const [category, setCategory] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [attributesJson, setAttributesJson] = useState("");
  const [attributesError, setAttributesError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const categories = [
    "chair", "2-seater-sofa", "l-shape-sofa", "sofa",
    "bed", "bedspread", "pillow", "mattresses",
    "service-table", "center-table", "side-table", "console",
    "dressing-table", "comforter", "tv-table", "dining-table",
    "storage-box", "carpet", "flower-pot-and-plant", "statue-and-antique",
    "laundry-basket", "candle", "vase", "flower",
    "wall-clock", "shelve", "decorative-hanger", "lighting",
    "lampshade", "floor-stand", "wall-lighting", "outdoor-lighting",
    "chandelier", "pendant-lighting", "coffee-maker", "cooking-appliance",
    "food-processor", "cooking-pot", "serving-utensil-and-tray",
    "cup", "plate", "chaise-lounge", "art-canvas", "office-table", "office-chair"
  ];

  const validateImageDimensions = (file: File): Promise<boolean> => {
    return new Promise((resolve) => {
      const img = new Image();
      const objectUrl = URL.createObjectURL(file);

      img.onload = () => {
        URL.revokeObjectURL(objectUrl);
        const isValid = img.width >= 400 && img.height >= 400;

        if (!isValid) {
          setError(
            `Image too small (${img.width}×${img.height}px). ` +
            `Minimum 400×400px required for quality detection and embeddings.`
          );
        }
        resolve(isValid);
      };

      img.onerror = () => {
        URL.revokeObjectURL(objectUrl);
        setError("Failed to load image. Please try another file.");
        resolve(false);
      };

      img.src = objectUrl;
    });
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        setError("Please select an image file");
        return;
      }

      // Validate dimensions
      const isValid = await validateImageDimensions(file);
      if (!isValid) {
        return; // Error already set
      }

      setImage(file);
      setError(null);
      setSuccess(null);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      // Validate dimensions
      const isValid = await validateImageDimensions(file);
      if (!isValid) {
        return; // Error already set
      }

      setImage(file);
      setError(null);
      setSuccess(null);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleRemoveImage = () => {
    setImage(null);
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleAttributesChange = (value: string) => {
    setAttributesJson(value);
    setAttributesError(null);
    if (value.trim()) {
      try {
        JSON.parse(value);
      } catch (e) {
        setAttributesError("Invalid JSON format");
      }
    }
  };

  const parseAttributes = (): Record<string, any> | undefined => {
    if (!attributesJson.trim()) {
      return undefined;
    }
    try {
      const parsed = JSON.parse(attributesJson);
      return typeof parsed === "object" && parsed !== null ? parsed : undefined;
    } catch {
      return undefined;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!skuId.trim()) {
      setError("SKU ID is required");
      return;
    }

    if (!category) {
      setError("Category is required");
      return;
    }

    if (!image) {
      setError("Product image is required");
      return;
    }

    if (attributesError) {
      setError("Please fix the JSON format in attributes");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const attributes = parseAttributes();
      const response = await upsertCatalogItem(
        skuId.trim(),
        category,
        image,
        attributes
      );
      setSuccess(
        `Product "${response.sku_id}" added successfully! Image ID: ${response.image_id}`
      );
      setSkuId("");
      setCategory("");
      setImage(null);
      setPreview(null);
      setAttributesJson("");
      setAttributesError(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to add product to catalog"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-8 border border-neutral-200">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* SKU ID */}
        <div>
          <label className="block text-sm font-medium text-neutral-700 mb-2">
            SKU ID <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            value={skuId}
            onChange={(e) => setSkuId(e.target.value)}
            placeholder="e.g., BED-001"
            required
            className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
          />
        </div>

        {/* Category */}
        <div>
          <label className="block text-sm font-medium text-neutral-700 mb-2">
            Category <span className="text-red-500">*</span>
          </label>
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            required
            className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
          >
            <option value="">Select a category</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat.charAt(0).toUpperCase() + cat.slice(1)}
              </option>
            ))}
          </select>
        </div>

        {/* Attributes JSON */}
        <div>
          <label className="block text-sm font-medium text-neutral-700 mb-2">
            Attributes (JSON) <span className="text-neutral-500 text-xs">(Optional)</span>
          </label>
          <div className="relative">
            <textarea
              value={attributesJson}
              onChange={(e) => handleAttributesChange(e.target.value)}
              placeholder='{"color": "brown", "material": "wood", "dimensions": {"width": 100, "height": 200}}'
              rows={4}
              className={cn(
                "w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none font-mono text-sm",
                attributesError
                  ? "border-red-300 bg-red-50"
                  : "border-neutral-300"
              )}
            />
            {attributesJson && (
              <div className="absolute top-2 right-2">
                <FileJson className="w-4 h-4 text-neutral-400" />
              </div>
            )}
          </div>
          {attributesError && (
            <p className="mt-1 text-sm text-red-600">{attributesError}</p>
          )}
          {attributesJson && !attributesError && (
            <p className="mt-1 text-sm text-green-600 flex items-center space-x-1">
              <CheckCircle className="w-4 h-4" />
              <span>Valid JSON</span>
            </p>
          )}
          <p className="mt-1 text-xs text-neutral-500">
            Enter product attributes as JSON (e.g., color, material, dimensions)
          </p>
        </div>

        {/* Image Upload */}
        <div>
          <label className="block text-sm font-medium text-neutral-700 mb-2">
            Product Image <span className="text-red-500">*</span>
          </label>
          {preview ? (
            <div className="relative group">
              <div className="relative w-full h-64 rounded-lg overflow-hidden border-2 border-neutral-200">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-full object-contain bg-neutral-50"
                />
                <button
                  type="button"
                  onClick={handleRemoveImage}
                  className="absolute top-2 right-2 p-2 bg-white rounded-full shadow-lg opacity-0 group-hover:opacity-100 transition-opacity hover:bg-neutral-100"
                >
                  <X className="w-5 h-5 text-neutral-700" />
                </button>
              </div>
            </div>
          ) : (
            <div
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              className="relative border-2 border-dashed border-neutral-300 rounded-lg p-12 text-center cursor-pointer hover:border-primary-400 hover:bg-primary-50/50 transition-colors"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
              <Upload className="w-12 h-12 text-neutral-400 mx-auto mb-4" />
              <p className="text-neutral-600 font-medium">
                Click to upload or drag and drop
              </p>
              <p className="text-sm text-neutral-500 mt-1">
                PNG, JPG, WEBP up to 15MB (minimum 400×400px recommended)
              </p>
            </div>
          )}
        </div>

        {/* Success Message */}
        {success && (
          <div className="p-4 bg-green-50 border border-green-200 rounded-lg flex items-start space-x-3">
            <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-green-700">{success}</p>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className={cn(
            "w-full py-3 px-6 rounded-lg font-semibold text-white transition-colors flex items-center justify-center space-x-2",
            loading
              ? "bg-neutral-300 cursor-not-allowed"
              : "bg-primary-600 hover:bg-primary-700 shadow-md hover:shadow-lg"
          )}
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Adding to Catalog...</span>
            </>
          ) : (
            <span>Add to Catalog</span>
          )}
        </button>
      </form>
    </div>
  );
}

