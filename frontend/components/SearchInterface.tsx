"use client";

import { useState, useRef, useEffect } from "react";
import { Upload, X, Loader2, AlertCircle, Search as SearchIcon, Package } from "lucide-react";
import { detectAndSegmentObjects, searchProducts, ApiError, SegmentedObject, SearchHit } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ObjectDetectionView } from "./ObjectDetectionView";
import { SearchResults } from "./SearchResults";

export function SearchInterface() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [detectedObjects, setDetectedObjects] = useState<SegmentedObject[]>([]);
  const [imageWidth, setImageWidth] = useState<number>(0);
  const [imageHeight, setImageHeight] = useState<number>(0);
  const [detectionLoading, setDetectionLoading] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<SearchHit[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedProduct, setSelectedProduct] = useState<SearchHit | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    await processImageFile(file);
  };

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    await processImageFile(file);
  };

  const processImageFile = async (file: File) => {
    // Import preprocessing utilities
    const { preprocessImage, getImageDimensions, validateImageFile } = await import("@/lib/imageUtils");

    // Validate file type and size
    const validationError = validateImageFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    try {
      // Get original dimensions
      const dims = await getImageDimensions(file);

      // Preprocess if needed (resize large images)
      let processedFile = file;
      if (dims.width > 1920 || dims.height > 1920) {
        setError(null);
        setDetectionLoading(true);
        processedFile = await preprocessImage(file, 1920, 0.9);
        setDetectionLoading(false);
      }

      setImage(processedFile);
      setError(null);
      setSearchResults([]);
      setSelectedCategory(null);

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        const previewUrl = reader.result as string;
        setPreview(previewUrl);

        // Get image dimensions for display
        const img = new Image();
        img.onload = () => {
          setImageWidth(img.width);
          setImageHeight(img.height);
        };
        img.src = previewUrl;
      };
      reader.readAsDataURL(processedFile);

      // Automatically run detection and segmentation
      await runDetectionAndSegmentation(processedFile);
    } catch (err) {
      setError(`Failed to process image: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setDetectionLoading(false);
    }
  };

  const runDetectionAndSegmentation = async (imageFile: File) => {
    setDetectionLoading(true);
    setError(null);
    setDetectedObjects([]);

    try {
      const response = await detectAndSegmentObjects(imageFile);
      setDetectedObjects(response.objects);
      setImageWidth(response.image_width);
      setImageHeight(response.image_height);

      if (response.objects.length === 0) {
        setError("No objects detected in the image. Please try a different image.");
      }
    } catch (err) {
      let errorMessage = "Failed to detect objects";

      if (err instanceof ApiError) {
        errorMessage = err.message;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      }

      setError(errorMessage);
    } finally {
      setDetectionLoading(false);
    }
  };

  const handleObjectClick = async (object: SegmentedObject) => {
    setSearchLoading(true);
    setError(null);
    setSelectedCategory(object.category);

    try {
      if (!image) {
        setError("No image available");
        return;
      }

      // Search using the detected object's bbox and mask (faster - skips re-detection)
      const response = await searchProducts(image, {
        category: object.category,
        bbox: object.bbox,  // Pass bbox from detection
        mask_polygon: object.mask_polygon || undefined,  // Pass mask polygon if available
        top_k: 50,  // Display top 50 matches
      });

      setSearchResults(response.hits);

      if (response.hits.length === 0 && response.message) {
        setError(response.message);
      }
    } catch (err) {
      let errorMessage = "Failed to search products";

      if (err instanceof ApiError) {
        errorMessage = err.message;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      }

      setError(errorMessage);
      setSearchResults([]);
    } finally {
      setSearchLoading(false);
    }
  };

  const handleRemoveImage = () => {
    setImage(null);
    setPreview(null);
    setDetectedObjects([]);
    setSearchResults([]);
    setSelectedCategory(null);
    setImageWidth(0);
    setImageHeight(0);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  return (
    <div className="w-full h-full flex flex-col bg-white">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-neutral-200">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-neutral-900">Visual Search</h2>
            {preview && (
              <div className="text-sm text-neutral-600 mt-1">
                {detectedObjects.length > 0 ? (
                  <span>
                    Detected {detectedObjects.length} object{detectedObjects.length !== 1 ? "s" : ""}.
                    Click on any object to search for similar products.
                  </span>
                ) : detectionLoading ? (
                  <span>Processing image...</span>
                ) : null}
              </div>
            )}
          </div>
          {preview && (
            <button
              onClick={handleRemoveImage}
              className="p-2 hover:bg-neutral-100 rounded-lg transition-colors"
              title="Remove image"
            >
              <X className="w-5 h-5 text-neutral-600" />
            </button>
          )}
        </div>
      </div>

      {/* Main Content Area - 70/30 Split - Always Visible */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Side: Upload Area or Image with Detected Objects - 70% */}
        <div className="w-[70%] flex flex-col border-r border-neutral-200 overflow-hidden">
          {!preview ? (
            // Upload Area
            <div className="flex-1 flex items-center justify-center p-6">
              <div
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                className="relative w-full max-w-2xl border-2 border-dashed border-neutral-300 rounded-lg p-16 text-center cursor-pointer hover:border-primary-400 hover:bg-primary-50/50 transition-colors"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <Upload className="w-16 h-16 text-neutral-400 mx-auto mb-6" />
                <p className="text-lg text-neutral-600 font-medium mb-2">
                  Click to upload or drag and drop
                </p>
                <p className="text-sm text-neutral-500">
                  PNG, JPG, WEBP up to 15MB
                </p>
              </div>
            </div>
          ) : (
            // Image View
            <>
              <div className="flex-1 p-6 overflow-auto">
                <div className="relative h-full flex items-center justify-center">
                  <ObjectDetectionView
                    imageUrl={preview}
                    objects={detectedObjects}
                    imageWidth={imageWidth}
                    imageHeight={imageHeight}
                    onObjectClick={handleObjectClick}
                    loading={detectionLoading}
                  />
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <div className="flex-shrink-0 px-6 pb-4">
                  <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-start space-x-3">
                      <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <p className="text-sm text-red-700 whitespace-pre-line">{error}</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Right Side: Search Results - 30% - Always Visible */}
        <div className="w-[30%] flex flex-col overflow-hidden bg-neutral-50">
          <div className="flex-shrink-0 px-6 py-4 border-b border-neutral-200 bg-white">
            {selectedCategory ? (
              <div>
                <h3 className="text-lg font-bold text-neutral-900 mb-1">
                  Search Results
                </h3>
                <p className="text-sm text-neutral-600">
                  Showing results for: <span className="font-medium capitalize">{selectedCategory}</span>
                </p>
              </div>
            ) : (
              <div>
                <h3 className="text-lg font-bold text-neutral-900 mb-1">
                  Product Recommendations
                </h3>
              </div>
            )}
          </div>

          <div className="flex-1 overflow-y-auto">
            {searchLoading ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <Loader2 className="w-12 h-12 animate-spin text-primary-600 mx-auto mb-4" />
                  <p className="text-neutral-600">Searching for products...</p>
                </div>
              </div>
            ) : searchResults.length > 0 ? (
              <div className="p-4 space-y-3">
                {searchResults.map((hit, index) => (
                  <ProductCard
                    key={`${hit.pinecone_id}-${index}`}
                    hit={hit}
                    onImageClick={() => {
                      setSelectedProduct(hit);
                      setIsModalOpen(true);
                    }}
                  />
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full px-6">
                <div className="text-center">
                  <SearchIcon className="w-16 h-16 text-neutral-300 mx-auto mb-4" />
                  {!preview ? (
                    <>
                      <p className="text-lg text-neutral-600 font-medium mb-2">
                        Upload an Image
                      </p>
                      <p className="text-sm text-neutral-500">
                        You need to upload an image and then click on any product for recommendation
                      </p>
                    </>
                  ) : (
                    <>
                      <p className="text-lg text-neutral-600 font-medium mb-2">
                        No products selected
                      </p>
                      <p className="text-sm text-neutral-500">
                        On hover you can see the detected segmented part of an object and click on that the recommendation product visible here
                      </p>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Product Image Modal */}
      {isModalOpen && selectedProduct && (
        <ProductImageModal
          product={selectedProduct}
          onClose={() => {
            setIsModalOpen(false);
            setSelectedProduct(null);
          }}
        />
      )}
    </div>
  );
}

// Product Image Modal Component
function ProductImageModal({
  product,
  onClose
}: {
  product: SearchHit;
  onClose: () => void;
}) {
  const [imageError, setImageError] = useState(false);
  const [imageLoading, setImageLoading] = useState(true);
  const imageUrl = product.image_url || null;
  const productName = product.product_name || product.name_english || product.name_arabic || product.pinecone_id;

  // Close on Escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative bg-white rounded-lg shadow-2xl max-w-4xl max-h-[90vh] w-full m-4 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-neutral-200">
          <div>
            <h3 className="text-lg font-bold text-neutral-900">{productName}</h3>
            {product.category && (
              <p className="text-sm text-neutral-600 capitalize">{product.category.replace(/-/g, " ")}</p>
            )}
            <p className="text-sm font-medium text-primary-600 mt-1">
              {Math.round(product.score * 100)}% match
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-neutral-100 rounded-lg transition-colors"
            title="Close"
          >
            <X className="w-6 h-6 text-neutral-600" />
          </button>
        </div>

        {/* Image */}
        <div className="relative bg-neutral-100 flex items-center justify-center min-h-[400px] max-h-[70vh] overflow-auto">
          {imageUrl && !imageError ? (
            <>
              {imageLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-neutral-100 to-neutral-200">
                  <div className="text-center">
                    <Loader2 className="w-12 h-12 animate-spin text-primary-600 mx-auto mb-4" />
                    <p className="text-sm text-neutral-600">Loading image...</p>
                  </div>
                </div>
              )}
              <img
                src={imageUrl}
                alt={productName}
                className="max-w-full max-h-[70vh] w-auto h-auto object-contain"
                onLoad={() => setImageLoading(false)}
                onError={() => {
                  setImageError(true);
                  setImageLoading(false);
                }}
              />
            </>
          ) : (
            <div className="flex flex-col items-center justify-center p-12">
              <Package className="w-24 h-24 text-neutral-400 mb-4" />
              <p className="text-neutral-600">Image not available</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-neutral-200 bg-neutral-50">
          <div className="flex items-center justify-center">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 rounded-full bg-primary-500"></div>
              <span className="text-sm font-medium text-neutral-700">
                {Math.round(product.score * 100)}% match
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Product Card Component for Side Panel
function ProductCard({ hit, onImageClick }: { hit: SearchHit; onImageClick: () => void }) {
  const [imageError, setImageError] = useState(false);
  const [imageLoading, setImageLoading] = useState(true);
  const imageUrl = hit.image_url || null;
  const productName = hit.product_name || hit.name_english || hit.name_arabic || hit.pinecone_id;

  return (
    <div className="bg-white rounded-lg border border-neutral-200 overflow-hidden hover:shadow-md transition-shadow">
      <div className="flex gap-3 p-3">
        {/* Image - Clickable */}
        <div
          className="w-20 h-20 bg-neutral-100 flex-shrink-0 flex items-center justify-center relative overflow-hidden rounded cursor-pointer hover:opacity-90 transition-opacity"
          onClick={onImageClick}
        >
          {imageUrl && !imageError ? (
            <>
              {imageLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-neutral-100 to-neutral-200">
                  <div className="w-5 h-5 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
                </div>
              )}
              <img
                src={imageUrl}
                alt={productName}
                className="w-full h-full object-cover"
                onLoad={() => setImageLoading(false)}
                onError={() => {
                  setImageError(true);
                  setImageLoading(false);
                }}
              />
            </>
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-neutral-100 to-neutral-200">
              <Package className="w-6 h-6 text-neutral-400" />
            </div>
          )}
        </div>

        {/* Details */}
        <div className="flex-1 py-1 min-w-0 flex flex-col justify-center">
          <h4 className="font-semibold text-sm text-neutral-900 mb-1 truncate" title={productName}>
            {productName}
          </h4>
          {hit.category && (
            <p className="text-xs text-neutral-500 capitalize mb-1">
              {hit.category.replace(/-/g, " ")}
            </p>
          )}
          <p className="text-xs text-primary-600 font-medium">
            {Math.round(hit.score * 100)}% match
          </p>
        </div>
      </div>
    </div>
  );
}
