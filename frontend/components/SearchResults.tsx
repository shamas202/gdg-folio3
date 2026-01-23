"use client";

import { useState } from "react";
import { SearchHit } from "@/lib/api";
import { Package, Star, Tag, ExternalLink } from "lucide-react";
import Link from "next/link";

interface SearchResultsProps {
  hits: SearchHit[];
  message?: string | null;
}

export function SearchResults({ hits, message }: SearchResultsProps) {
  if (hits.length === 0) {
    return (
      <div className="text-center py-12">
        <Package className="w-16 h-16 text-neutral-300 mx-auto mb-4" />
        <p className="text-xl text-neutral-600 mb-2 font-semibold">
          No products found
        </p>
        {message && (
          <p className="text-base text-neutral-500 mb-4 max-w-md mx-auto">
            {message}
          </p>
        )}
        <div className="flex flex-col sm:flex-row gap-3 justify-center items-center">
          <Link
            href="/"
            className="text-primary-600 hover:text-primary-700 underline font-medium"
          >
            Try a different search
          </Link>
          <span className="text-neutral-400">or</span>
          <Link
            href="/catalog"
            className="text-primary-600 hover:text-primary-700 underline font-medium"
          >
            Add products to catalog
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-neutral-900 mb-2">
          Search Results
        </h1>
        <p className="text-neutral-600">
          Found {hits.length} product{hits.length !== 1 ? "s" : ""}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {hits.map((hit, index) => (
          <ProductCard key={`${hit.pinecone_id}-${index}`} hit={hit} rank={index + 1} />
        ))}
      </div>
    </div>
  );
}

function ProductCard({ hit, rank }: { hit: SearchHit; rank: number }) {
  const [imageError, setImageError] = useState(false);
  const [imageLoading, setImageLoading] = useState(true);

  const imageUrl = hit.image_url || null;
  const productName = hit.name_english || hit.name_arabic || hit.pinecone_id;
  const hasPrice = hit.price_amount !== null && hit.price_amount !== undefined;

  const CardContent = (
    <div className="bg-white rounded-lg border border-neutral-200 overflow-hidden hover:shadow-lg transition-shadow">
      <div className="aspect-square bg-neutral-100 flex items-center justify-center relative overflow-hidden">
        {imageUrl && !imageError ? (
          <>
            {imageLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-neutral-100 to-neutral-200">
                <Package className="w-12 h-12 text-neutral-300 animate-pulse" />
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
            <Package className="w-24 h-24 text-neutral-400" />
          </div>
        )}
        {rank === 1 && (
          <div className="absolute top-2 left-2 bg-primary-600 text-white px-2 py-1 rounded-md text-xs font-semibold flex items-center space-x-1">
            <Star className="w-3 h-3 fill-current" />
            <span>Best Match</span>
          </div>
        )}
      </div>
      <div className="p-4">
        <div className="flex items-start justify-between mb-2">
          <div className="flex-1">
            <h3 className="font-semibold text-neutral-900 mb-1 line-clamp-2" title={productName}>
              {productName}
            </h3>
            {hit.category && (
              <div className="flex items-center space-x-1 text-sm text-neutral-600 mb-2">
                <Tag className="w-3 h-3" />
                <span className="capitalize">{hit.category.replace(/-/g, ' ')}</span>
              </div>
            )}
          </div>
        </div>

        {/* Price */}
        {hasPrice && (
          <div className="mb-3">
            <span className="text-lg font-bold text-primary-600">
              {hit.price_amount} {hit.price_unit || 'SAR'}
            </span>
          </div>
        )}

        {/* Store and Countries */}
        {(hit.store || (hit.countries && hit.countries.length > 0)) && (
          <div className="mt-3 pt-3 border-t border-neutral-100">
            <div className="text-xs text-neutral-500 space-y-1">
              {hit.store && (
                <div className="flex justify-between">
                  <span>Store:</span>
                  <span className="text-neutral-700 font-medium">{hit.store}</span>
                </div>
              )}
              {hit.countries && hit.countries.length > 0 && (
                <div className="flex justify-between">
                  <span>Available in:</span>
                  <span className="text-neutral-700 font-medium">{hit.countries.join(', ')}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Product URL Link */}
        {hit.product_url && (
          <a
            href={hit.product_url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-3 flex items-center justify-center space-x-2 text-sm text-primary-600 hover:text-primary-700 font-medium transition-colors"
          >
            <span>View Product</span>
            <ExternalLink className="w-4 h-4" />
          </a>
        )}
      </div>
    </div>
  );

  // If there's a product URL, make the whole card clickable
  if (hit.product_url) {
    return (
      <a href={hit.product_url} target="_blank" rel="noopener noreferrer" className="block">
        {CardContent}
      </a>
    );
  }

  return CardContent;
}

