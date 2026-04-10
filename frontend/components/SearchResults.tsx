"use client";

import { useState } from "react";
import { SearchHit } from "@/lib/api";
import { Package, Star, Tag } from "lucide-react";
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
        <Link
          href="/"
          className="text-primary-600 hover:text-primary-700 underline font-medium"
        >
          Try a different search
        </Link>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-neutral-900 mb-1">
          Search Results
        </h1>
        <p className="text-neutral-600">
          Found {hits.length} product{hits.length !== 1 ? "s" : ""}
        </p>
      </div>

      <div
        className="overflow-y-scroll border border-neutral-100 rounded-lg p-2"
        style={{ height: "calc(100vh - 210px)" }}
      >
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 pb-4">
          {hits.map((hit, index) => (
            <ProductCard key={`${hit.pinecone_id}-${index}`} hit={hit} rank={index + 1} />
          ))}
        </div>
      </div>
    </div>
  );
}

function ProductCard({ hit, rank }: { hit: SearchHit; rank: number }) {
  const [imageError, setImageError] = useState(false);
  const [imageLoading, setImageLoading] = useState(true);

  const imageUrl = hit.image_url || null;
  const productName = hit.product_name || hit.name_english || hit.name_arabic || hit.pinecone_id;

  return (
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
        {/* Match score badge */}
        <div className="absolute top-2 right-2 bg-black/60 text-white px-2 py-1 rounded-md text-xs font-medium">
          {Math.round(hit.score * 100)}%
        </div>
      </div>
      <div className="p-4">
        <h3 className="font-semibold text-neutral-900 mb-1 line-clamp-2" title={productName}>
          {productName}
        </h3>
        {hit.category && (
          <div className="flex items-center space-x-1 text-sm text-neutral-500">
            <Tag className="w-3 h-3" />
            <span className="capitalize">{hit.category.replace(/-/g, " ")}</span>
          </div>
        )}
      </div>
    </div>
  );
}
