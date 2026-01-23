"use client";

import { CatalogManager } from "@/components/CatalogManager";

export default function CatalogPage() {
  return (
    <div className="container mx-auto px-4 py-12">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-neutral-900 mb-2">
            Catalog Management
          </h1>
          <p className="text-neutral-600">
            Add products to the catalog by uploading product images
          </p>
        </div>
        <CatalogManager />
      </div>
    </div>
  );
}

