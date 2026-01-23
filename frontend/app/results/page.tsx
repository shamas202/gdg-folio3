"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { SearchResults } from "@/components/SearchResults";
import { SearchHit } from "@/lib/api";
import { Loader2 } from "lucide-react";

interface ResultsData {
  hits: SearchHit[];
  message?: string | null;
}

export default function ResultsPage() {
  const searchParams = useSearchParams();
  const [results, setResults] = useState<ResultsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const resultsData = searchParams.get("results");
    if (resultsData) {
      try {
        const parsed = JSON.parse(decodeURIComponent(resultsData));
        // Handle both old format (array) and new format (object with hits and message)
        if (Array.isArray(parsed)) {
          setResults({ hits: parsed, message: null });
        } else {
          setResults(parsed);
        }
      } catch (e) {
        setError("Failed to parse results");
      } finally {
        setLoading(false);
      }
    } else {
      setError("No results found");
      setLoading(false);
    }
  }, [searchParams]);

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-12">
        <div className="flex items-center justify-center min-h-[60vh]">
          <div className="text-center">
            <Loader2 className="w-12 h-12 animate-spin text-primary-600 mx-auto mb-4" />
            <p className="text-neutral-600">Loading results...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error || !results) {
    return (
      <div className="container mx-auto px-4 py-12">
        <div className="text-center">
          <p className="text-red-600 mb-4">{error || "No results available"}</p>
          <a
            href="/"
            className="text-primary-600 hover:text-primary-700 underline"
          >
            Return to search
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <SearchResults hits={results.hits} message={results.message} />
    </div>
  );
}

