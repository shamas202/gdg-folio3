"use client";

import { useState, useRef, useEffect } from "react";
import { X, Square } from "lucide-react";
import { cn } from "@/lib/utils";

interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface BoundingBoxSelectorProps {
  imageUrl: string;
  onBboxChange?: (bbox: BoundingBox | null) => void;
  enabled?: boolean;
}

export function BoundingBoxSelector({
  imageUrl,
  onBboxChange,
  enabled = true,
}: BoundingBoxSelectorProps) {
  const [isDrawing, setIsDrawing] = useState(false);
  const [bbox, setBbox] = useState<BoundingBox | null>(null);
  const [startPos, setStartPos] = useState<{ x: number; y: number; originalX: number; originalY: number } | null>(
    null
  );
  const [currentPos, setCurrentPos] = useState<{ x: number; y: number; originalX: number; originalY: number } | null>(
    null
  );
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [originalImageSize, setOriginalImageSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      setOriginalImageSize({ width: img.width, height: img.height });
    };
    img.src = imageUrl;
  }, [imageUrl]);

  const getRelativeCoordinates = (
    e: React.MouseEvent<HTMLDivElement>
  ): { x: number; y: number; originalX: number; originalY: number } | null => {
    if (!containerRef.current || !imageRef.current || originalImageSize.width === 0) return null;

    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Get actual displayed image dimensions
    const imgRect = imageRef.current.getBoundingClientRect();
    const imgX = imgRect.left - rect.left;
    const imgY = imgRect.top - rect.top;
    const displayWidth = imgRect.width;
    const displayHeight = imgRect.height;

    // Check if click is within image bounds
    if (x < imgX || x > imgX + displayWidth || y < imgY || y > imgY + displayHeight) {
      return null;
    }

    // Convert to display-relative coordinates
    const displayX = x - imgX;
    const displayY = y - imgY;

    // Convert to original image coordinates
    const scaleX = originalImageSize.width / displayWidth;
    const scaleY = originalImageSize.height / displayHeight;
    const originalX = Math.round(displayX * scaleX);
    const originalY = Math.round(displayY * scaleY);

    return { x: displayX, y: displayY, originalX, originalY };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!enabled) return;
    const pos = getRelativeCoordinates(e);
    if (!pos) return;

    setIsDrawing(true);
    setStartPos(pos);
    setCurrentPos(pos);
    setBbox(null);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDrawing || !enabled) return;
    const pos = getRelativeCoordinates(e);
    if (!pos || !startPos) return;

    setCurrentPos(pos);

    // Use original image coordinates for the bbox
    const x1 = Math.min(startPos.originalX, pos.originalX);
    const y1 = Math.min(startPos.originalY, pos.originalY);
    const x2 = Math.max(startPos.originalX, pos.originalX);
    const y2 = Math.max(startPos.originalY, pos.originalY);

    setBbox({ x1, y1, x2, y2 });
  };

  const handleMouseUp = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    if (bbox && onBboxChange) {
      onBboxChange(bbox);
    }
  };

  const handleClear = () => {
    setBbox(null);
    setStartPos(null);
    setCurrentPos(null);
    setIsDrawing(false);
    if (onBboxChange) {
      onBboxChange(null);
    }
  };

  useEffect(() => {
    if (onBboxChange) {
      onBboxChange(bbox);
    }
  }, [bbox, onBboxChange]);

  // Calculate display bbox for visual feedback
  const getDisplayBbox = () => {
    if (!imageRef.current || !containerRef.current || originalImageSize.width === 0) return null;
    
    const containerRect = containerRef.current.getBoundingClientRect();
    const imgRect = imageRef.current.getBoundingClientRect();
    const displayWidth = imgRect.width;
    const displayHeight = imgRect.height;
    const scaleX = displayWidth / originalImageSize.width;
    const scaleY = displayHeight / originalImageSize.height;

    // Calculate image position relative to container
    const imgX = imgRect.left - containerRect.left;
    const imgY = imgRect.top - containerRect.top;

    if (bbox) {
      return {
        x1: imgX + bbox.x1 * scaleX,
        y1: imgY + bbox.y1 * scaleY,
        x2: imgX + bbox.x2 * scaleX,
        y2: imgY + bbox.y2 * scaleY,
      };
    }

    if (startPos && currentPos) {
      return {
        x1: Math.min(startPos.x, currentPos.x),
        y1: Math.min(startPos.y, currentPos.y),
        x2: Math.max(startPos.x, currentPos.x),
        y2: Math.max(startPos.y, currentPos.y),
      };
    }

    return null;
  };

  const displayBbox = getDisplayBbox();

  return (
    <div className="relative">
      <div
        ref={containerRef}
        className="relative bg-neutral-50 rounded-lg overflow-hidden border-2 border-neutral-200"
        style={{ minHeight: "200px" }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <img
          ref={imageRef}
          src={imageUrl}
          alt="Preview"
          className="max-w-full h-auto mx-auto block"
          style={{
            maxHeight: "400px",
            userSelect: "none",
            pointerEvents: enabled ? "auto" : "none",
          }}
          draggable={false}
        />
        {enabled && displayBbox && (
          <>
            <div
              className="absolute border-2 border-primary-500 bg-primary-500/10 pointer-events-none z-10"
              style={{
                left: `${displayBbox.x1}px`,
                top: `${displayBbox.y1}px`,
                width: `${displayBbox.x2 - displayBbox.x1}px`,
                height: `${displayBbox.y2 - displayBbox.y1}px`,
              }}
            />
            {bbox && (
              <div className="absolute top-2 left-2 bg-primary-600 text-white px-2 py-1 rounded text-xs font-medium z-10">
                ({bbox.x1}, {bbox.y1}) - ({bbox.x2}, {bbox.y2})
              </div>
            )}
          </>
        )}
      </div>
      {enabled && bbox && (
        <button
          onClick={handleClear}
          className="absolute top-2 right-2 p-2 bg-white rounded-full shadow-lg hover:bg-neutral-100 transition-colors z-10"
          title="Clear bounding box"
        >
          <X className="w-4 h-4 text-neutral-700" />
        </button>
      )}
      {enabled && !bbox && (
        <div className="absolute bottom-2 left-2 bg-neutral-800/80 text-white px-3 py-1.5 rounded text-xs flex items-center space-x-2">
          <Square className="w-3 h-3" />
          <span>Click and drag to select area</span>
        </div>
      )}
    </div>
  );
}

