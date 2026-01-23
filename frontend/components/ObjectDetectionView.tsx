"use client";

import { useState, useRef, useEffect } from "react";
import { SegmentedObject } from "@/lib/api";
import { cn } from "@/lib/utils";

interface ObjectDetectionViewProps {
  imageUrl: string;
  objects: SegmentedObject[];
  imageWidth: number;
  imageHeight: number;
  onObjectClick: (object: SegmentedObject) => void;
  loading?: boolean;
}

export function ObjectDetectionView({
  imageUrl,
  objects,
  imageWidth,
  imageHeight,
  onObjectClick,
  loading = false,
}: ObjectDetectionViewProps) {
  const [hoveredObjectId, setHoveredObjectId] = useState<number | null>(null);
  const [clickedObjectId, setClickedObjectId] = useState<number | null>(null);
  const [maskImages, setMaskImages] = useState<Map<number, string>>(new Map());
  const [maskOutlines, setMaskOutlines] = useState<Map<number, string>>(new Map());
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [imageInfo, setImageInfo] = useState({
    displayWidth: 0,
    displayHeight: 0,
    offsetX: 0,
    offsetY: 0,
  });

  // Helper function to create outline from mask
  const createOutline = (maskUrl: string): Promise<string> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        
        if (!ctx) {
          resolve(maskUrl); // Fallback to original mask
          return;
        }
        
        // Draw the mask
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        // Create outline canvas
        const outlineCanvas = document.createElement('canvas');
        outlineCanvas.width = canvas.width;
        outlineCanvas.height = canvas.height;
        const outlineCtx = outlineCanvas.getContext('2d');
        
        if (!outlineCtx) {
          resolve(maskUrl);
          return;
        }
        
        // Fill with cyan color
        outlineCtx.fillStyle = '#00FFFF';
        
        // Find edge pixels (pixels that are part of mask but have transparent neighbors)
        for (let y = 0; y < canvas.height; y++) {
          for (let x = 0; x < canvas.width; x++) {
            const idx = (y * canvas.width + x) * 4;
            const alpha = data[idx + 3];
            
            if (alpha > 128) {
              // Check 4-directional neighbors for edge detection
              let isEdge = false;
              const checks = [
                [x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]
              ];
              
              for (const [nx, ny] of checks) {
                if (nx < 0 || nx >= canvas.width || ny < 0 || ny >= canvas.height) {
                  isEdge = true;
                  break;
                }
                const nIdx = (ny * canvas.width + nx) * 4;
                if (data[nIdx + 3] <= 128) {
                  isEdge = true;
                  break;
                }
              }
              
              if (isEdge) {
                outlineCtx.fillRect(x, y, 1, 1);
              }
            }
          }
        }
        
        resolve(outlineCanvas.toDataURL('image/png'));
      };
      img.onerror = () => resolve(maskUrl);
      img.src = maskUrl;
    });
  };

  // Convert base64 masks to data URLs and generate outlines
  useEffect(() => {
    const newMaskImages = new Map<number, string>();
    const newMaskOutlines = new Map<number, string>();
    
    objects.forEach(async (obj) => {
      if (obj.mask_base64) {
        const maskUrl = `data:image/png;base64,${obj.mask_base64}`;
        newMaskImages.set(obj.object_id, maskUrl);
        
        // Generate outline asynchronously
        const outlineUrl = await createOutline(maskUrl);
        newMaskOutlines.set(obj.object_id, outlineUrl);
        setMaskOutlines(new Map(newMaskOutlines));
      }
    });
    
    setMaskImages(newMaskImages);
  }, [objects]);

  // Calculate display size and position when image loads or resizes
  useEffect(() => {
    const img = imageRef.current;
    const container = containerRef.current;
    
    if (!img || !container) return;

    const updateImageInfo = () => {
      const imgRect = img.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();
      
      // Calculate image position relative to container
      const offsetX = imgRect.left - containerRect.left;
      const offsetY = imgRect.top - containerRect.top;
      
      setImageInfo({
        displayWidth: imgRect.width,
        displayHeight: imgRect.height,
        offsetX,
        offsetY,
      });
    };

    // Update on load
    img.addEventListener("load", updateImageInfo);
    updateImageInfo();

    // Update on window resize
    window.addEventListener("resize", updateImageInfo);

    return () => {
      img.removeEventListener("load", updateImageInfo);
      window.removeEventListener("resize", updateImageInfo);
    };
  }, [imageUrl, objects]);

  // Convert original coordinates to display coordinates
  const getDisplayBbox = (bbox: SegmentedObject["bbox"]) => {
    if (
      !imageRef.current || 
      !containerRef.current ||
      imageInfo.displayWidth === 0 || 
      imageInfo.displayHeight === 0 ||
      imageWidth === 0 ||
      imageHeight === 0
    ) {
      return null;
    }

    const scaleX = imageInfo.displayWidth / imageWidth;
    const scaleY = imageInfo.displayHeight / imageHeight;

    return {
      x1: imageInfo.offsetX + bbox.x1 * scaleX,
      y1: imageInfo.offsetY + bbox.y1 * scaleY,
      x2: imageInfo.offsetX + bbox.x2 * scaleX,
      y2: imageInfo.offsetY + bbox.y2 * scaleY,
      width: (bbox.x2 - bbox.x1) * scaleX,
      height: (bbox.y2 - bbox.y1) * scaleY,
    };
  };

  // Reset clicked state when objects change
  useEffect(() => {
    setClickedObjectId(null);
  }, [objects]);

  // Handle mouse move to detect hover
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!imageRef.current || !containerRef.current || objects.length === 0) return;

    const imgRect = imageRef.current.getBoundingClientRect();
    
    // Calculate mouse position relative to image (not container)
    const mouseX = e.clientX - imgRect.left;
    const mouseY = e.clientY - imgRect.top;

    // Check if mouse is within image bounds
    if (mouseX < 0 || mouseY < 0 || mouseX > imgRect.width || mouseY > imgRect.height) {
      setHoveredObjectId(null);
      return;
    }

    // Check which object is hovered (in reverse order to prioritize top objects)
    let hoveredId: number | null = null;
    for (let i = objects.length - 1; i >= 0; i--) {
      const obj = objects[i];
      const displayBbox = getDisplayBbox(obj.bbox);
      if (!displayBbox) continue;
      
      // Convert display bbox to image-relative coordinates for comparison
      const bboxImageX1 = displayBbox.x1 - imageInfo.offsetX;
      const bboxImageY1 = displayBbox.y1 - imageInfo.offsetY;
      const bboxImageX2 = displayBbox.x2 - imageInfo.offsetX;
      const bboxImageY2 = displayBbox.y2 - imageInfo.offsetY;
      
      if (
        mouseX >= bboxImageX1 && 
        mouseX <= bboxImageX2 && 
        mouseY >= bboxImageY1 && 
        mouseY <= bboxImageY2
      ) {
        hoveredId = obj.object_id;
        break;
      }
    }

    setHoveredObjectId(hoveredId);
  };

  const handleMouseLeave = () => {
    setHoveredObjectId(null);
  };

  const handleObjectClick = (object: SegmentedObject) => {
    // Toggle selection: if already clicked, deselect; otherwise select
    if (clickedObjectId === object.object_id) {
      setClickedObjectId(null);
      // Don't call onObjectClick when deselecting
    } else {
      setClickedObjectId(object.object_id);
      onObjectClick(object);
    }
  };

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      <div
        ref={containerRef}
        className="relative bg-neutral-50 rounded-lg overflow-hidden border-2 border-neutral-200 w-full h-full flex items-center justify-center"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <img
          ref={imageRef}
          src={imageUrl}
          alt="Room image with detected objects"
          className="max-w-full max-h-full w-auto h-auto object-contain"
          style={{
            userSelect: "none",
            pointerEvents: "auto",
          }}
          draggable={false}
        />

        {/* Render invisible hit areas for all objects (for hover detection) */}
        {objects.map((obj) => {
          const displayBbox = getDisplayBbox(obj.bbox);
          if (!displayBbox) return null;

          const isHovered = hoveredObjectId === obj.object_id;
          const isClicked = clickedObjectId === obj.object_id;
          const shouldShow = isHovered || isClicked;
          const maskUrl = maskImages.get(obj.object_id);

          return (
            <div
              key={obj.object_id}
              className="absolute"
              style={{
                left: `${displayBbox.x1}px`,
                top: `${displayBbox.y1}px`,
                width: `${displayBbox.width}px`,
                height: `${displayBbox.height}px`,
              }}
            >
              {/* Invisible hit area - always present for hover detection */}
              <div
                className="absolute inset-0 cursor-pointer z-10"
                style={{
                  pointerEvents: shouldShow ? "none" : "auto",
                }}
                onMouseEnter={() => setHoveredObjectId(obj.object_id)}
                onClick={() => handleObjectClick(obj)}
                title={`Click to search for ${obj.category === "unknown" ? "this object" : obj.category}`}
              />

              {/* Visible elements - only show on hover or click */}
              {shouldShow && (
                <>
                  {/* Clickable area - entire bounding box */}
                  <div
                    className="absolute inset-0 cursor-pointer z-20"
                    onClick={() => handleObjectClick(obj)}
                    title={`Click to search for ${obj.category === "unknown" ? "this object" : obj.category}`}
                  />

                  {/* Bounding box outline - visible on hover only (not when clicked) */}
                  {isHovered && !isClicked && (
                    <div
                      className="absolute inset-0 border-2 border-primary-500 bg-primary-500/20 transition-all duration-200 pointer-events-none"
                      style={{
                        boxShadow: "0 0 0 2px rgba(59, 130, 246, 0.5)",
                      }}
                    />
                  )}

                  {/* Segmentation highlight - visible on click only, shows segmented area with bright outline */}
                  {isClicked && maskUrl && (
                    <>
                      {/* Semi-transparent overlay on segmented area for pressed effect */}
                      <div
                        className="absolute pointer-events-none"
                        style={{
                          left: `${-displayBbox.x1 + imageInfo.offsetX}px`,
                          top: `${-displayBbox.y1 + imageInfo.offsetY}px`,
                          width: `${imageInfo.displayWidth}px`,
                          height: `${imageInfo.displayHeight}px`,
                          backgroundImage: `url(${maskUrl})`,
                          backgroundSize: `${imageInfo.displayWidth}px ${imageInfo.displayHeight}px`,
                          backgroundPosition: `0px 0px`,
                          backgroundRepeat: "no-repeat",
                          backgroundColor: "rgba(0, 255, 255, 0.15)", // Cyan overlay
                          mixBlendMode: "overlay",
                          opacity: 0.8,
                          zIndex: 24,
                        }}
                      />
                      
                      {/* Bright cyan outline - only the edge pixels */}
                      <div
                        className="absolute pointer-events-none"
                        style={{
                          left: `${-displayBbox.x1 + imageInfo.offsetX}px`,
                          top: `${-displayBbox.y1 + imageInfo.offsetY}px`,
                          width: `${imageInfo.displayWidth}px`,
                          height: `${imageInfo.displayHeight}px`,
                          backgroundImage: maskOutlines.get(obj.object_id) 
                            ? `url(${maskOutlines.get(obj.object_id)})` 
                            : `url(${maskUrl})`,
                          backgroundSize: `${imageInfo.displayWidth}px ${imageInfo.displayHeight}px`,
                          backgroundPosition: `0px 0px`,
                          backgroundRepeat: "no-repeat",
                          filter: maskOutlines.get(obj.object_id)
                            ? "drop-shadow(0 0 3px #00FFFF) drop-shadow(0 0 6px rgba(0, 255, 255, 0.8)) drop-shadow(0 0 9px rgba(0, 255, 255, 0.6))"
                            : "drop-shadow(0 0 4px #00FFFF) drop-shadow(0 0 8px rgba(0, 255, 255, 0.8)) drop-shadow(0 0 12px rgba(0, 255, 255, 0.6))",
                          opacity: 1,
                          zIndex: 25,
                          transform: "scale(1.005)", // Slight scale for pressed effect
                          transition: "all 0.2s ease-out",
                        }}
                      />
                      
                      {/* Outer glow for depth and pressed state */}
                      <div
                        className="absolute pointer-events-none"
                        style={{
                          left: `${-displayBbox.x1 + imageInfo.offsetX}px`,
                          top: `${-displayBbox.y1 + imageInfo.offsetY}px`,
                          width: `${imageInfo.displayWidth}px`,
                          height: `${imageInfo.displayHeight}px`,
                          backgroundImage: `url(${maskUrl})`,
                          backgroundSize: `${imageInfo.displayWidth}px ${imageInfo.displayHeight}px`,
                          backgroundPosition: `0px 0px`,
                          backgroundRepeat: "no-repeat",
                          filter: "blur(8px)",
                          opacity: 0.4,
                          zIndex: 23,
                          backgroundColor: "#00FFFF",
                        }}
                      />
                    </>
                  )}

                  {/* Category label - visible on hover only (not when clicked) */}
                  {!isClicked && (
                    <div
                      className={cn(
                        "absolute top-1 left-1 px-2 py-1 rounded text-xs font-medium transition-all duration-200 pointer-events-none z-30",
                        isHovered
                          ? "bg-primary-600 text-white shadow-lg"
                          : "bg-white/90 text-neutral-700 shadow-sm"
                      )}
                    >
                      {obj.category === "unknown" ? "Object" : obj.category} ({(obj.score * 100).toFixed(0)}%)
                    </div>
                  )}
                </>
              )}
            </div>
          );
        })}

        {/* Loading overlay */}
        {loading && (
          <div className="absolute inset-0 bg-black/20 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg px-4 py-2 shadow-lg">
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
                <span className="text-sm text-neutral-700">Detecting objects...</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

