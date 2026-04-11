// =====================
// Image validation
// =====================

const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
const MAX_FILE_SIZE_MB = 15;

/**
 * Validates file type and size.
 * Returns an error string if invalid, or null if OK.
 */
export function validateImageFile(file: File): string | null {
  if (!ALLOWED_TYPES.includes(file.type)) {
    return `Unsupported file type: ${file.type}. Please upload a PNG, JPG, or WEBP image.`;
  }

  const sizeMb = file.size / (1024 * 1024);
  if (sizeMb > MAX_FILE_SIZE_MB) {
    return `File is too large (${sizeMb.toFixed(1)} MB). Maximum allowed size is ${MAX_FILE_SIZE_MB} MB.`;
  }

  return null;
}

// =====================
// Dimension helpers
// =====================

export interface ImageDimensions {
  width: number;
  height: number;
}

/**
 * Reads the natural width and height of an image file without rendering it.
 */
export function getImageDimensions(file: File): Promise<ImageDimensions> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();

    img.onload = () => {
      resolve({ width: img.naturalWidth, height: img.naturalHeight });
      URL.revokeObjectURL(url);
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to load image for dimension check."));
    };

    img.src = url;
  });
}

// =====================
// Preprocessing
// =====================

/**
 * Resizes an image so its longest side does not exceed `maxDimension` px.
 * Converts the result to a File with the given JPEG quality (0–1).
 * If the image is already within bounds, returns the original file unchanged.
 */
export async function preprocessImage(
  file: File,
  maxDimension = 1920,
  quality = 0.9,
): Promise<File> {
  const dims = await getImageDimensions(file);

  const longestSide = Math.max(dims.width, dims.height);
  if (longestSide <= maxDimension) {
    return file;
  }

  const scale = maxDimension / longestSide;
  const targetWidth = Math.round(dims.width * scale);
  const targetHeight = Math.round(dims.height * scale);

  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();

    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = targetWidth;
      canvas.height = targetHeight;

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        URL.revokeObjectURL(url);
        reject(new Error("Canvas 2D context unavailable."));
        return;
      }

      ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
      URL.revokeObjectURL(url);

      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error("Canvas toBlob failed."));
            return;
          }
          const resizedFile = new File([blob], file.name, {
            type: "image/jpeg",
            lastModified: Date.now(),
          });
          resolve(resizedFile);
        },
        "image/jpeg",
        quality,
      );
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to load image for preprocessing."));
    };

    img.src = url;
  });
}
