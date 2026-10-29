// Phone photos are 3 to 8 MB; the model looks at 224 px. Shrinking before upload makes
// the check fast on mobile data. (The server also copes with full-size photos.)

export const MAX_SIDE = 1024;

export function fitWithin(width, height, maxSide = MAX_SIDE) {
  const longest = Math.max(width, height);
  if (longest <= maxSide) return { width, height }; // never upscale
  const scale = maxSide / longest;
  return { width: Math.round(width * scale), height: Math.round(height * scale) };
}

export function isImageFile(file) {
  // Some systems report an empty type for HEIC or unusual files, so an empty type is let through.
  return Boolean(file) && (!file.type || file.type.startsWith("image/"));
}

// Returns a JPEG Blob no larger than MAX_SIDE, or the original file if the browser cannot decode it.
export async function prepareImage(file, { maxSide = MAX_SIDE, quality = 0.85 } = {}) {
  try {
    const bitmap = await createImageBitmap(file); // applies the phone's EXIF rotation
    const { width, height } = fitWithin(bitmap.width, bitmap.height, maxSide);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0, width, height);
    bitmap.close?.();
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
    return blob ?? file;
  } catch {
    return file;
  }
}
