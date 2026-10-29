"""Image decoding and normalisation. Depends only on Pillow and NumPy."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Reject decompression bombs: a tiny file that expands to a huge bitmap.
Image.MAX_IMAGE_PIXELS = 40_000_000

MIN_SIDE = 64
PREPROCESS_MODES = ("none", "mobilenet_v2", "rescale")
_FILTERS = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


class InvalidImageError(ValueError):
    """The upload is not a usable image. The message is safe to show to the farmer."""


def decode_image(data: bytes, size: int, resize: str = "bilinear") -> np.ndarray:
    """bytes -> float32 RGB array (size, size, 3), values 0..255.

    The image is squashed to size x size without cropping, which is how Keras
    resizes when training with image_dataset_from_directory. Phone photos are
    4000 px wide, so JPEGs are shrunk while decoding (`draft`), which is much faster.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            if min(img.size) < MIN_SIDE:
                raise InvalidImageError(
                    f"The photo is too small ({img.width}x{img.height}). Take a closer, larger photo of the leaf."
                )
            img.draft("RGB", (size * 2, size * 2))  # no-op for non-JPEG
            img = ImageOps.exif_transpose(img)  # honour the phone's rotation
            img = img.convert("RGB").resize((size, size), _FILTERS[resize])
            return np.asarray(img, dtype=np.float32)
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
        raise InvalidImageError("Could not read the photo. Please send a clear JPG or PNG.") from exc


def normalize(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":  # the model rescales internally
        return arr
    if mode == "mobilenet_v2":  # keras.applications.mobilenet_v2.preprocess_input
        return arr / 127.5 - 1.0
    if mode == "rescale":
        return arr / 255.0
    raise ValueError(f"Unknown preprocess mode {mode!r}; use one of {PREPROCESS_MODES}")


def resolve_preprocess(setting: str, runtime: str, has_rescaling: bool | None) -> str:
    """Pick the pixel scaling.

    The shipped model starts with Rescaling(1/127.5, -1) ("mobilenet_preprocess"),
    so scaling again outside would normalise twice and wreck the predictions.
    With setting "auto": Keras models are inspected; an ONNX file is assumed to be
    exported from this same model (so it also scales internally).
    """
    if setting != "auto":
        if setting not in PREPROCESS_MODES:
            raise ValueError(f"Unknown PREPROCESS {setting!r}")
        return setting
    if runtime == "keras" and has_rescaling is False:
        return "mobilenet_v2"
    return "none"


def to_batch(data: bytes, size: int, mode: str, resize: str = "bilinear") -> np.ndarray:
    """bytes -> model input of shape (1, size, size, 3), float32."""
    return normalize(decode_image(data, size, resize), mode)[np.newaxis].astype(np.float32, copy=False)
