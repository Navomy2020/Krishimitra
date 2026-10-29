"""Settings, read once from environment variables (see backend/.env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent

PREPROCESS_CHOICES = ("auto", "none", "mobilenet_v2", "rescale")
RESIZE_CHOICES = ("bilinear", "bicubic", "lanczos")


@dataclass(frozen=True)
class Settings:
    # Where the trained model lives: <repo>/paddy_model (model, class_names.json, metrics.json).
    model_dir: Path = REPO_DIR / "paddy_model"
    model_file: str = "mobilenetv2_paddy_final.keras"
    onnx_file: str = "mobilenetv2_paddy_final.onnx"  # optional; used instead of .keras if present
    disease_info_path: Path = BACKEND_DIR / "app" / "data" / "disease_info.json"
    frontend_dist: Path = REPO_DIR / "frontend" / "dist"  # served by FastAPI if it exists

    image_size: int = 224

    # What to do to pixels before the network. This model has a Rescaling layer
    # built in, so it must receive RAW 0-255 pixels ("none"). "auto" detects that
    # from the loaded model; see services/preprocess.resolve_preprocess.
    preprocess: str = "auto"
    resize_filter: str = "bilinear"

    max_upload_bytes: int = 8 * 1024 * 1024
    # Confidence bands on the top-1 probability: >= high -> "high",
    # >= low -> "medium", below -> "low" (which also means: ask an expert).
    # Tune with `python -m scripts.evaluate` on held-out images.
    high_confidence: float = 0.85
    low_confidence: float = 0.60
    top_k: int = 3

    cors_origins: tuple[str, ...] = ("*",)
    onnx_threads: int = 2

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_file

    @property
    def onnx_path(self) -> Path:
        return self.model_dir / self.onnx_file

    @property
    def class_names_path(self) -> Path:
        return self.model_dir / "class_names.json"

    @property
    def metrics_path(self) -> Path:
        return self.model_dir / "metrics.json"

    @property
    def model_version(self) -> str:
        return Path(self.model_file).stem


def load_settings(env: dict[str, str] | None = None) -> Settings:
    e = os.environ if env is None else env
    d = Settings()

    def path(name: str, default: Path) -> Path:
        return Path(e[name]).expanduser() if e.get(name) else default

    origins = tuple(o.strip() for o in e.get("CORS_ORIGINS", "*").split(",") if o.strip()) or ("*",)
    s = Settings(
        model_dir=path("MODEL_DIR", d.model_dir),
        model_file=e.get("MODEL_FILE", d.model_file),
        onnx_file=e.get("ONNX_FILE", d.onnx_file),
        disease_info_path=path("DISEASE_INFO_PATH", d.disease_info_path),
        frontend_dist=path("FRONTEND_DIST", d.frontend_dist),
        image_size=int(e.get("IMAGE_SIZE", d.image_size)),
        preprocess=e.get("PREPROCESS", d.preprocess).lower(),
        resize_filter=e.get("RESIZE_FILTER", d.resize_filter).lower(),
        max_upload_bytes=int(e.get("MAX_UPLOAD_BYTES", d.max_upload_bytes)),
        high_confidence=float(e.get("HIGH_CONFIDENCE", d.high_confidence)),
        low_confidence=float(e.get("LOW_CONFIDENCE", d.low_confidence)),
        top_k=int(e.get("TOP_K", d.top_k)),
        cors_origins=origins,
        onnx_threads=int(e.get("ONNX_THREADS", d.onnx_threads)),
    )
    if s.preprocess not in PREPROCESS_CHOICES:
        raise ValueError(f"PREPROCESS must be one of {PREPROCESS_CHOICES}, got {s.preprocess!r}")
    if s.resize_filter not in RESIZE_CHOICES:
        raise ValueError(f"RESIZE_FILTER must be one of {RESIZE_CHOICES}, got {s.resize_filter!r}")
    if not 0.0 < s.low_confidence < s.high_confidence <= 1.0:
        raise ValueError("Confidence thresholds must satisfy 0 < LOW_CONFIDENCE < HIGH_CONFIDENCE <= 1")
    if not 1 <= s.top_k <= 10:
        raise ValueError("TOP_K must be between 1 and 10")
    return s


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
