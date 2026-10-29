from typing import Literal

from pydantic import BaseModel


class Prediction(BaseModel):
    rank: int
    label: str
    name: str
    confidence: float  # 0..1 for downstream agents and the Coordinator; the app shows words, not this
    is_healthy: bool


class ScanResponse(BaseModel):
    success: bool = True
    label: str
    disease: str
    is_healthy: bool
    confidence: float
    confidence_level: Literal["high", "medium", "low"]
    needs_expert: bool  # True -> escalate to KVK / Krishi Bhavan
    message: str
    advice: list[str]
    predictions: list[Prediction]  # top-k, best first
    model_version: str
    inference_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    model_loaded: bool
    runtime: str | None = None


class ModelInfo(BaseModel):
    model_version: str
    runtime: str
    preprocess: str
    image_size: int
    classes: list[str]
    thresholds: dict[str, float]
    metrics: dict | None = None  # validation results saved by the training notebook
