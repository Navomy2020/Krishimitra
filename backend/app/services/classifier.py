"""Leaf-disease classifier: runtime selection, inference and result shaping."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

import numpy as np

from ..config import Settings
from .knowledge import Disease, load_knowledge
from .preprocess import resolve_preprocess, to_batch

log = logging.getLogger("krishimithra.classifier")

Backend = Callable[[np.ndarray], np.ndarray]  # (1,H,W,3) float32 -> (1,N)


class _KerasBackend:
    runtime = "keras"

    def __init__(self, path):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow start-up noise
        import keras  # imported lazily so ONNX-only installs don't need TensorFlow

        self._model = keras.models.load_model(str(path), compile=False)
        self.has_rescaling = any(type(layer).__name__ == "Rescaling" for layer in self._model.layers)
        self._lock = threading.Lock()  # one prediction at a time per model

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self._lock:
            return np.asarray(self._model(x, training=False))


class _OnnxBackend:
    runtime = "onnx"
    has_rescaling = None  # not inspectable; assumed exported from the shipped model

    def __init__(self, path, threads: int):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._sess.run(None, {self._input: x})[0]


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def confidence_level(conf: float, high: float, low: float) -> str:
    return "high" if conf >= high else "medium" if conf >= low else "low"


def build_result(probs: np.ndarray, class_ids: list[str], diseases: dict[str, Disease], settings: Settings) -> dict:
    """Turn class probabilities into the API response (see schemas.ScanResponse)."""
    order = np.argsort(probs)[::-1][: settings.top_k]
    predictions = [
        {
            "rank": rank,
            "label": class_ids[int(i)],
            "name": diseases[class_ids[int(i)]].name,
            "confidence": round(float(probs[i]), 4),
            "is_healthy": diseases[class_ids[int(i)]].healthy,
        }
        for rank, i in enumerate(order, start=1)
    ]
    top = predictions[0]
    disease = diseases[top["label"]]
    level = confidence_level(top["confidence"], settings.high_confidence, settings.low_confidence)
    needs_expert = level == "low"

    if needs_expert:
        message = "The scan is not certain. Retake a clearer photo, or show the plant to your local KVK or Krishi Bhavan."
    elif disease.healthy:
        message = "No disease found. This leaf looks healthy." if level == "high" else "This leaf may be healthy. Retake a clearer photo to be sure."
    elif level == "high":
        message = f"This leaf most likely has {disease.name}."
    else:
        message = f"This leaf may have {disease.name}. Retake a clearer photo to be sure."

    return {
        "success": True,
        "label": top["label"],
        "disease": disease.name,
        "is_healthy": disease.healthy,
        "confidence": top["confidence"],
        "confidence_level": level,
        "needs_expert": needs_expert,
        "message": message,
        # Low confidence: no autonomous advice, the farmer is sent to an expert instead.
        "advice": [] if needs_expert else list(disease.advice),
        "predictions": predictions,
    }


class Classifier:
    def __init__(
        self,
        backend: Backend,
        class_ids: list[str],
        diseases: dict[str, Disease],
        settings: Settings,
        runtime: str = "custom",
        preprocess: str = "none",
    ):
        self._backend = backend
        self.class_ids = class_ids
        self.diseases = diseases
        self.settings = settings
        self.runtime = runtime
        self.preprocess = preprocess
        self._warmup()

    @classmethod
    def load(cls, settings: Settings) -> "Classifier":
        class_ids, diseases = load_knowledge(settings.class_names_path, settings.disease_info_path)
        if settings.onnx_path.exists():
            backend: Backend = _OnnxBackend(settings.onnx_path, settings.onnx_threads)
        elif settings.model_path.exists():
            backend = _KerasBackend(settings.model_path)
        else:
            raise FileNotFoundError(
                f"No model found at {settings.model_path}. Put the .keras file in {settings.model_dir} or set MODEL_DIR."
            )
        runtime = backend.runtime
        mode = resolve_preprocess(settings.preprocess, runtime, backend.has_rescaling)
        log.info("Loaded %s model: %d classes, preprocess=%s", runtime, len(class_ids), mode)
        if settings.preprocess != "auto" and backend.has_rescaling and mode != "none":
            log.warning("The model scales pixels itself but PREPROCESS=%s adds scaling again; predictions will be wrong.", mode)
        return cls(backend, class_ids, diseases, settings, runtime, mode)

    def _warmup(self) -> None:
        """The first call is slow (graph build). Do it at start-up and check the output size."""
        s = self.settings.image_size
        out = np.asarray(self._backend(np.zeros((1, s, s, 3), np.float32)))
        if out.shape[-1] != len(self.class_ids):
            raise ValueError(f"The model outputs {out.shape[-1]} classes but class_names.json lists {len(self.class_ids)}.")

    def probabilities(self, image_bytes: bytes) -> np.ndarray:
        x = to_batch(image_bytes, self.settings.image_size, self.preprocess, self.settings.resize_filter)
        probs = np.asarray(self._backend(x), dtype=np.float64).reshape(-1)
        if probs.min() < 0 or not np.isclose(probs.sum(), 1.0, atol=1e-3):  # e.g. an export that returns logits
            probs = _softmax(probs)
        return probs

    def predict(self, image_bytes: bytes) -> dict:
        t0 = time.perf_counter()
        result = build_result(self.probabilities(image_bytes), self.class_ids, self.diseases, self.settings)
        result["model_version"] = f"{self.settings.model_version} ({self.runtime})"
        result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result
