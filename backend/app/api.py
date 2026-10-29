import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from .schemas import HealthResponse, ModelInfo, ScanResponse
from .services.classifier import Classifier
from .services.knowledge import load_metrics
from .services.preprocess import InvalidImageError

router = APIRouter()
log = logging.getLogger("krishimithra.api")
_CHUNK = 64 * 1024


def get_classifier(request: Request) -> Classifier | None:
    return getattr(request.app.state, "classifier", None)


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    """Read the upload, but stop as soon as it goes over `limit` bytes."""
    chunks, total = [], 0
    while chunk := await file.read(_CHUNK):
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, f"The photo is too large. The limit is {limit // (1024 * 1024)} MB.")
        chunks.append(chunk)
    return b"".join(chunks)


@router.get("/health", response_model=HealthResponse)
def health(clf: Classifier | None = Depends(get_classifier)):
    return HealthResponse(
        status="ok" if clf else "degraded",
        service="KrishiMithra API",
        model_loaded=clf is not None,
        runtime=clf.runtime if clf else None,
    )


@router.get("/api/model/info", response_model=ModelInfo)
def model_info(clf: Classifier | None = Depends(get_classifier)):
    if clf is None:
        raise HTTPException(503, "The disease model is not available right now.")
    s = clf.settings
    return ModelInfo(
        model_version=s.model_version,
        runtime=clf.runtime,
        preprocess=clf.preprocess,
        image_size=s.image_size,
        classes=clf.class_ids,
        thresholds={"high": s.high_confidence, "low": s.low_confidence},
        metrics=load_metrics(s.metrics_path),
    )


@router.post("/api/leaf/predict", response_model=ScanResponse)
async def predict(file: UploadFile = File(...), clf: Classifier | None = Depends(get_classifier)):
    if clf is None:
        raise HTTPException(503, "The disease model is not available right now. Try again in a moment.")

    data = await _read_limited(file, clf.settings.max_upload_bytes)
    if not data:
        raise HTTPException(400, "The photo is empty. Please choose it again.")

    try:
        # CPU-bound: run it off the event loop so /health and other requests stay responsive.
        result = await run_in_threadpool(clf.predict, data)
    except InvalidImageError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception:
        log.exception("Prediction failed")
        raise HTTPException(500, "The scan failed on the server. Please try again.") from None

    log.info("scan label=%s level=%s ms=%.0f", result["label"], result["confidence_level"], result["inference_ms"])
    return result
