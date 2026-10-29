"""HTTP tests. Need fastapi + httpx (requirements-dev.txt). Run with: pytest"""
import io
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.classifier import Classifier  # noqa: E402
from app.services.knowledge import load_knowledge  # noqa: E402

S = Settings()
IDS, DISEASES = load_knowledge(S.class_names_path, S.disease_info_path)


def _jpeg(w=300, h=200) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (w, h), (20, 120, 40)).save(b, "JPEG")
    return b.getvalue()


def _backend(x):
    out = np.full((1, 10), 0.01, np.float32)
    out[0, IDS.index("blast")] = 0.91
    return out


@pytest.fixture()
def client():
    app = create_app()
    with TestClient(app) as c:  # the lifespan runs first; then swap in a fake model
        app.state.classifier = Classifier(_backend, IDS, DISEASES, replace(S, max_upload_bytes=1024 * 1024), preprocess="none")
        yield c


def test_health(client):
    assert client.get("/health").json()["model_loaded"] is True


def test_model_info(client):
    body = client.get("/api/model/info").json()
    assert body["classes"] == IDS and body["preprocess"] == "none"


def test_predict_ok(client):
    r = client.post("/api/leaf/predict", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "blast" and len(body["predictions"]) == 3


def test_predict_bad_image(client):
    r = client.post("/api/leaf/predict", files={"file": ("x.jpg", b"junk", "image/jpeg")})
    assert r.status_code == 400 and "photo" in r.json()["detail"].lower()


def test_predict_empty(client):
    assert client.post("/api/leaf/predict", files={"file": ("x.jpg", b"", "image/jpeg")}).status_code == 400


def test_predict_too_large(client):
    r = client.post("/api/leaf/predict", files={"file": ("x.jpg", b"0" * (2 * 1024 * 1024), "image/jpeg")})
    assert r.status_code == 413


def test_predict_without_model_is_503(client):
    client.app.state.classifier = None
    assert client.post("/api/leaf/predict", files={"file": ("x.jpg", _jpeg(), "image/jpeg")}).status_code == 503
    assert client.get("/health").json()["status"] == "degraded"


def test_predict_missing_file_is_422(client):
    assert client.post("/api/leaf/predict").status_code == 422
