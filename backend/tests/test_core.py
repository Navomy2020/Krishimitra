"""Core tests: no TensorFlow or FastAPI needed.

Run:  python -m tests.test_core        (works without pytest)
  or: pytest
"""
import io
import json
from dataclasses import replace

import numpy as np
from PIL import Image

from app.config import Settings, load_settings
from app.schemas import ScanResponse
from app.services.classifier import Classifier, build_result, confidence_level
from app.services.knowledge import KnowledgeError, load_class_ids, load_knowledge
from app.services.preprocess import InvalidImageError, decode_image, normalize, resolve_preprocess, to_batch
from scripts.evaluate import summarize

S = Settings()
IDS, DISEASES = load_knowledge(S.class_names_path, S.disease_info_path)
BLAST, NORMAL, TUNGRO = IDS.index("blast"), IDS.index("normal"), IDS.index("tungro")

# The order printed by the training notebook (train_ds.class_indices).
TRAINING_ORDER = {
    "bacterial_leaf_blight": 0, "bacterial_leaf_streak": 1, "bacterial_panicle_blight": 2, "blast": 3,
    "brown_spot": 4, "dead_heart": 5, "downy_mildew": 6, "hispa": 7, "normal": 8, "tungro": 9,
}


def _img(w=640, h=480, fmt="JPEG", mode="RGB", color=(30, 140, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (w, h), color).save(buf, fmt)
    return buf.getvalue()


def _fake(top: int, p: float, n: int = 10):
    """A model that puts probability p on class `top` and spreads the rest evenly."""
    def backend(x):
        out = np.full((1, n), (1 - p) / (n - 1), np.float32)
        out[0, top] = p
        return out
    return backend


def _clf(backend, **over):
    return Classifier(backend, IDS, DISEASES, replace(S, **over), preprocess="none")


# ---- knowledge ---------------------------------------------------------
def test_class_order_matches_training():
    assert {c: i for i, c in enumerate(IDS)} == TRAINING_ORDER


def test_every_class_has_name_and_advice():
    for cid in IDS:
        d = DISEASES[cid]
        assert d.name and d.advice, cid
    assert [c for c in IDS if DISEASES[c].healthy] == ["normal"]


def test_missing_disease_entry_fails_fast(tmp_path=None):
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "names.json").write_text(json.dumps(["blast", "mystery"]))
    (d / "info.json").write_text(json.dumps({"diseases": {"blast": {"name": "Blast", "advice": []}}}))
    try:
        load_knowledge(d / "names.json", d / "info.json")
    except KnowledgeError as exc:
        assert "mystery" in str(exc)
        return
    raise AssertionError("expected KnowledgeError")


def test_duplicate_class_ids_rejected():
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mkdtemp()) / "n.json"
    p.write_text(json.dumps(["a", "a"]))
    try:
        load_class_ids(p)
    except KnowledgeError:
        return
    raise AssertionError("expected KnowledgeError")


# ---- preprocessing -----------------------------------------------------
def test_decode_shape_and_range():
    arr = decode_image(_img(), 224)
    assert arr.shape == (224, 224, 3) and arr.dtype == np.float32 and 0 <= arr.min() <= arr.max() <= 255


def test_png_rgba_and_grayscale_are_accepted():
    assert decode_image(_img(fmt="PNG", mode="RGBA", color=(10, 200, 10, 128)), 224).shape == (224, 224, 3)
    assert decode_image(_img(fmt="PNG", mode="L", color=120), 224).shape == (224, 224, 3)


def test_normalize_modes():
    a = np.array([0.0, 127.5, 255.0], np.float32)
    assert np.array_equal(normalize(a, "none"), a)
    assert np.allclose(normalize(a, "mobilenet_v2"), [-1, 0, 1])
    assert np.allclose(normalize(a, "rescale"), [0, 0.5, 1])


def test_batch_shape():
    assert to_batch(_img(), 224, "none").shape == (1, 224, 224, 3)


def test_phone_rotation_is_applied():
    # 200x100 photo, left half red, right half blue, tagged EXIF orientation 6 (rotate 90 deg clockwise).
    im = Image.new("RGB", (200, 100), (0, 0, 255))
    im.paste((255, 0, 0), (0, 0, 100, 100))
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95, exif=exif)
    arr = decode_image(buf.getvalue(), 224)
    assert arr[20, 112, 0] > 200 and arr[20, 112, 2] < 60      # top is now red
    assert arr[200, 112, 2] > 200 and arr[200, 112, 0] < 60    # bottom is now blue


def test_bad_uploads_are_rejected_with_friendly_errors():
    for data in (b"not an image", b"", _img()[:200]):
        try:
            decode_image(data, 224)
        except InvalidImageError as exc:
            assert "photo" in str(exc).lower()
        else:
            raise AssertionError("expected InvalidImageError")


def test_tiny_image_rejected():
    try:
        decode_image(_img(20, 20), 224)
    except InvalidImageError as exc:
        assert "too small" in str(exc)
        return
    raise AssertionError("expected InvalidImageError")


def test_decompression_bomb_rejected():
    # 10000 x 10000 one-bit PNG: a few KB on disk, 100 million pixels when decoded.
    buf = io.BytesIO()
    Image.new("1", (10000, 10000), 0).save(buf, "PNG")
    try:
        decode_image(buf.getvalue(), 224)
    except InvalidImageError:
        return
    raise AssertionError("expected InvalidImageError")


def test_resolve_preprocess_protects_against_double_scaling():
    assert resolve_preprocess("auto", "keras", True) == "none"            # this model: Rescaling inside
    assert resolve_preprocess("auto", "keras", False) == "mobilenet_v2"   # a model without it
    assert resolve_preprocess("auto", "onnx", None) == "none"
    assert resolve_preprocess("rescale", "keras", True) == "rescale"      # explicit override wins


# ---- settings ----------------------------------------------------------
def test_settings_defaults_and_validation():
    s = load_settings({})
    assert s.preprocess == "auto" and s.low_confidence < s.high_confidence
    assert s.model_path.name == "mobilenetv2_paddy_final.keras" and s.model_path.parent.name == "paddy_model"
    for bad in ({"PREPROCESS": "wat"}, {"LOW_CONFIDENCE": "0.9", "HIGH_CONFIDENCE": "0.8"}, {"TOP_K": "0"}):
        try:
            load_settings(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
    assert load_settings({"CORS_ORIGINS": "http://a.test, http://b.test"}).cors_origins == ("http://a.test", "http://b.test")


# ---- classifier --------------------------------------------------------
def test_confidence_bands():
    assert confidence_level(0.97, 0.85, 0.60) == "high"
    assert confidence_level(0.70, 0.85, 0.60) == "medium"
    assert confidence_level(0.30, 0.85, 0.60) == "low"


def test_high_confidence_result():
    r = _clf(_fake(BLAST, 0.97)).predict(_img())
    assert r["label"] == "blast" and r["disease"] == "Rice Blast"
    assert r["confidence_level"] == "high" and not r["needs_expert"] and r["advice"]
    assert "most likely" in r["message"]


def test_medium_confidence_is_hedged_but_keeps_advice():
    r = _clf(_fake(BLAST, 0.70)).predict(_img())
    assert r["confidence_level"] == "medium" and not r["needs_expert"] and r["advice"]
    assert "may have" in r["message"]


def test_low_confidence_escalates_and_withholds_advice():
    r = _clf(_fake(TUNGRO, 0.30)).predict(_img())
    assert r["confidence_level"] == "low" and r["needs_expert"] and r["advice"] == []
    assert "KVK" in r["message"]
    # The top guess must still be reported so the app can show it as a possibility.
    assert r["predictions"][0]["label"] == "tungro"


def test_top_three_are_ranked_best_first():
    probs = np.zeros(10)
    probs[[TUNGRO, BLAST, NORMAL]] = [0.5, 0.3, 0.2]
    r = build_result(probs, IDS, DISEASES, S)
    assert [p["label"] for p in r["predictions"]] == ["tungro", "blast", "normal"]
    assert [p["rank"] for p in r["predictions"]] == [1, 2, 3]
    assert r["confidence"] == r["predictions"][0]["confidence"] == 0.5


def test_healthy_leaf_messages():
    assert "healthy" in _clf(_fake(NORMAL, 0.95)).predict(_img())["message"].lower()
    r = _clf(_fake(NORMAL, 0.95)).predict(_img())
    assert r["is_healthy"] is True


def test_logits_are_softmaxed():
    clf = _clf(lambda x: np.arange(10, dtype=np.float32)[None] * 3)
    p = clf.probabilities(_img())
    assert np.isclose(p.sum(), 1.0) and p.argmax() == 9


def test_wrong_class_count_fails_fast():
    try:
        _clf(lambda x: np.zeros((1, 7), np.float32))
    except ValueError as exc:
        assert "7" in str(exc)
        return
    raise AssertionError("expected ValueError")


def test_response_matches_pydantic_schema():
    r = _clf(_fake(BLAST, 0.9)).predict(_img())
    m = ScanResponse(**r)
    assert m.confidence_level == "high" and len(m.predictions) == 3
    json.dumps(m.model_dump())  # serialisable


# ---- evaluation maths --------------------------------------------------
def test_evaluate_summary():
    probs = np.array([
        [0.9, 0.05, 0.05],   # true 0 -> right, high
        [0.5, 0.4, 0.1],     # true 1 -> wrong (top-1) but in top-3, low
        [0.1, 0.2, 0.7],     # true 2 -> right, medium
        [0.2, 0.7, 0.1],     # true 0 -> wrong, medium
    ])
    r = summarize(np.array([0, 1, 2, 0]), probs, ["a", "b", "c"], high=0.85, low=0.60)
    assert r["top1"] == 0.5 and r["top3"] == 1.0
    assert r["per_class"]["a"] == {"n": 2, "recall": 0.5}
    assert r["share_sent_to_expert"] == 0.25
    assert r["accuracy_when_not_sent_to_expert"] == 2 / 3
    assert sum(b["n"] for b in r["bands"]) == 4


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(tests)} passed")
