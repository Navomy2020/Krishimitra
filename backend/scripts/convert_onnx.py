"""OPTIONAL: export the Keras model to ONNX for lighter, faster serving (no TensorFlow in production).

    pip install onnx tf2onnx
    python -m scripts.convert_onnx

Writes paddy_model/mobilenetv2_paddy_final.onnx, then compares it with the Keras
model on random images. Only keep the .onnx file if it says OK. If an .onnx file
exists the server uses it automatically; delete it to go back to the .keras model.
The preprocessing stays inside the model, so the server still sends raw 0-255 pixels.
"""
from __future__ import annotations

import sys

import numpy as np

from app.config import get_settings


def main() -> None:
    s = get_settings()
    import keras

    model = keras.models.load_model(str(s.model_path), compile=False)
    print("exporting to", s.onnx_path)
    try:
        model.export(str(s.onnx_path), format="onnx")
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"Export failed: {exc}\n"
            "Fallback: model.export('saved_model_dir') then\n"
            f"  python -m tf2onnx.convert --saved-model saved_model_dir --output {s.onnx_path} --opset 17"
        )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(s.onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).uniform(0, 255, (4, s.image_size, s.image_size, 3)).astype("float32")
    ref = np.asarray(model(x, training=False))
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    diff = float(np.abs(ref - got).max())
    ok = diff < 1e-3
    print(f"max |keras - onnx| = {diff:.2e}", "OK" if ok else "MISMATCH: delete the .onnx file")
    if not ok:
        s.onnx_path.unlink(missing_ok=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
