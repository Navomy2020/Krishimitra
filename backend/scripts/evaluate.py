"""Measure the model through the real server pipeline, and tune the confidence thresholds.

    python -m scripts.evaluate path/to/val_dir [--limit 50]

val_dir must hold one folder per class, named exactly like the ids in
paddy_model/class_names.json:  val_dir/blast/*.jpg, val_dir/tungro/*.jpg ...
Use photos the model did NOT train on.

It prints top-1 / top-3 accuracy, recall per class, and an accuracy-by-confidence
table. That table is what to read when choosing HIGH_CONFIDENCE / LOW_CONFIDENCE.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.services.classifier import Classifier

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def summarize(y_true: np.ndarray, probs: np.ndarray, class_ids: list[str], high: float, low: float) -> dict:
    """Pure maths, separated from file loading so it can be tested."""
    top1 = probs.argmax(axis=1)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    conf = probs.max(axis=1)
    hit1 = top1 == y_true
    hit3 = np.array([t in row for t, row in zip(y_true, top3)])

    per_class = {}
    for i, cid in enumerate(class_ids):
        m = y_true == i
        per_class[cid] = {"n": int(m.sum()), "recall": float(hit1[m].mean()) if m.any() else None}

    edges = [0.0, low, high, 0.95, 1.0000001]
    names = [f"< {low:.2f} (low)", f"{low:.2f} to {high:.2f} (medium)", f"{high:.2f} to 0.95 (high)", ">= 0.95 (high)"]
    bands = []
    for name, lo, hi in zip(names, edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        bands.append({"band": name, "n": int(m.sum()), "accuracy": float(hit1[m].mean()) if m.any() else None})

    accepted = conf >= low
    return {
        "n": int(len(y_true)),
        "top1": float(hit1.mean()),
        "top3": float(hit3.mean()),
        "per_class": per_class,
        "bands": bands,
        "share_sent_to_expert": float((~accepted).mean()),
        "accuracy_when_not_sent_to_expert": float(hit1[accepted].mean()) if accepted.any() else None,
    }


def collect(root: Path, class_ids: list[str], limit: int | None) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name not in class_ids:
            print(f"skipping folder {folder.name!r}: not in class_names.json")
            continue
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in EXTS)
        samples += [(f, class_ids.index(folder.name)) for f in (files[:limit] if limit else files)]
    return samples


def pct(x: float | None) -> str:
    return "  n/a " if x is None else f"{x:6.1%}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("val_dir", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="max images per class")
    args = ap.parse_args()

    settings = get_settings()
    clf = Classifier.load(settings)
    samples = collect(args.val_dir, clf.class_ids, args.limit)
    if not samples:
        sys.exit("No images found. Expected val_dir/<class_id>/*.jpg")
    print(f"{len(samples)} images, preprocess={clf.preprocess}, resize={settings.resize_filter}\n")

    probs = np.stack([clf.probabilities(p.read_bytes()) for p, _ in samples])
    y = np.array([label for _, label in samples])
    r = summarize(y, probs, clf.class_ids, settings.high_confidence, settings.low_confidence)

    print(f"top-1 {r['top1']:.1%}   top-3 {r['top3']:.1%}\n")
    print("recall per class")
    for cid, v in r["per_class"].items():
        print(f"  {cid:26s} n={v['n']:4d}  {pct(v['recall'])}")
    print("\naccuracy by confidence band")
    for b in r["bands"]:
        print(f"  {b['band']:26s} n={b['n']:4d}  {pct(b['accuracy'])}")
    print(f"\nsent to an expert at the current thresholds: {r['share_sent_to_expert']:.1%}")
    print(f"accuracy on the rest: {pct(r['accuracy_when_not_sent_to_expert'])}")
    print("\nPick HIGH_CONFIDENCE where the band accuracy is one you would trust a farmer with,")
    print("and LOW_CONFIDENCE where accuracy falls below what is useful.")


if __name__ == "__main__":
    main()
