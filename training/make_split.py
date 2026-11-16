"""
make_split.py - Run ONCE on the CPU, before training.

What it does
------------
1. Reads Paddy Doctor's train.csv and checks every image opens.
2. Computes a perceptual hash (pHash) of every image and finds near-duplicates
   (Hamming distance <= --threshold bits out of 64).
3. Groups near-duplicates together (union-find):
     * groups whose images carry DIFFERENT labels are dropped (ambiguous / mislabelled),
     * exact duplicates (distance 0, same label) are collapsed to one image,
     * remaining near-duplicates stay, but the whole group goes into ONE split.
4. Writes a stratified 70/15/15 train/val/test split to splits.csv, plus
   duplicate_report.csv listing everything that was removed and why.
5. Runs sanity checks (no leakage, class proportions within +-1%).

splits.csv is committed to git so everyone (incl. the friend with the GPU) trains
and tests on exactly the same images.

Usage
-----
    python make_split.py                      # defaults: data/paddy, threshold 4, seed 42
    python make_split.py --data-dir data/paddy --threshold 4 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}


# ---------------------------------------------------------------- hashing
def _hash_one(path: str):
    """Return (64 phash bits, width, height) or None if the image is unreadable.
    Top-level function so ProcessPoolExecutor can pickle it on Windows."""
    import imagehash
    from PIL import Image

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            bits = imagehash.phash(im, hash_size=8).hash.flatten().astype(np.uint8)
        return bits, w, h
    except Exception:
        return None


def compute_hashes(paths: list[str], cache_file: Path, workers: int) -> pd.DataFrame:
    """pHash every image (cached to disk so re-runs are instant)."""
    if cache_file.exists():
        cache = pd.read_csv(cache_file, dtype={"bits": str})
        if set(cache["path"]) >= set(paths):
            print(f"Using cached hashes from {cache_file}")
            return cache.set_index("path").loc[paths].reset_index()

    from tqdm import tqdm

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = ex.map(_hash_one, paths, chunksize=64)
        for p, r in tqdm(zip(paths, results), total=len(paths), desc="Hashing images"):
            if r is None:
                rows.append({"path": p, "ok": False, "bits": "", "width": -1, "height": -1})
            else:
                bits, w, h = r
                rows.append({"path": p, "ok": True, "bits": "".join(map(str, bits)),
                             "width": w, "height": h})
    df = pd.DataFrame(rows)
    df.to_csv(cache_file, index=False)
    return df


# ---------------------------------------------------------------- grouping
class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_pairs(bits: np.ndarray, threshold: int, chunk: int = 1024):
    """Yield (i, j, distance) for all i<j with Hamming distance <= threshold.

    Hamming(a, b) = |a| + |b| - 2 a.b   (for 0/1 vectors), computed as a matrix product
    in row chunks so memory stays small (~40 MB per chunk for 10k images)."""
    B = bits.astype(np.float32)
    ones = B.sum(1)
    n = len(B)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = ones[start:stop, None] + ones[None, :] - 2.0 * (B[start:stop] @ B.T)
        ii, jj = np.nonzero(d <= threshold + 0.5)
        ii = ii + start
        keep = jj > ii
        for i, j in zip(ii[keep], jj[keep]):
            yield int(i), int(j), int(round(d[i - start, j]))


# ---------------------------------------------------------------- splitting
def stratified_group_split(df: pd.DataFrame, seed: int) -> pd.Series:
    """Assign whole groups to splits, per class, so each class gets ~70/15/15 of its IMAGES."""
    rng = np.random.default_rng(seed)
    split = pd.Series(index=df.index, dtype=object)
    for label, cdf in df.groupby("label"):
        sizes = cdf.groupby("group_id").size()
        # NOTE: rng.permutation returns a NEW array. Shuffling sizes.index.to_numpy() in place
        # would scramble the index itself (it's a view) and break the size lookups below.
        groups = rng.permutation(sizes.index.to_numpy())
        total = len(cdf)
        targets = {"test": SPLIT_FRACS["test"] * total, "val": SPLIT_FRACS["val"] * total}
        filled = {"test": 0, "val": 0}
        assignment = {}
        for g in groups:
            # fill test first, then val, everything else -> train
            for s in ("test", "val"):
                if filled[s] + sizes[g] / 2 <= targets[s]:
                    assignment[g] = s
                    filled[s] += sizes[g]
                    break
            else:
                assignment[g] = "train"
        split.loc[cdf.index] = cdf["group_id"].map(assignment)
    return split


# ---------------------------------------------------------------- checks
def sanity_checks(out: pd.DataFrame) -> bool:
    ok = True
    if out["image_id"].duplicated().any():
        print("FAIL: an image_id appears more than once"); ok = False
    spans = out.groupby("group_id")["split"].nunique()
    if (spans > 1).any():
        print(f"FAIL: {int((spans > 1).sum())} duplicate groups span several splits"); ok = False

    overall = out["label"].value_counts(normalize=True)
    print("\nClass proportions (% of each split) - should be within +-1% of overall:")
    table = pd.DataFrame({"overall": overall})
    for s in SPLIT_FRACS:
        table[s] = out[out["split"] == s]["label"].value_counts(normalize=True)
    print((table * 100).round(2).to_string())
    worst = (table[list(SPLIT_FRACS)].sub(table["overall"], axis=0)).abs().max().max()
    if worst > 0.01:
        print(f"FAIL: max proportion deviation {worst * 100:.2f}% > 1%"); ok = False

    print("\nSplit sizes:")
    print(out["split"].value_counts().reindex(list(SPLIT_FRACS)).to_string())
    return ok


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/paddy", help="folder containing train.csv and train_images/")
    ap.add_argument("--out", default="splits.csv")
    ap.add_argument("--report", default="duplicate_report.csv")
    ap.add_argument("--threshold", type=int, default=4, help="max pHash Hamming distance for near-duplicates (0-64)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "train.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Download the dataset first (GUIDE.md section 2).")
        return 1

    # ---- 1. read metadata ------------------------------------------------
    meta = pd.read_csv(csv_path)
    meta["path"] = "train_images/" + meta["label"] + "/" + meta["image_id"]
    print(f"train.csv: {len(meta)} rows, {meta['label'].nunique()} classes")
    print(meta["label"].value_counts().to_string())

    # ---- 2. hash ---------------------------------------------------------
    full_paths = [str(data_dir / p) for p in meta["path"]]
    h = compute_hashes(full_paths, data_dir / "phash_cache.csv", args.workers)
    meta["ok"] = h["ok"].to_numpy()
    meta["bits"] = h["bits"].to_numpy()
    meta["width"] = h["width"].to_numpy()
    meta["height"] = h["height"].to_numpy()

    removed = []  # rows for duplicate_report.csv
    bad = meta[~meta["ok"]]
    for _, r in bad.iterrows():
        removed.append({"image_id": r.image_id, "label": r.label, "reason": "unreadable", "group_id": -1, "kept_as": ""})
    meta = meta[meta["ok"]].reset_index(drop=True)
    print(f"\nUnreadable images: {len(bad)}")
    print("Image sizes (WxH, top 5):")
    print((meta["width"].astype(str) + "x" + meta["height"].astype(str)).value_counts().head().to_string())

    # ---- 3. near-duplicate groups ---------------------------------------
    bits = np.array([[int(c) for c in s] for s in meta["bits"]], dtype=np.uint8)
    uf_near, uf_exact = UnionFind(len(meta)), UnionFind(len(meta))
    n_pairs = n_exact = 0
    for i, j, d in near_duplicate_pairs(bits, args.threshold):
        uf_near.union(i, j); n_pairs += 1
        if d == 0:
            uf_exact.union(i, j); n_exact += 1
    meta["group_id"] = [uf_near.find(i) for i in range(len(meta))]
    meta["exact_id"] = [uf_exact.find(i) for i in range(len(meta))]
    print(f"\nNear-duplicate pairs (distance <= {args.threshold}): {n_pairs}  (exact: {n_exact})")

    # 3a. groups with conflicting labels -> drop all members
    n_labels = meta.groupby("group_id")["label"].transform("nunique")
    conflict = meta[n_labels > 1]
    for _, r in conflict.iterrows():
        removed.append({"image_id": r.image_id, "label": r.label, "reason": "label_conflict_in_duplicate_group",
                        "group_id": r.group_id, "kept_as": ""})
    meta = meta[n_labels == 1]
    print(f"Dropped {len(conflict)} images in {conflict['group_id'].nunique()} groups with conflicting labels")

    # 3b. exact duplicates (same label) -> keep the first of each
    dup_mask = meta.duplicated("exact_id", keep="first")
    first_of = meta.drop_duplicates("exact_id").set_index("exact_id")["image_id"]
    for _, r in meta[dup_mask].iterrows():
        removed.append({"image_id": r.image_id, "label": r.label, "reason": "exact_duplicate",
                        "group_id": r.group_id, "kept_as": first_of[r.exact_id]})
    meta = meta[~dup_mask].reset_index(drop=True)
    print(f"Removed {int(dup_mask.sum())} exact duplicates")
    multi = meta.groupby("group_id").size()
    print(f"Remaining near-duplicate groups (size>1): {int((multi > 1).sum())} covering {int(multi[multi > 1].sum())} images")

    # ---- 4. split --------------------------------------------------------
    meta["split"] = stratified_group_split(meta, args.seed)
    out = meta[["image_id", "label", "variety", "age", "path", "group_id", "split"]]
    ok = sanity_checks(out)

    out.to_csv(args.out, index=False)
    pd.DataFrame(removed, columns=["image_id", "label", "reason", "group_id", "kept_as"]).to_csv(args.report, index=False)
    print(f"\nWrote {args.out} ({len(out)} images) and {args.report} ({len(removed)} removed)")
    print("All checks passed." if ok else "SOME CHECKS FAILED - see above.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
