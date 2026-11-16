"""
krishi_data.py - reusable pieces for training the Krishimitra paddy disease classifier.

Why a separate module (and not just notebook cells)?
    On Windows, DataLoader worker processes are started with "spawn": they re-import
    the code they need. Classes/functions defined inside a notebook cannot be imported
    that way, so the Dataset and transforms MUST live in a real .py file.

Contents
    CLASS_NAMES, IMAGENET_MEAN/STD     - fixed label order + default normalisation constants
    model_norm_stats                   - the mean/std the chosen backbone was pretrained with
    seed_everything                    - reproducibility
    PaddyDataset                       - reads images listed in splits.csv
    build_transforms                   - train (augmenting) / eval (deterministic) pipelines
    make_loaders                       - DataLoaders for train / val / test
    set_backbone_trainable, build_optimizer, cosine_with_warmup - training setup
    train_one_epoch, predict, compute_metrics                 - loops + metrics
    EarlyStopping, save_checkpoint, load_checkpoint           - long-run safety
    fit_temperature, expected_calibration_error, reliability_bins - calibration
    TTAWrapper                                                - flip-TTA baked into the exported model
"""

from __future__ import annotations

import math
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# Same order as the old Keras model (paddy_model/class_names.json) -> outputs stay comparable.
CLASS_NAMES = [
    "bacterial_leaf_blight", "bacterial_leaf_streak", "bacterial_panicle_blight", "blast",
    "brown_spot", "dead_heart", "downy_mildew", "hispa", "normal", "tungro",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def model_norm_stats(model_name: str) -> tuple[tuple, tuple]:
    """Mean/std the backbone was PRETRAINED with. Not always ImageNet's!
    e.g. tf_efficientnetv2_s.in21k_ft_in1k uses mean=std=(0.5, 0.5, 0.5)."""
    import timm
    cfg = timm.models.get_pretrained_cfg(model_name)
    if cfg is None or cfg.mean is None:
        return IMAGENET_MEAN, IMAGENET_STD
    return tuple(cfg.mean), tuple(cfg.std)


# ====================================================================== reproducibility
def seed_everything(seed: int = 42) -> None:
    """Same seed -> same shuffling / augmentation / init -> (nearly) same results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ====================================================================== data
class PaddyDataset(Dataset):
    """One row of splits.csv -> (image tensor, class index).

    df   : rows of splits.csv (already filtered to one split)
    root : folder containing train_images/ (e.g. data/paddy)
    """

    def __init__(self, df: pd.DataFrame, root: str | Path, transform=None):
        self.paths = [str(Path(root) / p) for p in df["path"]]
        self.labels = [CLASS_TO_IDX[l] for l in df["label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]


def build_transforms(img_size: int, train: bool, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> v2.Compose:
    """Train: random augmentations that mimic real farmer phone photos.
    Eval:  deterministic resize + centre crop ONLY (never augment val/test)."""
    to_float_and_normalise = [
        v2.ToDtype(torch.float32, scale=True),           # uint8 0-255 -> float 0-1
        v2.Normalize(mean, std),                          # same stats the backbone was pretrained with
    ]
    if not train:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(img_size, antialias=True),          # shorter side -> img_size
            v2.CenterCrop(img_size),
            *to_float_and_normalise,
        ])
    return v2.Compose([
        v2.ToImage(),                                                        # PIL -> uint8 tensor
        v2.RandomResizedCrop(img_size, scale=(0.6, 1.0), antialias=True),    # zoom / framing
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),                                             # leaves have no "up"
        v2.RandomRotation(15),                                               # tilted phone
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),  # sun / shade / camera
        v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),      # out-of-focus shots
        v2.RandomApply([v2.JPEG(quality=(40, 95))], p=0.5),                 # WhatsApp compression
        *to_float_and_normalise,
    ])


def default_num_workers() -> int:
    """Windows 'spawn' workers are slow to start; keep a modest number."""
    n = os.cpu_count() or 2
    return min(4 if platform.system() == "Windows" else 8, max(0, n - 1))


def make_loaders(splits: pd.DataFrame, root: str | Path, img_size: int, batch_size: int,
                 num_workers: int | None = None, pin_memory: bool | None = None,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD) -> dict:
    """Return {'train','val','test'} DataLoaders from splits.csv."""
    if num_workers is None:
        num_workers = default_num_workers()
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    loaders = {}
    for split in ("train", "val", "test"):
        df = splits[splits["split"] == split]
        is_train = split == "train"
        ds = PaddyDataset(df, root, build_transforms(img_size, train=is_train, mean=mean, std=std))
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            drop_last=is_train,                 # Mixup/CutMix needs even, full batches
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
    return loaders


# ====================================================================== model / optimiser
def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Stage A: freeze everything except the classifier head. Stage B: unfreeze all."""
    head_params = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad = trainable or id(p) in head_params


def build_optimizer(model: nn.Module, lr_head: float, lr_backbone: float,
                    weight_decay: float = 0.05) -> torch.optim.Optimizer:
    """AdamW with two learning rates (head learns faster than the pretrained backbone)
    and no weight decay on biases / norm layers (standard practice)."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    groups = {k: [] for k in ("head_decay", "head_no_decay", "bb_decay", "bb_no_decay")}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head_ids else "bb"
        decay = "no_decay" if p.ndim <= 1 else "decay"   # 1-D = bias / norm weights
        groups[f"{part}_{decay}"].append(p)
    param_groups = [
        {"params": groups["head_decay"], "lr": lr_head, "weight_decay": weight_decay},
        {"params": groups["head_no_decay"], "lr": lr_head, "weight_decay": 0.0},
        {"params": groups["bb_decay"], "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": groups["bb_no_decay"], "lr": lr_backbone, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([g for g in param_groups if g["params"]])


def cosine_with_warmup(optimizer: torch.optim.Optimizer, total_steps: int,
                       warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """LR multiplier: linear 0->1 over warmup, then cosine 1->0.01 until total_steps.
    Call scheduler.step() once per OPTIMIZER step (not per batch when accumulating)."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ====================================================================== loops
def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device,
                    accum_steps: int = 1, mixup_fn=None, amp: bool = True,
                    max_grad_norm: float = 1.0, max_batches: int | None = None,
                    progress: bool = True) -> dict:
    """One pass over the training data.

    - AMP (mixed precision) on CUDA: faster, less VRAM.
    - Gradient accumulation: effective batch = batch_size * accum_steps.
    - Mixup/CutMix (optional): targets become soft -> criterion must accept soft targets.
    - NaN guard: stop immediately if the loss becomes NaN/inf (diverged training).
    Returns {'loss', 'acc'} (acc measured against the ORIGINAL hard labels).
    """
    model.train()
    use_amp = amp and device.type == "cuda"
    n_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    total_loss, correct, seen = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    it = enumerate(loader)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, total=n_batches, desc="train", leave=False)

    for i, (x, y) in it:
        if i >= n_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_hard = y
        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Loss became {loss.item()} at batch {i}. Training diverged - "
                "lower the learning rate or check the data, then resume from the last checkpoint.")

        scaler.scale(loss / accum_steps).backward()

        last_batch = (i + 1) == n_batches
        if (i + 1) % accum_steps == 0 or last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        bs = y_hard.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y_hard).sum().item()
        seen += bs
        if progress:
            it.set_postfix(loss=f"{total_loss / seen:.3f}", acc=f"{correct / seen:.3f}")

    return {"loss": total_loss / max(1, seen), "acc": correct / max(1, seen)}


@torch.no_grad()
def predict(model, loader, device, tta: bool = False, amp: bool = True,
            max_batches: int | None = None, progress: bool = True):
    """Run the model over a loader. Returns (logits [N,C] float32 CPU, labels [N] CPU).
    tta=True averages the logits of the original image and its horizontal flip."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    all_logits, all_labels = [], []
    it = loader
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(loader, desc="eval", leave=False)
    for i, (x, y) in enumerate(it):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            if tta:
                logits = (logits + model(torch.flip(x, dims=[3]))) / 2
        all_logits.append(logits.float().cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> dict:
    """Loss (NLL), top-1, top-3 and macro-F1 from raw logits."""
    from sklearn.metrics import f1_score

    scaled = logits / temperature
    nll = nn.functional.cross_entropy(scaled, labels).item()
    top3 = scaled.topk(min(3, scaled.size(1)), dim=1).indices
    pred = top3[:, 0]
    return {
        "loss": nll,
        "top1": (pred == labels).float().mean().item(),
        "top3": (top3 == labels[:, None]).any(1).float().mean().item(),
        "macro_f1": f1_score(labels.numpy(), pred.numpy(), average="macro",
                             labels=list(range(logits.size(1))), zero_division=0),
    }


# ====================================================================== safety nets
@dataclass
class EarlyStopping:
    """Stop when the monitored score (higher = better) hasn't improved for `patience` epochs."""
    patience: int = 6
    min_delta: float = 1e-4
    best: float = -math.inf
    bad_epochs: int = 0

    def step(self, score: float) -> bool:
        """Returns True if this epoch is a new best."""
        if score > self.best + self.min_delta:
            self.best, self.bad_epochs = score, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def save_checkpoint(path: str | Path, **state) -> None:
    """Save anything with a state_dict() as its state_dict; everything else as-is.
    Writes to a temp file then renames, so a crash mid-save never corrupts the old checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: (v.state_dict() if hasattr(v, "state_dict") else v) for k, v in state.items()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(out, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location="cpu", **objects) -> dict:
    """Restore state_dicts into the given objects (model=..., optimizer=..., ...).
    Returns the remaining plain values (epoch, history, ...)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    for k, obj in objects.items():
        if obj is not None and k in ckpt:
            obj.load_state_dict(ckpt[k])
    return {k: v for k, v in ckpt.items() if k not in objects}


# ====================================================================== calibration
def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Temperature scaling: find one number T > 0 so that softmax(logits / T) has the
    lowest NLL on the VALIDATION set. T > 1 softens over-confident predictions.
    It never changes which class is predicted (argmax), only how confident it is."""
    log_t = torch.zeros(1, requires_grad=True)            # optimise log(T) so T stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    logits, labels = logits.float(), labels.long()

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def reliability_bins(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    """Group predictions by confidence. Returns per-bin (mean confidence, accuracy, count)."""
    conf, pred = probs.max(1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        n = int(m.sum())
        out.append((float(conf[m].mean()) if n else float((lo + hi) / 2),
                    float(correct[m].mean()) if n else 0.0, n))
    return out


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """ECE = average |confidence - accuracy| over confidence bins, weighted by bin size.
    0 = perfectly calibrated ("90% sure" is right 90% of the time)."""
    bins = reliability_bins(probs, labels, n_bins)
    total = sum(n for _, _, n in bins)
    return sum(n * abs(c - a) for c, a, n in bins) / max(1, total)


# ====================================================================== export
class TTAWrapper(nn.Module):
    """Wraps the classifier so ONE forward pass = average logits of (image, mirrored image).
    Exporting this wrapper bakes test-time augmentation into model.onnx, so the server
    gets exactly the same numbers as the notebook without any extra code."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.model(x) + self.model(torch.flip(x, dims=[3]))) / 2
