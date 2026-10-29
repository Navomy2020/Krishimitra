"""Class order (from training) + disease display info and advice (from disease_info.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class KnowledgeError(ValueError):
    """class_names.json and disease_info.json disagree, or a file is malformed."""


@dataclass(frozen=True)
class Disease:
    id: str
    name: str
    healthy: bool
    advice: tuple[str, ...]


def load_class_ids(path: Path) -> list[str]:
    """The model's output order. This file is written by the training notebook."""
    ids = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        raise KnowledgeError(f"{path} must be a non-empty JSON list of class ids")
    if len(set(ids)) != len(ids):
        raise KnowledgeError(f"{path} contains duplicate class ids")
    return ids


def load_diseases(path: Path) -> dict[str, Disease]:
    raw = json.loads(Path(path).read_text(encoding="utf-8")).get("diseases")
    if not isinstance(raw, dict):
        raise KnowledgeError(f"{path} must contain a 'diseases' object")
    return {
        cid: Disease(
            id=cid,
            name=str(v.get("name") or cid.replace("_", " ").title()),
            healthy=bool(v.get("healthy", False)),
            advice=tuple(str(a) for a in v.get("advice", ())),
        )
        for cid, v in raw.items()
    }


def load_knowledge(class_names_path: Path, disease_info_path: Path) -> tuple[list[str], dict[str, Disease]]:
    ids = load_class_ids(class_names_path)
    diseases = load_diseases(disease_info_path)
    missing = [i for i in ids if i not in diseases]
    if missing:
        raise KnowledgeError(
            f"disease_info.json has no entry for: {', '.join(missing)}. "
            "Every class in class_names.json needs a name and advice."
        )
    return ids, diseases


def load_metrics(path: Path) -> dict | None:
    """Validation metrics saved by the training notebook (optional)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
