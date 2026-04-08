"""
Merge subject/object relations into one unified relations list.

Usage:
    python data_process/处理2.py \
            --input data/less_move.json \
            --output data/less_move.merged_relations.json

Default behavior:
- For each object, merge `subject_relations` and `object_relations` into `relations`.
- `relations` is a list (not a dict), using the original item style:
    {"idx": ..., "predicate": ..., "object_tag": ..., "confidence": ...}
- Direction is removed (no subject/object distinction in output entries).
- By default, original `subject_relations` and `object_relations` are removed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _relation_key(predicate: str, object_tag: str) -> str:
    return f"{predicate}|{object_tag}"


def _pick_better(old_item: Dict[str, Any], new_item: Dict[str, Any]) -> Dict[str, Any]:
    """Pick one relation record when key collides.

    Strategy:
    - Prefer higher confidence if both have confidence.
    - Otherwise keep the old one.
    """
    old_conf = old_item.get("confidence")
    new_conf = new_item.get("confidence")

    if isinstance(old_conf, (int, float)) and isinstance(new_conf, (int, float)):
        return new_item if new_conf > old_conf else old_item

    if old_conf is None and isinstance(new_conf, (int, float)):
        return new_item

    return old_item


def _merge_object_relations(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for rel in obj.get("subject_relations", []) or []:
        predicate = _normalize_text(rel.get("predicate", rel.get("name", rel.get("relation", ""))))
        object_tag = _normalize_text(rel.get("object_tag", rel.get("object", rel.get("target", ""))))
        if not predicate or not object_tag:
            continue

        item: Dict[str, Any] = {
            "predicate": predicate,
            "object_tag": object_tag,
        }

        idx = rel.get("idx")
        if idx is not None:
            item["idx"] = idx

        conf = _normalize_confidence(rel.get("confidence"))
        if conf is not None:
            item["confidence"] = conf

        key = _relation_key(predicate, object_tag)
        if key in merged:
            merged[key] = _pick_better(merged[key], item)
        else:
            merged[key] = item

    for rel in obj.get("object_relations", []) or []:
        predicate = _normalize_text(rel.get("predicate", rel.get("name", rel.get("relation", ""))))
        object_tag = _normalize_text(rel.get("subject_tag", rel.get("object", rel.get("source", ""))))
        if not predicate or not object_tag:
            continue

        item = {
            "predicate": predicate,
            "object_tag": object_tag,
        }

        idx = rel.get("idx")
        if idx is not None:
            item["idx"] = idx

        conf = _normalize_confidence(rel.get("confidence"))
        if conf is not None:
            item["confidence"] = conf

        key = _relation_key(predicate, object_tag)
        if key in merged:
            merged[key] = _pick_better(merged[key], item)
        else:
            merged[key] = item

    return list(merged.values())


def _iter_frames(payload: Any) -> Tuple[List[Dict[str, Any]], Any]:
    """Return (frames, root_payload).

    Supports:
    - list[frame]
    - {"frames": list[frame]}
    """
    if isinstance(payload, list):
        return payload, payload

    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        return payload["frames"], payload

    raise ValueError("Input JSON must be a frame list or an object with key 'frames'.")


def merge_relations_in_payload(payload: Any, keep_original_fields: bool) -> Any:
    frames, root = _iter_frames(payload)

    for frame in frames:
        objects = frame.get("objects", [])
        if not isinstance(objects, list):
            continue

        for obj in objects:
            if not isinstance(obj, dict):
                continue

            merged = _merge_object_relations(obj)
            obj["relations"] = merged

            if not keep_original_fields:
                obj.pop("subject_relations", None)
                obj.pop("object_relations", None)

    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge subject_relations/object_relations into unified relations list."
    )
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=False, help="Output JSON path")
    parser.add_argument(
        "--keep-original-fields",
        action="store_true",
        help="Keep subject_relations and object_relations in output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}.merged_relations.json")

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    merged_payload = merge_relations_in_payload(payload, keep_original_fields=args.keep_original_fields)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_payload, f, ensure_ascii=False, indent=2)

    print(f"Done. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
