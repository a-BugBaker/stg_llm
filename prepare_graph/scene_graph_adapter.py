from __future__ import annotations

"""HSGG 场景图与当前 STG 输入格式之间的适配层。

职责分三步：
1. 找到场景图目录对应的 `scene_graphs.json`
2. 调用 `data_process.raw_to_use.run_pipeline` 做两步预处理
3. 把合并后的 JSON 规范化为 `stg_system` 可直接读取的帧列表
"""

import importlib
import json
import sys
from pathlib import Path
from typing import Any


def scene_graph_output_dir(
    clip: dict[str, Any],
    *,
    scene_graph_dir: Path,
    scene_graph_pattern: str,
) -> Path:
    """根据 manifest 中的 clip 信息生成场景图输出目录。"""
    format_values = {key: value for key, value in clip.items() if isinstance(key, str)}
    resolved = scene_graph_pattern.format(**format_values)
    return scene_graph_dir / resolved


def load_scene_graph_as_stg_frames(clip_scene_dir: Path) -> list[dict[str, Any]]:
    """从场景图目录中加载并转换为 STG 输入帧列表。

    优先复用已有的 `scene_graphs.merged.json`，避免重复跑预处理。
    若不存在，则会自动对 `scene_graphs.json` 执行 `raw_to_use` 两步处理。
    """
    clip_scene_dir = Path(clip_scene_dir)
    merged_path = clip_scene_dir / "scene_graphs.merged.json"
    source_path = clip_scene_dir / "scene_graphs.json"
    converted_path = clip_scene_dir / "scene_graphs.converted.json"

    if not merged_path.exists():
        if not source_path.exists():
            raise FileNotFoundError(f"Cannot find scene graph source file: {source_path}")
        raw_to_use = _load_raw_to_use_module()
        raw_to_use.run_pipeline(
            input_path=source_path,
            output_path=merged_path,
            intermediate_path=converted_path,
            keep_original_fields=False,
        )

    payload = json.loads(merged_path.read_text(encoding="utf-8"))
    return _normalize_stg_frames(payload)


def _load_raw_to_use_module():
    """动态加载 `data_process.raw_to_use`，兼容脚本与模块两种运行方式。"""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    importlib.invalidate_caches()
    return importlib.import_module("data_process.raw_to_use")


def _normalize_stg_frames(payload: Any) -> list[dict[str, Any]]:
    """把合并后的 payload 统一整理成 `list[frame]`。"""
    if isinstance(payload, dict):
        frames = payload.get("frames")
        if not isinstance(frames, list):
            raise ValueError("Merged scene graph payload must be a list or contain key 'frames'.")
    elif isinstance(payload, list):
        frames = payload
    else:
        raise ValueError("Merged scene graph payload must be a list or dict.")

    normalized_frames: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        objects = frame.get("objects", [])
        if not isinstance(objects, list):
            objects = []

        normalized_objects: list[dict[str, Any]] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            normalized_objects.append(
                {
                    "idx": int(obj.get("idx", -1)),
                    "box": list(obj.get("box", obj.get("bbox", [0, 0, 0, 0]))),
                    "score": float(obj.get("score", 0.0)),
                    "label": str(obj.get("label", "unknown")),
                    "tag": str(obj.get("tag", obj.get("label", "unknown"))),
                    "attributes": str(obj.get("attributes", "")),
                    "layer_id": int(obj.get("layer_id", 1)),
                    "layer_mapping": list(obj.get("layer_mapping", []) or []),
                    "relations": list(obj.get("relations", []) or []),
                }
            )

        normalized_frames.append(
            {
                "image_path": str(frame.get("image_path", "")),
                "objects": normalized_objects,
            }
        )

    return normalized_frames
