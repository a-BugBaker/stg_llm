from __future__ import annotations

"""prepare_graph 子模块共用的小工具。

这里集中放路径创建、JSON 读写、JSONL 追加等基础能力，
让场景图生成脚本和 OpenEQA 新增脚本都使用同一套约定。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .defaults import OUTPUT_SUBDIRS


def now_utc_iso() -> str:
    """返回 UTC ISO8601 时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


def parse_csv_strings(value: str | None) -> list[str]:
    """把逗号分隔字符串解析成非空字符串列表。"""
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def read_json(path: str | Path) -> Any:
    """读取 JSON 文件并返回 Python 对象。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    """写入 JSON 文件，统一使用 UTF-8 和缩进格式。"""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """向 JSONL 文件追加多行记录。"""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return out_path


def ensure_output_tree(output_root: str | Path) -> dict[str, Path]:
    """按项目约定创建输出目录树。

    返回值中的 key 与目录名一致，便于上层脚本按名字直接取目录。
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    result = {name: root / name for name in OUTPUT_SUBDIRS}
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result
