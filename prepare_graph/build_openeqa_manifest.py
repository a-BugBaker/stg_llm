from __future__ import annotations

"""构建 OpenEQA pilot 实验所需的 manifest 与问题清单。"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .common import append_jsonl, ensure_output_tree, now_utc_iso, parse_csv_strings, write_json
from .defaults import (
    DEFAULT_OPENEQA_EPISODES_DIR,
    DEFAULT_OPENEQA_MANIFEST_PATH,
    DEFAULT_OPENEQA_OUTPUT_ROOT,
    DEFAULT_OPENEQA_PARQUET_PATH,
    DEFAULT_OPENEQA_PILOT_PER_SOURCE,
    DEFAULT_OPENEQA_QUESTIONS_PATH,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 OpenEQA pilot manifest。")
    parser.add_argument("--parquet", default=str(DEFAULT_OPENEQA_PARQUET_PATH))
    parser.add_argument("--episodes-root", default=str(DEFAULT_OPENEQA_EPISODES_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OPENEQA_OUTPUT_ROOT))
    parser.add_argument("--manifest-output", default=str(DEFAULT_OPENEQA_MANIFEST_PATH))
    parser.add_argument("--questions-output", default=str(DEFAULT_OPENEQA_QUESTIONS_PATH))
    parser.add_argument("--per-source", type=int, default=DEFAULT_OPENEQA_PILOT_PER_SOURCE)
    parser.add_argument("--selection", choices=("pilot", "all"), default="pilot")
    parser.add_argument("--sources", default="hm3d-v0,scannet-v0")
    parser.add_argument("--resource-type", choices=("any", "image_dir", "video"), default="any")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    ensure_output_tree(output_root)

    manifest, question_rows = build_openeqa_manifest(
        parquet_path=Path(args.parquet).expanduser().resolve(),
        episodes_root=Path(args.episodes_root).expanduser().resolve(),
        per_source=args.per_source,
        selection=args.selection,
        sources=parse_csv_strings(args.sources),
        resource_type=args.resource_type,
    )

    manifest_output = Path(args.manifest_output).expanduser().resolve()
    questions_output = Path(args.questions_output).expanduser().resolve()
    write_json(manifest_output, manifest)
    if questions_output.exists():
        questions_output.unlink()
    append_jsonl(questions_output, question_rows)

    print("OpenEQA manifest build finished")
    print(f"Manifest: {manifest_output}")
    print(f"Questions: {questions_output}")
    print(f"Selected episodes: {len(manifest['clips'])}")
    print(f"Selected questions: {len(question_rows)}")


def build_openeqa_manifest(
    *,
    parquet_path: Path,
    episodes_root: Path,
    per_source: int,
    selection: str = "pilot",
    sources: list[str] | None = None,
    resource_type: str = "any",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """读取 parquet 并构建 pilot 使用的 clips 与问题表。"""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    if not episodes_root.exists():
        raise FileNotFoundError(f"Episodes root not found: {episodes_root}")

    df = pd.read_parquet(parquet_path)
    required_columns = {"answer", "question_id", "question", "episode_history", "extra_answers", "category"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in parquet: {sorted(missing)}")

    selected_sources = sources or ["hm3d-v0", "scannet-v0"]
    if selection == "pilot":
        selected_episodes = _select_pilot_episodes(df, per_source=per_source, sources=selected_sources)
    elif selection == "all":
        selected_episodes = _select_all_episodes(df, sources=selected_sources)
    else:
        raise ValueError(f"Unsupported selection strategy: {selection}")

    clips: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []

    for episode_history in selected_episodes:
        episode_df = df[df["episode_history"] == episode_history].copy()
        dataset_source = str(episode_history).split("/")[0]
        clip_id = str(episode_history).split("/")[-1]
        sample_id = _sample_id_from_episode(episode_history)
        resource = _locate_episode_resource(episodes_root, episode_history)
        if resource_type != "any" and resource["resource_type"] != resource_type:
            continue

        clips.append(
            {
                "sample_id": sample_id,
                "split": "test",
                "dataset_source": dataset_source,
                "episode_history": episode_history,
                "video_id": dataset_source,
                "clip_id": clip_id,
                "image_dir": resource["image_dir"],
                "video_path": resource["video_path"],
                "resource_type": resource["resource_type"],
                "question_count": int(len(episode_df)),
            }
        )

        for row in episode_df.itertuples(index=False):
            extra_answers = [] if row.extra_answers is None else list(row.extra_answers)
            question_rows.append(
                {
                    "question_id": str(row.question_id),
                    "episode_history": str(row.episode_history),
                    "question": str(row.question),
                    "answer": str(row.answer),
                    "extra_answers": extra_answers,
                    "category": str(row.category),
                    "sample_id": sample_id,
                }
            )

    manifest = {
        "generated_at": now_utc_iso(),
        "dataset": "OpenEQA",
        "parquet_path": str(parquet_path.resolve()),
        "episodes_root": str(episodes_root.resolve()),
        "selection_strategy": {
            "type": selection,
            "per_source": per_source if selection == "pilot" else None,
            "sources": selected_sources,
            "resource_type": resource_type,
        },
        "clips": clips,
    }
    if not clips:
        raise ValueError(
            "No episodes were selected. Check --sources, --selection, --per-source, and --resource-type."
        )
    return manifest, question_rows


def _select_pilot_episodes(df: pd.DataFrame, *, per_source: int, sources: list[str]) -> list[str]:
    """按数据源稳定抽样，保证 pilot 可复现。"""
    selected: list[str] = []
    unique_episodes = sorted(str(item) for item in df["episode_history"].drop_duplicates().tolist())
    for source in sources:
        source_episodes = [episode for episode in unique_episodes if episode.startswith(f"{source}/")]
        if len(source_episodes) < per_source:
            raise ValueError(f"Not enough episodes for source {source}: need {per_source}, got {len(source_episodes)}")
        selected.extend(source_episodes[:per_source])
    return selected


def _select_all_episodes(df: pd.DataFrame, *, sources: list[str]) -> list[str]:
    """Return every episode from the requested OpenEQA sources."""
    unique_episodes = sorted(str(item) for item in df["episode_history"].drop_duplicates().tolist())
    if not sources:
        return unique_episodes
    return [episode for episode in unique_episodes if any(episode.startswith(f"{source}/") for source in sources)]


def _sample_id_from_episode(episode_history: str) -> str:
    """把带 `/` 的 episode 标识转换为适合文件系统使用的 sample_id。"""
    return str(episode_history).replace("/", "__")


def _locate_episode_resource(episodes_root: Path, episode_history: str) -> dict[str, str]:
    """定位 episode 的视觉资源。

    OpenEQA 资源可能有两种形态：
    1. 一个已经解帧好的图片目录
    2. 一个原始 mp4 文件

    第一版优先识别图片目录；如果找不到，再识别同名视频文件。
    """
    dataset_source, clip_id = str(episode_history).split("/", 1)
    exact_candidates = [
        episodes_root / episode_history,
        episodes_root / dataset_source / clip_id,
        episodes_root / clip_id,
    ]
    for candidate in exact_candidates:
        resolved_dir = _pick_image_dir(candidate)
        if resolved_dir is not None:
            return {
                "image_dir": str(resolved_dir.resolve()),
                "video_path": "",
                "resource_type": "image_dir",
            }

        resolved_video = _pick_video_file(candidate)
        if resolved_video is not None:
            return {
                "image_dir": "",
                "video_path": str(resolved_video.resolve()),
                "resource_type": "video",
            }

    leaf_name = clip_id.split("/")[-1]
    leaf_matches = sorted(path for path in episodes_root.rglob(leaf_name) if path.is_dir())
    for candidate in leaf_matches:
        resolved_dir = _pick_image_dir(candidate)
        if resolved_dir is not None:
            return {
                "image_dir": str(resolved_dir.resolve()),
                "video_path": "",
                "resource_type": "image_dir",
            }

    video_matches = sorted(
        path
        for path in episodes_root.rglob(f"{leaf_name}.*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if video_matches:
        return {
            "image_dir": "",
            "video_path": str(video_matches[0].resolve()),
            "resource_type": "video",
        }

    raise FileNotFoundError(f"Cannot locate image directory or video file for episode: {episode_history}")


def _pick_image_dir(root: Path) -> Path | None:
    """从一个候选根目录中选择最合适的图像目录。"""
    if not root.exists() or not root.is_dir():
        return None

    if _count_images(root) > 0:
        return root

    nested_candidates = [path for path in root.rglob("*") if path.is_dir() and _count_images(path) > 0]
    if not nested_candidates:
        return None

    nested_candidates.sort(key=lambda path: (-_count_images(path), len(path.parts), str(path)))
    return nested_candidates[0]


def _pick_video_file(root: Path) -> Path | None:
    """从候选路径推断对应的视频文件。"""
    if root.is_file() and root.suffix.lower() in VIDEO_SUFFIXES:
        return root

    if root.exists() and root.is_dir():
        direct_video_files = sorted(
            path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
        if direct_video_files:
            return direct_video_files[0]

    parent = root.parent
    if parent.exists():
        exact_video_files = sorted(
            path
            for path in parent.glob(f"{root.name}.*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
        if exact_video_files:
            return exact_video_files[0]

    return None


def _count_images(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


if __name__ == "__main__":
    main()
