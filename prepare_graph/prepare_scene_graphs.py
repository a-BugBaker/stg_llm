from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .common import append_jsonl, ensure_output_tree, now_utc_iso, parse_csv_strings, read_json, write_json
from .defaults import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_FRAME_CACHE_DIR,
    DEFAULT_HSGG_CUDA_DEVICE,
    DEFAULT_HSGG_DEPTH_MODEL,
    DEFAULT_HSGG_LLMDET_MODEL,
    DEFAULT_HSGG_LLM_MODEL,
    DEFAULT_HSGG_REPO_ROOT,
    DEFAULT_HSGG_SAVE_INTERVAL,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCENE_GRAPH_DIR,
    DEFAULT_SCENE_GRAPH_PATTERN,
    DEFAULT_TARGET_FPS,
    DEFAULT_VIDEO_ROOT,
    DEFAULT_YOUCOOK2_ROOT,
)
from .scene_graph_adapter import load_scene_graph_as_stg_frames, scene_graph_output_dir


VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YouCook2 clip scene graphs by extracting frames and running HSGG.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_YOUCOOK2_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--scene-graph-dir", default=None)
    parser.add_argument("--scene-graph-pattern", default=DEFAULT_SCENE_GRAPH_PATTERN)
    parser.add_argument("--hsgg-repo-root", default=str(DEFAULT_HSGG_REPO_ROOT))
    parser.add_argument("--llmdet-model", default=str(DEFAULT_HSGG_LLMDET_MODEL))
    parser.add_argument("--hsgg-llm-model", default=str(DEFAULT_HSGG_LLM_MODEL))
    parser.add_argument("--depth-model", default=str(DEFAULT_HSGG_DEPTH_MODEL))
    parser.add_argument("--fps", type=float, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument("--cuda", type=int, default=DEFAULT_HSGG_CUDA_DEVICE)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--save-interval", type=int, default=DEFAULT_HSGG_SAVE_INTERVAL)
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--use-vcd", action="store_true")
    parser.add_argument("--skip-attributes", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--reextract-frames", action="store_true")
    parser.add_argument("--rerun-hsgg", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false")
    parser.set_defaults(show_progress=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser()
    frames_root = (
        Path(args.frames_root).expanduser()
        if args.frames_root
        else output_root / "intermediate" / Path(DEFAULT_FRAME_CACHE_DIR).name
    )
    scene_graph_dir = (
        Path(args.scene_graph_dir).expanduser()
        if args.scene_graph_dir
        else output_root / Path(DEFAULT_SCENE_GRAPH_DIR).name
    )
    summary = prepare_scene_graphs(
        manifest_path=Path(args.manifest).expanduser(),
        dataset_root=Path(args.dataset_root).expanduser(),
        output_root=output_root,
        video_root=Path(args.video_root).expanduser(),
        frames_root=frames_root,
        scene_graph_dir=scene_graph_dir,
        scene_graph_pattern=args.scene_graph_pattern,
        hsgg_repo_root=Path(args.hsgg_repo_root).expanduser(),
        llmdet_model=Path(args.llmdet_model).expanduser(),
        hsgg_llm_model=Path(args.hsgg_llm_model).expanduser(),
        depth_model=Path(args.depth_model).expanduser(),
        fps=args.fps,
        frame_stride=args.frame_stride,
        cuda_device=args.cuda,
        temperature=args.temperature,
        save_interval=args.save_interval,
        allowed_splits=parse_csv_strings(args.splits),
        max_clips=args.max_clips,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        save_images=args.save_images,
        use_vcd=args.use_vcd,
        skip_attributes=args.skip_attributes,
        resume=args.resume,
        reextract_frames=args.reextract_frames,
        rerun_hsgg=args.rerun_hsgg,
        dry_run=args.dry_run,
        show_progress=args.show_progress,
    )
    print(
        f"Scene graph prep finished: clips={summary['selected_clips']} ready={summary['already_ready']} "
        f"generated={summary['generated']} converted={summary['converted_existing']} "
        f"planned={summary['planned']} errors={summary['errors']}"
    )
    print(f"Summary log: {summary['summary_path']}")


def prepare_scene_graphs(
    *,
    manifest_path: Path,
    dataset_root: Path,
    output_root: Path,
    video_root: Path,
    frames_root: Path,
    scene_graph_dir: Path,
    scene_graph_pattern: str,
    hsgg_repo_root: Path,
    llmdet_model: Path,
    hsgg_llm_model: Path,
    depth_model: Path,
    fps: float,
    frame_stride: int,
    cuda_device: int,
    temperature: float,
    save_interval: int,
    allowed_splits: list[str],
    max_clips: int | None,
    num_shards: int,
    shard_index: int,
    save_images: bool,
    use_vcd: bool,
    skip_attributes: bool,
    resume: bool,
    reextract_frames: bool,
    rerun_hsgg: bool,
    dry_run: bool,
    show_progress: bool = True,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be > 0")
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    manifest = read_json(manifest_path)
    clips = [
        clip
        for clip in manifest.get("clips", [])
        if not allowed_splits or str(clip.get("split", "")) in set(allowed_splits)
    ]
    if max_clips is not None:
        clips = clips[:max_clips]
    total_clips_before_shard = len(clips)
    if num_shards > 1:
        clips = [clip for index, clip in enumerate(clips) if index % num_shards == shard_index]

    paths = ensure_output_tree(output_root)
    frames_root.mkdir(parents=True, exist_ok=True)
    scene_graph_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        _validate_hsgg_repo_root(hsgg_repo_root)

    shard_label = f".shard{shard_index:02d}-of-{num_shards:02d}" if num_shards > 1 else ""
    event_log_path = paths["logs"] / f"prepare_scene_graphs{shard_label}.jsonl"
    error_log_path = paths["logs"] / f"prepare_scene_graphs{shard_label}.errors.jsonl"
    summary_path = paths["logs"] / f"prepare_scene_graphs{shard_label}.summary.json"
    job_dump_dir = paths["intermediate"] / "scene_graph_jobs"
    job_dump_dir.mkdir(parents=True, exist_ok=True)

    video_index: dict[str, list[Path]] | None = None
    generator = None
    summary = {
        "generated_at": now_utc_iso(),
        "manifest": str(manifest_path),
        "dataset_root": str(dataset_root),
        "video_root": str(video_root),
        "scene_graph_dir": str(scene_graph_dir),
        "frames_root": str(frames_root),
        "hsgg_repo_root": str(hsgg_repo_root),
        "total_clips_before_shard": total_clips_before_shard,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "selected_clips": len(clips),
        "already_ready": 0,
        "converted_existing": 0,
        "generated": 0,
        "planned": 0,
        "errors": 0,
        "summary_path": str(summary_path),
        "last_update_at": now_utc_iso(),
        "last_clip": None,
        "last_status": "started",
    }
    _write_progress_summary(summary_path, summary, "started")

    for progress_index, clip in enumerate(clips, start=1):
        clip_id = str(clip.get("clip_id", ""))
        video_id = str(clip.get("video_id", ""))
        image_dir_value = str(clip.get("image_dir", "")).strip()
        clip_scene_dir = scene_graph_output_dir(
            clip,
            scene_graph_dir=scene_graph_dir,
            scene_graph_pattern=scene_graph_pattern,
        )
        clip_scene_dir.mkdir(parents=True, exist_ok=True)
        stg_input_path = clip_scene_dir / "stg_input.json"
        scene_graph_json_path = clip_scene_dir / "scene_graphs.json"
        clip_frame_dir = frames_root / video_id / clip_id

        try:
            if show_progress:
                print(
                    f"[scene-graph {progress_index}/{len(clips)} shard {shard_index + 1}/{num_shards}] "
                    f"start video_id={video_id} clip_id={clip_id}",
                    flush=True,
                )

            if resume and stg_input_path.exists() and not rerun_hsgg:
                summary["already_ready"] += 1
                append_jsonl(
                    event_log_path,
                    [
                        {
                            "timestamp": now_utc_iso(),
                            "clip_id": clip_id,
                            "video_id": video_id,
                            "status": "ready",
                            "scene_graph_dir": str(clip_scene_dir),
                            "reason": "stg_input_exists",
                        }
                    ],
                )
                _write_progress_summary(
                    summary_path, summary, "ready", clip=clip, progress_index=progress_index
                )
                if show_progress:
                    print(
                        f"[scene-graph {progress_index}/{len(clips)} shard {shard_index + 1}/{num_shards}] "
                        f"ready video_id={video_id} clip_id={clip_id}",
                        flush=True,
                    )
                continue

            if resume and scene_graph_json_path.exists() and not rerun_hsgg:
                if not dry_run:
                    frames = load_scene_graph_as_stg_frames(clip_scene_dir)
                    write_json(stg_input_path, frames)
                    _write_job_dump(
                        job_dump_dir / f"{clip_id}.json",
                        clip=clip,
                        status="converted_existing",
                        video_paths=None,
                        frame_dir=clip_frame_dir,
                        scene_graph_dir=clip_scene_dir,
                        frame_count=len(frames),
                    )
                summary["converted_existing"] += 1
                append_jsonl(
                    event_log_path,
                    [
                        {
                            "timestamp": now_utc_iso(),
                            "clip_id": clip_id,
                            "video_id": video_id,
                            "status": "converted_existing",
                            "scene_graph_dir": str(clip_scene_dir),
                        }
                    ],
                )
                _write_progress_summary(
                    summary_path, summary, "converted_existing", clip=clip, progress_index=progress_index
                )
                if show_progress:
                    print(
                        f"[scene-graph {progress_index}/{len(clips)} shard {shard_index + 1}/{num_shards}] "
                        f"converted_existing video_id={video_id} clip_id={clip_id}",
                        flush=True,
                    )
                continue

            source_video_paths: list[Path] = []

            if image_dir_value:
                frame_count = ensure_clip_frames_from_image_dir(
                    image_dir=Path(image_dir_value).expanduser(),
                    frame_dir=clip_frame_dir,
                    frame_stride=frame_stride,
                    resume=resume,
                    reextract_frames=reextract_frames,
                    dry_run=dry_run,
                )
            else:
                if video_index is None:
                    video_index = build_video_index(video_root if video_root.exists() else dataset_root)
                source_video_paths = resolve_source_video_paths(
                    clip,
                    video_index,
                    dataset_root=dataset_root,
                    video_root=video_root,
                )
                clip_start_sec = float(clip.get("clip_start_sec", 0.0))
                clip_end_sec = float(clip.get("clip_end_sec", 0.0))
                has_valid_clip_range = clip_end_sec > clip_start_sec

                if clip.get("source_clip_paths") or not has_valid_clip_range:
                    frame_count = ensure_clip_frames_from_source_clips(
                        video_paths=source_video_paths,
                        clip=clip,
                        frame_dir=clip_frame_dir,
                        fps=fps,
                        frame_stride=frame_stride,
                        resume=resume,
                        reextract_frames=reextract_frames,
                        dry_run=dry_run,
                    )
                else:
                    frame_count = ensure_clip_frames(
                        video_path=source_video_paths[0],
                        frame_dir=clip_frame_dir,
                        clip_start_sec=clip_start_sec,
                        clip_end_sec=clip_end_sec,
                        fps=fps,
                        frame_stride=frame_stride,
                        resume=resume,
                        reextract_frames=reextract_frames,
                        dry_run=dry_run,
                    )

            if dry_run:
                summary["planned"] += 1
                append_jsonl(
                    event_log_path,
                    [
                        {
                            "timestamp": now_utc_iso(),
                            "clip_id": clip_id,
                            "video_id": video_id,
                            "status": "planned",
                            "image_dir": image_dir_value,
                            "video_paths": [str(path) for path in source_video_paths],
                            "frame_dir": str(clip_frame_dir),
                            "scene_graph_dir": str(clip_scene_dir),
                            "frame_count": frame_count,
                        }
                    ],
                )
                _write_progress_summary(
                    summary_path, summary, "planned", clip=clip, progress_index=progress_index
                )
                if show_progress:
                    print(
                        f"[scene-graph {progress_index}/{len(clips)} shard {shard_index + 1}/{num_shards}] "
                        f"planned video_id={video_id} clip_id={clip_id} frames={frame_count}",
                        flush=True,
                    )
                continue

            if generator is None:
                generator = instantiate_hsgg_generator(
                    hsgg_repo_root=hsgg_repo_root,
                    llmdet_model=llmdet_model,
                    hsgg_llm_model=hsgg_llm_model,
                    depth_model=depth_model,
                    cuda_device=cuda_device,
                    temperature=temperature,
                    use_vcd=use_vcd,
                    skip_attributes=skip_attributes,
                )

            image_paths = [
                str(path)
                for path in sorted(clip_frame_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
            if not image_paths:
                raise RuntimeError(f"No extracted clip frames found under {clip_frame_dir}")

            stage_start = time.time()
            generator.generate_batch(
                image_paths=image_paths,
                output_dir=str(clip_scene_dir),
                save_interval=save_interval,
                save_images=save_images,
            )
            frames = load_scene_graph_as_stg_frames(clip_scene_dir)
            write_json(stg_input_path, frames)
            elapsed = round(time.time() - stage_start, 3)

            _write_job_dump(
                job_dump_dir / f"{clip_id}.json",
                clip=clip,
                status="generated",
                video_paths=source_video_paths,
                frame_dir=clip_frame_dir,
                scene_graph_dir=clip_scene_dir,
                frame_count=len(frames),
                elapsed_sec=elapsed,
            )
            summary["generated"] += 1
            append_jsonl(
                event_log_path,
                [
                    {
                        "timestamp": now_utc_iso(),
                        "clip_id": clip_id,
                        "video_id": video_id,
                        "image_dir": image_dir_value,
                        "status": "generated",
                        "video_paths": [str(path) for path in source_video_paths],
                        "frame_dir": str(clip_frame_dir),
                        "scene_graph_dir": str(clip_scene_dir),
                        "frame_count": len(frames),
                        "elapsed_sec": elapsed,
                    }
                ],
            )
            _write_progress_summary(
                summary_path, summary, "generated", clip=clip, progress_index=progress_index
            )
            if show_progress:
                print(
                    f"[scene-graph {progress_index}/{len(clips)} shard {shard_index + 1}/{num_shards}] "
                    f"generated video_id={video_id} clip_id={clip_id} frames={len(frames)} elapsed={elapsed}s",
                    flush=True,
                )
        except Exception as exc:
            summary["errors"] += 1
            append_jsonl(
                error_log_path,
                [
                    {
                        "timestamp": now_utc_iso(),
                        "clip_id": clip_id,
                        "video_id": video_id,
                        "image_dir": image_dir_value,
                        "scene_graph_dir": str(clip_scene_dir),
                        "error": str(exc),
                    }
                ],
            )
            _write_progress_summary(
                summary_path, summary, "error", clip=clip, progress_index=progress_index
            )
            if show_progress:
                print(
                    f"[scene-graph {progress_index}/{len(clips)} shard {shard_index + 1}/{num_shards}] "
                    f"error video_id={video_id} clip_id={clip_id}: {exc}",
                    flush=True,
                )

    _write_progress_summary(summary_path, summary, "finished")
    write_json(summary_path, summary)
    return summary


def _validate_hsgg_repo_root(hsgg_repo_root: Path) -> None:
    repo_root = hsgg_repo_root.expanduser().resolve()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"HSGG repo root does not exist: {repo_root}. "
            "Pass the real path with --hsgg-repo-root /path/to/HSGG-main."
        )
    if not (repo_root / "scene_graph_generator.py").exists():
        raise FileNotFoundError(
            f"HSGG repo root is missing scene_graph_generator.py: {repo_root}. "
            "Check that --hsgg-repo-root points to the HSGG-main repository directory."
        )


def _write_progress_summary(
    summary_path: Path,
    summary: dict[str, Any],
    status: str,
    *,
    clip: dict[str, Any] | None = None,
    progress_index: int | None = None,
) -> None:
    summary["last_update_at"] = now_utc_iso()
    summary["last_status"] = status
    if progress_index is not None:
        summary["last_progress_index"] = progress_index
    if clip is not None:
        summary["last_clip"] = {
            "clip_id": clip.get("clip_id"),
            "video_id": clip.get("video_id"),
            "sample_id": clip.get("sample_id"),
            "dataset_source": clip.get("dataset_source"),
        }
    write_json(summary_path, summary)


def build_video_index(root: Path) -> dict[str, list[Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Video root does not exist: {root}")

    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        index.setdefault(path.stem.lower(), []).append(path)
    return index


def resolve_source_video_paths(
    clip: dict[str, Any],
    video_index: dict[str, list[Path]],
    *,
    dataset_root: Path,
    video_root: Path,
) -> list[Path]:
    source_clip_paths = clip.get("source_clip_paths")
    if isinstance(source_clip_paths, list) and source_clip_paths:
        return [
            _resolve_explicit_video_path(str(item), video_index, dataset_root=dataset_root, video_root=video_root)
            for item in source_clip_paths
        ]

    explicit = str(clip.get("video_path", "")).strip()
    if explicit:
        return [_resolve_explicit_video_path(explicit, video_index, dataset_root=dataset_root, video_root=video_root)]

    video_id = str(clip.get("video_id", "")).strip()
    if not video_id:
        raise ValueError("Clip is missing video_id")

    candidates = list(video_index.get(video_id.lower(), []))
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate raw video for {video_id}. "
            f"Searched under {video_root if video_root.exists() else dataset_root}."
        )

    subset = str(clip.get("source_subset", clip.get("subset", ""))).strip().lower()
    if subset:
        subset_matches = [
            path
            for path in candidates
            if any(part.lower() == subset for part in path.parts)
        ]
        if len(subset_matches) == 1:
            return [subset_matches[0]]
        if subset_matches:
            candidates = subset_matches

    return [sorted(candidates, key=lambda path: (len(path.parts), len(str(path))))[0]]


def _resolve_explicit_video_path(
    explicit: str,
    video_index: dict[str, list[Path]],
    *,
    dataset_root: Path,
    video_root: Path,
) -> Path:
    explicit_path = Path(explicit).expanduser()
    if explicit_path.exists():
        return explicit_path

    for root in (dataset_root, video_root):
        candidate = root / explicit
        if candidate.exists():
            return candidate

    stem = explicit_path.stem.lower()
    candidates = video_index.get(stem, [])
    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(f"Manifest video_path does not exist: {explicit}")


def ensure_clip_frames(
    *,
    video_path: Path,
    frame_dir: Path,
    clip_start_sec: float,
    clip_end_sec: float,
    fps: float,
    frame_stride: int,
    resume: bool,
    reextract_frames: bool,
    dry_run: bool,
) -> int:
    expected_frame_count = _estimate_clip_frame_count(
        video_path=video_path,
        clip_start_sec=clip_start_sec,
        clip_end_sec=clip_end_sec,
        fps_hint=fps,
        frame_stride=frame_stride,
    )
    if resume and not reextract_frames:
        existing = [
            path
            for path in sorted(frame_dir.glob("frame_*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if len(existing) >= expected_frame_count:
            return len(existing)

    if dry_run:
        return expected_frame_count

    extract_clip_frames(
        video_path=video_path,
        frame_dir=frame_dir,
        clip_start_sec=clip_start_sec,
        clip_end_sec=clip_end_sec,
        fps=fps,
        frame_stride=frame_stride,
    )
    existing = [
        path
        for path in sorted(frame_dir.glob("frame_*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not existing:
        raise RuntimeError(f"Frame extraction produced no frames for {video_path}")
    return len(existing)


def ensure_clip_frames_from_source_clips(
    *,
    video_paths: list[Path],
    clip: dict[str, Any],
    frame_dir: Path,
    fps: float,
    frame_stride: int,
    resume: bool,
    reextract_frames: bool,
    dry_run: bool,
) -> int:
    expected_frame_count = _estimate_source_clip_frame_count(
        clip=clip,
        video_paths=video_paths,
        fps_hint=fps,
        frame_stride=frame_stride,
    )
    if resume and not reextract_frames:
        existing = [
            path
            for path in sorted(frame_dir.glob("frame_*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if len(existing) >= expected_frame_count:
            return len(existing)

    if dry_run:
        return expected_frame_count

    extract_frames_from_source_clips(
        video_paths=video_paths,
        frame_dir=frame_dir,
        fps=fps,
        frame_stride=frame_stride,
    )
    existing = [
        path
        for path in sorted(frame_dir.glob("frame_*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not existing:
        raise RuntimeError(f"Frame extraction produced no frames for source clips: {video_paths}")
    return len(existing)


def ensure_clip_frames_from_image_dir(
    *,
    image_dir: Path,
    frame_dir: Path,
    frame_stride: int,
    resume: bool,
    reextract_frames: bool,
    dry_run: bool,
) -> int:
    """从现成图像目录中按固定步长抽样并复制到帧缓存目录。"""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    source_images = _list_source_images(image_dir)
    if not source_images:
        raise RuntimeError(f"No source images found under {image_dir}")

    selected_images = source_images[::frame_stride] or source_images[:1]
    expected_frame_count = len(selected_images)

    if resume and not reextract_frames:
        existing = [
            path
            for path in sorted(frame_dir.glob("frame_*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if len(existing) >= expected_frame_count:
            return len(existing)

    if dry_run:
        return expected_frame_count

    copy_sampled_images(
        source_images=selected_images,
        frame_dir=frame_dir,
        frame_stride=frame_stride,
        source_dir=image_dir,
    )
    existing = [
        path
        for path in sorted(frame_dir.glob("frame_*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not existing:
        raise RuntimeError(f"Image directory sampling produced no frames for {image_dir}")
    return len(existing)


def extract_clip_frames(
    *,
    video_path: Path,
    frame_dir: Path,
    clip_start_sec: float,
    clip_end_sec: float,
    fps: float,
    frame_stride: int,
) -> None:
    import cv2

    if clip_end_sec <= clip_start_sec:
        raise ValueError(f"Invalid clip range: {clip_start_sec} -> {clip_end_sec}")

    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*"):
        if old_frame.is_file():
            old_frame.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    saved = 0
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps and source_fps > 0:
        start_frame = max(int(clip_start_sec * source_fps), 0)
        end_frame = max(int(clip_end_sec * source_fps), start_frame + 1)
        frame_indices = list(range(start_frame, end_frame, frame_stride))
        if not frame_indices:
            frame_indices = [start_frame]
    else:
        # Fall back to the previous time-based behavior if FPS metadata is unavailable.
        timestamps = _build_time_based_timestamps(clip_start_sec=clip_start_sec, clip_end_sec=clip_end_sec, fps=fps)
        frame_indices = []

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_path = frame_dir / f"frame_{idx:06d}.jpg"
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"Failed to write frame image: {frame_path}")
        saved += 1

    if not source_fps or source_fps <= 0:
        for idx, timestamp_sec in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(timestamp_sec, 0.0) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_path = frame_dir / f"frame_{idx:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Failed to write frame image: {frame_path}")
            saved += 1

    cap.release()
    if saved == 0:
        raise RuntimeError(f"Could not extract any frames from {video_path}")

    write_json(
        frame_dir / "frames_manifest.json",
        {
            "video_path": str(video_path),
            "clip_start_sec": clip_start_sec,
            "clip_end_sec": clip_end_sec,
            "target_fps": fps,
            "frame_stride": frame_stride,
            "saved_frames": saved,
            "created_at": now_utc_iso(),
        },
    )


def extract_frames_from_source_clips(
    *,
    video_paths: list[Path],
    frame_dir: Path,
    fps: float,
    frame_stride: int,
) -> None:
    import cv2

    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*"):
        if old_frame.is_file():
            old_frame.unlink()

    global_idx = 0
    clip_manifests: list[dict[str, Any]] = []

    for clip_idx, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open source clip video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        saved_here = 0
        if total_frames > 0:
            frame_indices = list(range(0, total_frames, frame_stride))
            if not frame_indices:
                frame_indices = [0]
        else:
            frame_indices = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_path = frame_dir / f"frame_{global_idx:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Failed to write frame image: {frame_path}")
            global_idx += 1
            saved_here += 1

        if total_frames <= 0:
            timestamps_ms = _build_time_based_timestamps(
                clip_start_sec=0.0,
                clip_end_sec=max(1.0, _safe_duration_from_fps(total_frames=total_frames, fps=video_fps)),
                fps=fps,
            )
            for timestamp_ms in [timestamp * 1000.0 for timestamp in timestamps_ms]:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(timestamp_ms, 0.0))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frame_path = frame_dir / f"frame_{global_idx:06d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Failed to write frame image: {frame_path}")
                global_idx += 1
                saved_here += 1

        cap.release()
        clip_manifests.append(
            {
                "clip_index": clip_idx,
                "video_path": str(video_path),
                "saved_frames": saved_here,
                "frame_stride": frame_stride,
            }
        )

    if global_idx == 0:
        raise RuntimeError(f"Could not extract any frames from source clips: {video_paths}")

    write_json(
        frame_dir / "frames_manifest.json",
        {
            "video_paths": [str(path) for path in video_paths],
            "target_fps": fps,
            "frame_stride": frame_stride,
            "saved_frames": global_idx,
            "source_clip_count": len(video_paths),
            "source_clips": clip_manifests,
            "created_at": now_utc_iso(),
        },
    )


def copy_sampled_images(
    *,
    source_images: list[Path],
    frame_dir: Path,
    frame_stride: int,
    source_dir: Path,
) -> None:
    """把采样后的原始图像复制到 STG 统一使用的帧目录。"""
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*"):
        if old_frame.is_file():
            old_frame.unlink()

    saved = 0
    source_manifest: list[dict[str, Any]] = []
    for idx, source_path in enumerate(source_images):
        ext = source_path.suffix.lower() if source_path.suffix else ".jpg"
        target_path = frame_dir / f"frame_{idx:06d}{ext}"
        shutil.copy2(source_path, target_path)
        source_manifest.append(
            {
                "frame_index": idx,
                "source_path": str(source_path),
                "target_path": str(target_path),
            }
        )
        saved += 1

    write_json(
        frame_dir / "frames_manifest.json",
        {
            "image_dir": str(source_dir),
            "frame_stride": frame_stride,
            "saved_frames": saved,
            "source_images": source_manifest,
            "created_at": now_utc_iso(),
        },
    )


def _list_source_images(image_dir: Path) -> list[Path]:
    """列出图像目录中的原始图片，按路径名稳定排序。"""
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _estimate_clip_frame_count(
    *,
    video_path: Path,
    clip_start_sec: float,
    clip_end_sec: float,
    fps_hint: float,
    frame_stride: int,
) -> int:
    try:
        import cv2
    except ImportError:
        return max(int((clip_end_sec - clip_start_sec) * fps_hint + 1e-9), 1) if clip_end_sec > clip_start_sec else 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return max(int((clip_end_sec - clip_start_sec) * fps_hint + 1e-9), 1)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not source_fps or source_fps <= 0:
        return max(int((clip_end_sec - clip_start_sec) * fps_hint + 1e-9), 1)

    start_frame = max(int(clip_start_sec * source_fps), 0)
    end_frame = max(int(clip_end_sec * source_fps), start_frame + 1)
    span = max(end_frame - start_frame, 1)
    return max((span + frame_stride - 1) // frame_stride, 1)


def _estimate_source_clip_frame_count(
    *,
    clip: dict[str, Any],
    video_paths: list[Path],
    fps_hint: float,
    frame_stride: int,
) -> int:
    try:
        import cv2
    except ImportError:
        return max(len(video_paths), 1)

    total = 0
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            total += _fallback_step_frame_count(clip=clip, fps_hint=fps_hint)
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames > 0:
            total += max((total_frames + frame_stride - 1) // frame_stride, 1)
        else:
            total += _fallback_step_frame_count(clip=clip, fps_hint=fps_hint)
    return max(total, 1)


def _fallback_step_frame_count(*, clip: dict[str, Any], fps_hint: float) -> int:
    total_duration = 0.0
    for step in clip.get("steps", []) or []:
        try:
            total_duration += max(float(step.get("end_sec", 0.0)) - float(step.get("start_sec", 0.0)), 0.0)
        except Exception:
            continue
    return max(int(total_duration * fps_hint + 1e-9), 1)


def _build_time_based_timestamps(*, clip_start_sec: float, clip_end_sec: float, fps: float) -> list[float]:
    timestamps: list[float] = []
    interval = 1.0 / fps
    current = clip_start_sec
    while current < clip_end_sec:
        timestamps.append(current)
        current += interval
    return timestamps or [clip_start_sec]


def _safe_duration_from_fps(*, total_frames: int, fps: float) -> float:
    if fps and fps > 0 and total_frames > 0:
        return total_frames / fps
    return 1.0


def instantiate_hsgg_generator(
    *,
    hsgg_repo_root: Path,
    llmdet_model: Path,
    hsgg_llm_model: Path,
    depth_model: Path,
    cuda_device: int,
    temperature: float,
    use_vcd: bool,
    skip_attributes: bool,
):
    repo_root = hsgg_repo_root.expanduser().resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"HSGG repo root does not exist: {repo_root}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    importlib.invalidate_caches()
    module = importlib.import_module("scene_graph_generator")
    SceneGraphGenerator = getattr(module, "SceneGraphGenerator")
    return SceneGraphGenerator(
        llmdet_model_path=str(llmdet_model),
        llm_model_path=str(hsgg_llm_model),
        depth_model_path=str(depth_model),
        cuda_device=cuda_device,
        temperature=temperature,
        use_vcd=use_vcd,
        skip_attributes=skip_attributes,
    )


def _write_job_dump(
    path: Path,
    *,
    clip: dict[str, Any],
    status: str,
    video_paths: list[Path] | None,
    frame_dir: Path,
    scene_graph_dir: Path,
    frame_count: int,
    elapsed_sec: float | None = None,
) -> None:
    write_json(
        path,
        {
            "timestamp": now_utc_iso(),
            "status": status,
            "clip_id": clip.get("clip_id"),
            "video_id": clip.get("video_id"),
            "split": clip.get("split"),
            "sample_id": clip.get("sample_id"),
            "image_dir": clip.get("image_dir"),
            "dataset_source": clip.get("dataset_source"),
            "video_path": None if not video_paths else str(video_paths[0]),
            "video_paths": [] if not video_paths else [str(path) for path in video_paths],
            "frame_dir": str(frame_dir),
            "scene_graph_dir": str(scene_graph_dir),
            "frame_count": frame_count,
            "elapsed_sec": elapsed_sec,
        },
    )


if __name__ == "__main__":
    main()
