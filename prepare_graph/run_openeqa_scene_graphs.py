from __future__ import annotations

"""OpenEQA pilot 批量场景图生成入口。

相比直接调用 `prepare_scene_graphs.py`，这个脚本会：
1. 使用 OpenEQA 默认目录约定
2. 默认打开 `save_images`
3. 强制只处理 pilot manifest 中的 episode
"""

import argparse
import subprocess
import sys
from pathlib import Path

from .common import parse_csv_strings
from .defaults import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_HSGG_CUDA_DEVICE,
    DEFAULT_HSGG_DEPTH_MODEL,
    DEFAULT_HSGG_LLMDET_MODEL,
    DEFAULT_HSGG_LLM_MODEL,
    DEFAULT_HSGG_REPO_ROOT,
    DEFAULT_HSGG_SAVE_INTERVAL,
    DEFAULT_OPENEQA_FRAMES_DIR,
    DEFAULT_OPENEQA_MANIFEST_PATH,
    DEFAULT_OPENEQA_OUTPUT_ROOT,
    DEFAULT_OPENEQA_SCENE_GRAPH_DIR,
    DEFAULT_SCENE_GRAPH_PATTERN,
    DEFAULT_TARGET_FPS,
)
from .prepare_scene_graphs import prepare_scene_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量生成 OpenEQA pilot 的场景图与 STG 输入。")
    parser.add_argument("--manifest", default=str(DEFAULT_OPENEQA_MANIFEST_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OPENEQA_OUTPUT_ROOT))
    parser.add_argument("--frames-root", default=str(DEFAULT_OPENEQA_FRAMES_DIR))
    parser.add_argument("--scene-graph-dir", default=str(DEFAULT_OPENEQA_SCENE_GRAPH_DIR))
    parser.add_argument("--scene-graph-pattern", default=DEFAULT_SCENE_GRAPH_PATTERN)
    parser.add_argument("--hsgg-repo-root", default=str(DEFAULT_HSGG_REPO_ROOT))
    parser.add_argument("--llmdet-model", default=str(DEFAULT_HSGG_LLMDET_MODEL))
    parser.add_argument("--hsgg-llm-model", default=str(DEFAULT_HSGG_LLM_MODEL))
    parser.add_argument("--depth-model", default=str(DEFAULT_HSGG_DEPTH_MODEL))
    parser.add_argument("--fps", type=float, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument("--cuda", type=int, default=DEFAULT_HSGG_CUDA_DEVICE)
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--save-interval", type=int, default=DEFAULT_HSGG_SAVE_INTERVAL)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-images", dest="save_images", action="store_true")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    parser.set_defaults(save_images=True)
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
    cuda_devices = _parse_cuda_devices(args.cuda_devices)
    if len(cuda_devices) > 1:
        _run_cuda_shards(args, cuda_devices)
        return
    if len(cuda_devices) == 1:
        args.cuda = cuda_devices[0]

    summary = prepare_scene_graphs(
        manifest_path=Path(args.manifest).expanduser().resolve(),
        dataset_root=Path(args.output_root).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        video_root=Path(args.output_root).expanduser().resolve(),
        frames_root=Path(args.frames_root).expanduser().resolve(),
        scene_graph_dir=Path(args.scene_graph_dir).expanduser().resolve(),
        scene_graph_pattern=args.scene_graph_pattern,
        hsgg_repo_root=Path(args.hsgg_repo_root).expanduser().resolve(),
        llmdet_model=Path(args.llmdet_model).expanduser().resolve(),
        hsgg_llm_model=Path(args.hsgg_llm_model).expanduser().resolve(),
        depth_model=Path(args.depth_model).expanduser().resolve(),
        fps=args.fps,
        frame_stride=args.frame_stride,
        cuda_device=args.cuda,
        temperature=args.temperature,
        save_interval=args.save_interval,
        allowed_splits=["test"],
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

    print("OpenEQA scene graph generation finished")
    print(f"Selected clips: {summary['selected_clips']}")
    print(f"Generated: {summary['generated']}")
    print(f"Converted existing: {summary['converted_existing']}")
    print(f"Already ready: {summary['already_ready']}")
    print(f"Errors: {summary['errors']}")
    print(f"Summary log: {summary['summary_path']}")


def _parse_cuda_devices(value: str | None) -> list[int]:
    return [int(item) for item in parse_csv_strings(value)]


def _run_cuda_shards(args: argparse.Namespace, cuda_devices: list[int]) -> None:
    if args.num_shards != 1 or args.shard_index != 0:
        raise ValueError("--cuda-devices cannot be combined with explicit --num-shards/--shard-index")

    processes: list[tuple[int, subprocess.Popen[bytes]]] = []
    for shard_index, cuda_device in enumerate(cuda_devices):
        command = _build_shard_command(
            args,
            cuda_device=cuda_device,
            num_shards=len(cuda_devices),
            shard_index=shard_index,
        )
        print(
            f"Starting shard {shard_index + 1}/{len(cuda_devices)} on cuda:{cuda_device}",
            flush=True,
        )
        processes.append((cuda_device, subprocess.Popen(command)))

    failed: list[tuple[int, int]] = []
    for cuda_device, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((cuda_device, return_code))

    if failed:
        details = ", ".join(f"cuda:{device} rc={return_code}" for device, return_code in failed)
        raise SystemExit(f"One or more scene graph shards failed: {details}")

    print("OpenEQA scene graph generation finished on all CUDA shards")


def _build_shard_command(
    args: argparse.Namespace,
    *,
    cuda_device: int,
    num_shards: int,
    shard_index: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "prepare_graph.run_openeqa_scene_graphs",
        "--manifest",
        str(args.manifest),
        "--output-root",
        str(args.output_root),
        "--frames-root",
        str(args.frames_root),
        "--scene-graph-dir",
        str(args.scene_graph_dir),
        "--scene-graph-pattern",
        str(args.scene_graph_pattern),
        "--hsgg-repo-root",
        str(args.hsgg_repo_root),
        "--llmdet-model",
        str(args.llmdet_model),
        "--hsgg-llm-model",
        str(args.hsgg_llm_model),
        "--depth-model",
        str(args.depth_model),
        "--fps",
        str(args.fps),
        "--frame-stride",
        str(args.frame_stride),
        "--cuda",
        str(cuda_device),
        "--temperature",
        str(args.temperature),
        "--save-interval",
        str(args.save_interval),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
    ]
    if args.max_clips is not None:
        command.extend(["--max-clips", str(args.max_clips)])
    if args.save_images:
        command.append("--save-images")
    else:
        command.append("--no-save-images")
    if args.use_vcd:
        command.append("--use-vcd")
    if args.skip_attributes:
        command.append("--skip-attributes")
    if not args.resume:
        command.append("--no-resume")
    if args.reextract_frames:
        command.append("--reextract-frames")
    if args.rerun_hsgg:
        command.append("--rerun-hsgg")
    if args.dry_run:
        command.append("--dry-run")
    if not args.show_progress:
        command.append("--no-progress")
    return command


if __name__ == "__main__":
    main()
