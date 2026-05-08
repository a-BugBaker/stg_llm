from __future__ import annotations

"""批量把 OpenEQA pilot 的 `stg_input.json` 构建为语义时空图。"""

import argparse
import json
import os
from pathlib import Path

from prepare_graph.defaults import (
    DEFAULT_OPENEQA_MANIFEST_PATH,
    DEFAULT_OPENEQA_OUTPUT_ROOT,
    DEFAULT_OPENEQA_SCENE_GRAPH_DIR,
    DEFAULT_OPENEQA_STG_DIR,
)
from prepare_graph.scene_graph_adapter import scene_graph_output_dir

from .config import EngineConfig, LLMConfig
from .pipeline import SpatialTemporalPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量构建 OpenEQA pilot 的 STG 快照。")
    parser.add_argument("--manifest", default=str(DEFAULT_OPENEQA_MANIFEST_PATH))
    parser.add_argument("--scene-graph-dir", default=str(DEFAULT_OPENEQA_SCENE_GRAPH_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OPENEQA_STG_DIR))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--use-neo4j", action="store_true")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--neo4j-database", default="stg.llm")
    parser.add_argument("--enable-llm", action="store_true", help="启用 STG 构建阶段的 LLM 判定。")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--llm-timeout", type=int, default=40)
    parser.add_argument("--llm-timeout-retries", type=int, default=0)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).expanduser().resolve().read_text(encoding="utf-8"))
    scene_graph_dir = Path(args.scene_graph_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    clips = list(manifest.get("clips", []))
    if args.max_clips is not None:
        clips = clips[:args.max_clips]

    summaries: list[dict[str, object]] = []
    for clip in clips:
        sample_id = str(clip.get("sample_id", "unknown"))
        clip_scene_dir = scene_graph_output_dir(
            clip,
            scene_graph_dir=scene_graph_dir,
            scene_graph_pattern="{video_id}/{clip_id}",
        )
        stg_input_path = clip_scene_dir / "stg_input.json"
        if not stg_input_path.exists():
            raise FileNotFoundError(f"STG input not found: {stg_input_path}")

        sample_output_dir = output_root / sample_id
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = sample_output_dir / "semantic_graph_snapshot.json"
        report_path = sample_output_dir / "design_acceptance_report.json"
        if args.resume and snapshot_path.exists() and report_path.exists():
            summaries.append(
                {
                    "sample_id": sample_id,
                    "status": "skipped_existing",
                    "input": str(stg_input_path),
                    "snapshot": str(snapshot_path),
                    "report": str(report_path),
                }
            )
            continue

        cfg = EngineConfig(
            use_neo4j=args.use_neo4j,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
            sample_id=sample_id,
            llm=LLMConfig(
                enabled=args.enable_llm,
                base_url=args.llm_base_url,
                api_key=args.llm_api_key,
                model=args.llm_model,
                timeout_seconds=args.llm_timeout,
                timeout_retries=args.llm_timeout_retries,
                temperature=args.llm_temperature,
            ),
        )

        pipeline = SpatialTemporalPipeline(config=cfg)
        summary = pipeline.run(input_json_path=str(stg_input_path), max_frames=args.max_frames)
        pipeline.export_graph_snapshot(str(snapshot_path))
        report = pipeline.build_acceptance_report(summary)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        summaries.append(
            {
                "sample_id": sample_id,
                "status": "built",
                "input": str(stg_input_path),
                "snapshot": str(snapshot_path),
                "report": str(report_path),
                "summary": {
                    "frames": summary.total_frames,
                    "processed_objects": summary.total_objects,
                    "new_nodes": summary.total_new_nodes,
                    "updated_nodes": summary.total_updated_nodes,
                    "new_edges": summary.total_new_edges,
                    "duplicate_edges": summary.total_duplicate_edges,
                    "conflict_edges": summary.total_conflict_edges,
                    "owner_assigned": summary.total_owner_assigned,
                },
            }
        )

    summary_path = output_root / "build_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OpenEQA STG build finished")
    print(f"Summary: {summary_path}")
    print(f"Episodes: {len(summaries)}")


if __name__ == "__main__":
    main()
