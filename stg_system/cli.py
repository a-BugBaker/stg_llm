from __future__ import annotations

"""命令行入口。

用途：
1. 解析运行参数（输入、输出、Neo4j、LLM）。
2. 组装 EngineConfig。
3. 执行流水线并输出快照与验收报告。
"""

import argparse
import json
import os
from pathlib import Path

from .config import EngineConfig, LLMConfig
from .pipeline import SpatialTemporalPipeline


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Spatial-temporal graph offline builder")
    parser.add_argument("--input", required=True, help="Input frame JSON path")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to store snapshot and acceptance report",
    )
    parser.add_argument(
        "--sample-id",
        default="default_sample",
        help="Sample identifier for storage/output isolation",
    )
    parser.add_argument(
        "--snapshot-output",
        default=None,
        help="Output path for semantic graph snapshot",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Output path for design acceptance report",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Process only first N frames")
    parser.add_argument("--use-neo4j", action="store_true", help="Write nodes and edges to Neo4j")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--neo4j-database",default="stg.llm")
    parser.add_argument("--enable-llm", action="store_true", help="Enable LLM decisions")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--llm-timeout", type=int, default=40)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    """CLI 主流程。"""
    args = parse_args()

    cfg = EngineConfig(
        use_neo4j=args.use_neo4j,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database = args.neo4j_database,
        sample_id=args.sample_id,
        llm=LLMConfig(
            enabled=args.enable_llm,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            timeout_seconds=args.llm_timeout,
            temperature=args.llm_temperature,
        ),
    )

    pipeline = SpatialTemporalPipeline(config=cfg)
    summary = pipeline.run(input_json_path=args.input, max_frames=args.max_frames)

    output_dir = Path(args.output_dir) / args.sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.snapshot_output) if args.snapshot_output else output_dir / "semantic_graph_snapshot.json"
    pipeline.export_graph_snapshot(str(out_path))
    report = pipeline.build_acceptance_report(summary)
    report_path = Path(args.report_output) if args.report_output else output_dir / "design_acceptance_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Build finished")
    print(f"Frames: {summary.total_frames}")
    print(f"Processed objects: {summary.total_objects}")
    print(f"New nodes: {summary.total_new_nodes}")
    print(f"Updated nodes: {summary.total_updated_nodes}")
    print(f"New edges: {summary.total_new_edges}")
    print(f"Duplicate edges: {summary.total_duplicate_edges}")
    print(f"Conflict edges: {summary.total_conflict_edges}")
    print(f"Owner assigned: {summary.total_owner_assigned}")
    print(f"Sample ID: {args.sample_id}")
    print(f"Snapshot: {out_path}")
    print(f"Acceptance report: {report_path}")


if __name__ == "__main__":
    main()
