from __future__ import annotations

"""检索生成问答 CLI。"""

import argparse
import json
import os
from pathlib import Path

from .config import LLMConfig
from .qa_pipeline import GraphQAPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph QA retrieval-generation pipeline")
    parser.add_argument("--snapshot", required=True, help="Path to semantic_graph_snapshot.json")
    parser.add_argument("--question", required=True, help="Natural language question")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--enable-llm", action="store_true", help="Enable LLM generation")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--llm-timeout", type=int, default=400)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    llm_cfg = LLMConfig(
        enabled=args.enable_llm,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        model=args.llm_model,
        timeout_seconds=args.llm_timeout,
        temperature=args.llm_temperature,
    )

    qa = GraphQAPipeline(snapshot_path=args.snapshot, llm_config=llm_cfg)
    result = qa.run(args.question)

    out_path = Path(args.output) if args.output else Path(args.snapshot).parent / "qa_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("QA finished")
    print(f"Snapshot: {args.snapshot}")
    print(f"Question: {args.question}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
