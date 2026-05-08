from __future__ import annotations

"""运行 OpenEQA pilot 的逐题问答与 LLM 裁判评分。"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from prepare_graph.defaults import (
    DEFAULT_OPENEQA_EVAL_DIR,
    DEFAULT_OPENEQA_MANIFEST_PATH,
    DEFAULT_OPENEQA_QUESTIONS_PATH,
    DEFAULT_OPENEQA_STG_DIR,
)

from .config import LLMConfig
from .openeqa_judge import OpenEQAJudge
from .qa_pipeline import GraphQAPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 OpenEQA pilot 的问答与 LLM 裁判评分。")
    parser.add_argument("--questions", default=str(DEFAULT_OPENEQA_QUESTIONS_PATH))
    parser.add_argument("--manifest", default=str(DEFAULT_OPENEQA_MANIFEST_PATH))
    parser.add_argument("--stg-root", default=str(DEFAULT_OPENEQA_STG_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OPENEQA_EVAL_DIR))
    parser.add_argument("--enable-qa-llm", action="store_true", help="启用图问答阶段的 LLM 生成。")
    parser.add_argument("--enable-judge-llm", action="store_true", help="启用裁判阶段的 LLM 语义评分。")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--judge-model", default=None, help="裁判模型；默认与 --llm-model 相同。")
    parser.add_argument("--llm-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--llm-timeout", type=int, default=400)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).expanduser().resolve().read_text(encoding="utf-8"))
    sample_to_source = {
        str(clip.get("sample_id", "")): str(clip.get("dataset_source", ""))
        for clip in manifest.get("clips", [])
    }

    qa_llm_cfg = LLMConfig(
        enabled=args.enable_qa_llm,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        model=args.llm_model,
        timeout_seconds=args.llm_timeout,
        temperature=args.llm_temperature,
    )
    judge_llm_cfg = LLMConfig(
        enabled=args.enable_judge_llm,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        model=args.judge_model or args.llm_model,
        timeout_seconds=args.llm_timeout,
        temperature=args.llm_temperature,
    )
    judge = OpenEQAJudge(judge_llm_cfg)

    predictions_path = output_dir / "predictions.jsonl"
    judge_path = output_dir / "judge_results.jsonl"
    for path in (predictions_path, judge_path):
        if path.exists():
            path.unlink()

    pipelines: dict[str, GraphQAPipeline] = {}
    prediction_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []

    question_rows = _read_jsonl(Path(args.questions).expanduser().resolve())
    for row in question_rows:
        sample_id = str(row["sample_id"])
        if sample_id not in pipelines:
            snapshot_path = Path(args.stg_root).expanduser().resolve() / sample_id / "semantic_graph_snapshot.json"
            if not snapshot_path.exists():
                raise FileNotFoundError(f"Snapshot not found for sample {sample_id}: {snapshot_path}")
            pipelines[sample_id] = GraphQAPipeline(snapshot_path=str(snapshot_path), llm_config=qa_llm_cfg)

        qa_result = pipelines[sample_id].run(str(row["question"]))
        pred_answer = str(qa_result.get("generation", {}).get("answer", "")).strip()
        prediction_row = {
            "question_id": str(row["question_id"]),
            "sample_id": sample_id,
            "episode_history": str(row["episode_history"]),
            "question": str(row["question"]),
            "gold_answer": str(row["answer"]),
            "extra_answers": list(row.get("extra_answers", []) or []),
            "pred_answer": pred_answer,
            "retrieval": qa_result.get("retrieval", {}),
            "generation": qa_result.get("generation", {}),
        }
        prediction_rows.append(prediction_row)

        judge_result = judge.judge(
            question=str(row["question"]),
            gold_answer=str(row["answer"]),
            extra_answers=list(row.get("extra_answers", []) or []),
            pred_answer=pred_answer,
        )
        judge_rows.append(
            {
                "question_id": str(row["question_id"]),
                "sample_id": sample_id,
                "episode_history": str(row["episode_history"]),
                "dataset_source": sample_to_source.get(sample_id, str(row["episode_history"]).split("/")[0]),
                "category": str(row["category"]),
                "question": str(row["question"]),
                "gold_answer": str(row["answer"]),
                "extra_answers": list(row.get("extra_answers", []) or []),
                "pred_answer": pred_answer,
                "score": int(judge_result["score"]),
                "correct": bool(judge_result["correct"]),
                "reason": str(judge_result["reason"]),
            }
        )

    _write_jsonl(predictions_path, prediction_rows)
    _write_jsonl(judge_path, judge_rows)

    metrics = build_metrics_summary(judge_rows)
    metrics_path = output_dir / "metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OpenEQA evaluation finished")
    print(f"Predictions: {predictions_path}")
    print(f"Judge results: {judge_path}")
    print(f"Metrics: {metrics_path}")


def build_metrics_summary(judge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总总体、类别、数据源与 sample 级别指标。"""
    total = len(judge_rows)
    correct = sum(int(row["score"]) for row in judge_rows)
    by_category = _group_accuracy(judge_rows, key="category")
    by_source = _group_accuracy(judge_rows, key="dataset_source")
    by_sample = _group_accuracy(judge_rows, key="sample_id")

    return {
        "total_questions": total,
        "correct_questions": correct,
        "judge_accuracy": (correct / total) if total else 0.0,
        "by_category": by_category,
        "by_dataset_source": by_source,
        "by_sample_id": by_sample,
    }


def _group_accuracy(rows: list[dict[str, Any]], *, key: str) -> dict[str, dict[str, Any]]:
    grouped_scores: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped_scores[str(row.get(key, "unknown"))].append(int(row["score"]))

    result: dict[str, dict[str, Any]] = {}
    for group_name, scores in sorted(grouped_scores.items()):
        total = len(scores)
        correct = sum(scores)
        result[group_name] = {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total) if total else 0.0,
        }
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


if __name__ == "__main__":
    main()
