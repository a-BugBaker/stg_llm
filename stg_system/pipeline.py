from __future__ import annotations

"""离线流水线编排模块。

该模块负责把各子模块连接成端到端流程：
1. 读取输入 JSON。
2. 逐帧调用 FrameProcessor。
3. 可选写入 Neo4j。
4. 输出图快照与验收报告。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .config import EngineConfig
from .models import GraphState
from .node_processor import FrameProcessResult, FrameProcessor
from .storage import Neo4jConfig, Neo4jStorage

# 进度条
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass(slots=True)
class PipelineSummary:
    """全流程统计汇总。"""
    total_frames: int = 0
    total_objects: int = 0
    total_new_nodes: int = 0
    total_updated_nodes: int = 0
    total_new_edges: int = 0
    total_duplicate_edges: int = 0
    total_conflict_edges: int = 0
    total_owner_assigned: int = 0


class SpatialTemporalPipeline:
    """时空图离线主流水线。"""
    def __init__(self, config: EngineConfig):
        self.config = config
        self.processor = FrameProcessor(config=config)
        self.storage: Neo4jStorage | None = None

        if config.use_neo4j:
            self.storage = Neo4jStorage(
                Neo4jConfig(
                    uri=config.neo4j_uri,
                    user=config.neo4j_user,
                    password=config.neo4j_password,
                    database=config.neo4j_database,
                    sample_id=config.sample_id,
                )
            )

    @property
    def graph(self) -> GraphState:
        return self.processor.graph

    def run(self, input_json_path: str, max_frames: int | None = None) -> PipelineSummary:
        """执行完整离线处理。

        返回 PipelineSummary，供 CLI 与报告模块使用。
        """
        frames = self._load_json(input_json_path)
        if not isinstance(frames, list):
            raise ValueError("Input JSON should be a list[frame] or {'frames': list[frame]}")

        if max_frames is not None:
            frames = frames[:max_frames]

        if self.storage is not None:
            self.storage.connect()
            self.storage.ensure_schema()

        summary = PipelineSummary(total_frames=len(frames))

        progress_iter = tqdm(frames, desc="Processing frames", unit="frame") if tqdm else frames
        for frame_id, frame in enumerate(progress_iter):
            objects = frame.get("objects", [])
            if not isinstance(objects, list):
                continue

            frame_result = self.processor.process_frame(objects=objects, frame_id=frame_id)
            self._merge_summary(summary, frame_result)

        if self.storage is not None:
            self.storage.upsert_nodes(self.graph.nodes.values())
            self.storage.upsert_edges(self.graph.edges.values())
            self.storage.close()

        return summary

    def export_graph_snapshot(self, output_path: str) -> None:
        """导出完整图快照（含事件日志）。"""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sample_id": self.config.sample_id,
            "nodes": [self._node_to_dict(n) for n in self.graph.nodes.values()],
            "edges": [self._edge_to_dict(e) for e in self.graph.edges.values()],
            "frame_idx_map": self.graph.frame_idx_map,
            "changed_nodes": self.graph.changed_nodes,
            "edge_action_events": self.graph.edge_action_events,
            "owner_decision_events": self.graph.owner_decision_events,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def build_acceptance_report(self, summary: PipelineSummary) -> Dict[str, Any]:
        """基于当前图状态构建验收报告。"""
        from .evaluation import build_design_acceptance_report

        return build_design_acceptance_report(self.graph, summary, self.config)

    def _merge_summary(self, summary: PipelineSummary, result: FrameProcessResult) -> None:
        """合并单帧统计到总统计。"""
        summary.total_objects += result.processed_objects
        summary.total_new_nodes += result.new_nodes
        summary.total_updated_nodes += result.updated_nodes
        summary.total_new_edges += result.new_edges
        summary.total_duplicate_edges += result.duplicate_edges
        summary.total_conflict_edges += result.conflict_edges
        summary.total_owner_assigned += result.owner_assigned

    def _load_json(self, input_path: str) -> Any:
        """读取输入 JSON 文件。"""
        return json.loads(Path(input_path).read_text(encoding="utf-8"))

    @staticmethod
    def _node_to_dict(node) -> Dict[str, Any]:
        return {
            "id": node.id,
            "entity_type": node.entity_type.value,
            "label": node.label,
            "attributes": node.attributes,
            "candidate": {
                "label": node.candidate.label,
                "attribute": node.candidate.attribute,
                "type": node.candidate.type,
            },
            "last_matched": node.last_matched,
            "first_frame": node.first_frame,
            "missed_frame": node.missed_frame,
            "disappeared_frame": node.disappeared_frame,
            "state": node.state.value,
            "trajectory": node.trajectory,
            "life_value": node.life_value,
            "owner": node.owner,
            "owner_candidates": node.owner_candidates,
        }

    @staticmethod
    def _edge_to_dict(edge) -> Dict[str, Any]:
        return {
            "id": edge.id,
            "from_node_id": edge.from_node_id,
            "to_node_id": edge.to_node_id,
            "describe": edge.describe,
            "predicate": edge.predicate,
            "source_label": edge.source_label,
            "target_label": edge.target_label,
            "type": edge.edge_type.value,
            "is_attached": edge.is_attached,
            "valid_at": edge.valid_at,
            "invalid_at": edge.invalid_at,
        }
