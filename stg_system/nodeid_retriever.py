from __future__ import annotations

"""基于 node_id 的简易自然语言检索模块。

设计目标：
1. 建立 label -> node_id 的反向索引。
2. 从问题中抽取 label 关键词。
3. 返回命中节点的全部关联边与一跳邻域。
"""

from dataclasses import dataclass
import re
from typing import Dict, List, Set

from .models import GraphState


@dataclass(slots=True)
class NodeIdKeywordRetriever:
    """围绕 node_id 的关键词检索器。"""

    graph: GraphState

    def __post_init__(self) -> None:
        self.label_to_node_ids = self._build_label_index()

    def _normalize_label(self, text: str) -> str:
        cleaned = text.strip().lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\d+$", "", cleaned).strip()
        return cleaned

    def _build_label_index(self) -> Dict[str, Set[int]]:
        index: Dict[str, Set[int]] = {}
        for node in self.graph.nodes.values():
            # 使用标签历史，确保所有出现过的 label 都可检索。
            for label, _frame in node.label:
                norm = self._normalize_label(str(label))
                if not norm:
                    continue
                index.setdefault(norm, set()).add(node.id)
        return index

    def _extract_labels(self, question: str) -> List[str]:
        q = self._normalize_label(question)
        matched: List[str] = []
        for label in self.label_to_node_ids.keys():
            if not label:
                continue
            # 单词边界优先，无法匹配时再退化为子串。
            pattern = r"\b" + re.escape(label) + r"\b"
            if re.search(pattern, q) or label in q:
                matched.append(label)
        matched.sort()
        return matched

    def _node_payload(self, node_id: int) -> dict:
        node = self.graph.nodes[node_id]
        return {
            "node_id": node.id,
            "entity_type": node.entity_type.value,
            "latest_label": node.latest_label(),
            "labels": [label for label, _ in node.label],
            "latest_attributes": node.attributes[-1][0] if node.attributes else "",
            "owner": node.owner,
            "state": node.state.value,
        }

    def _edge_payload(self, edge) -> dict:
        return {
            "edge_id": edge.id,
            "from_node_id": edge.from_node_id,
            "to_node_id": edge.to_node_id,
            "describe": edge.describe,
            "predicate": edge.predicate,
            "edge_type": edge.edge_type.value,
            "is_attached": edge.is_attached,
            "valid_at": edge.valid_at,
            "invalid_at": edge.invalid_at,
        }

    def search(self, question: str) -> dict:
        """基于问题中的 label 关键词执行 node_id 检索。"""
        matched_labels = self._extract_labels(question)

        target_node_ids: Set[int] = set()
        for label in matched_labels:
            target_node_ids.update(self.label_to_node_ids.get(label, set()))

        incident_edges = []
        seen_edge_ids: Set[int] = set()
        neighbor_node_ids: Set[int] = set()

        for edge in self.graph.edges.values():
            if edge.from_node_id in target_node_ids or edge.to_node_id in target_node_ids:
                if edge.id in seen_edge_ids:
                    continue
                seen_edge_ids.add(edge.id)
                incident_edges.append(self._edge_payload(edge))

                if edge.from_node_id not in target_node_ids:
                    neighbor_node_ids.add(edge.from_node_id)
                if edge.to_node_id not in target_node_ids:
                    neighbor_node_ids.add(edge.to_node_id)

        target_nodes = [self._node_payload(nid) for nid in sorted(target_node_ids)]
        neighbors = [self._node_payload(nid) for nid in sorted(neighbor_node_ids) if nid in self.graph.nodes]

        return {
            "question": question,
            "matched_labels": matched_labels,
            "matched_node_ids": sorted(target_node_ids),
            "target_nodes": target_nodes,
            "incident_edges": incident_edges,
            "neighbors": neighbors,
            "label_index_size": len(self.label_to_node_ids),
        }
