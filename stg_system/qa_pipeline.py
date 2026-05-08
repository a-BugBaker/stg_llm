from __future__ import annotations

"""专用检索-生成问答流水线。

输入：
1. 已构建的语义图快照 JSON（semantic_graph_snapshot.json）。
2. 自然语言问题。

流程：
1. 基于 label->node_id 索引做关键词检索。
2. 取命中节点、关联边、一跳邻域作为上下文。
3. 使用 LLM 在受限证据下生成回答。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
import re
import urllib.request
from typing import Any, Dict, List, Set

from .config import LLMConfig


@dataclass(slots=True)
class GraphQAPipeline:
    """读取快照并执行检索生成问答。"""

    snapshot_path: str
    llm_config: LLMConfig
    snapshot: Dict[str, Any] = field(init=False)
    nodes: List[dict] = field(init=False)
    edges: List[dict] = field(init=False)
    sample_id: str = field(init=False)
    label_index: Dict[str, Set[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.snapshot = json.loads(Path(self.snapshot_path).read_text(encoding="utf-8"))
        self.nodes: List[dict] = list(self.snapshot.get("nodes", []))
        self.edges: List[dict] = list(self.snapshot.get("edges", []))
        self.sample_id = str(self.snapshot.get("sample_id", "unknown"))
        self.label_index = self._build_label_index(self.nodes)

    def _available(self) -> bool:
        return self.llm_config.enabled and bool(self.llm_config.api_key)

    def _normalize(self, text: str) -> str:
        cleaned = text.strip().lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\d+$", "", cleaned).strip()
        return cleaned

    def _build_label_index(self, nodes: List[dict]) -> Dict[str, Set[int]]:
        index: Dict[str, Set[int]] = {}
        for node in nodes:
            node_id = int(node.get("id", -1))
            if node_id < 0:
                continue

            # 历史标签
            for item in node.get("label", []) or []:
                if isinstance(item, list) and item:
                    label = self._normalize(str(item[0]))
                    if label:
                        index.setdefault(label, set()).add(node_id)

            # latest label 冗余补齐
            latest = ""
            labels = node.get("label", []) or []
            if labels and isinstance(labels[-1], list) and labels[-1]:
                latest = str(labels[-1][0])
            latest_norm = self._normalize(latest)
            if latest_norm:
                index.setdefault(latest_norm, set()).add(node_id)

        return index

    def _extract_labels(self, question: str) -> List[str]:
        q = self._normalize(question)
        matched: List[str] = []
        for label in self.label_index.keys():
            if not label:
                continue
            pattern = r"\b" + re.escape(label) + r"\b"
            if re.search(pattern, q) or label in q:
                matched.append(label)
        matched.sort()
        return matched

    def _node_by_id(self) -> Dict[int, dict]:
        mapping: Dict[int, dict] = {}
        for n in self.nodes:
            nid = int(n.get("id", -1))
            if nid >= 0:
                mapping[nid] = n
        return mapping

    def retrieve(self, question: str) -> Dict[str, Any]:
        matched_labels = self._extract_labels(question)
        target_node_ids: Set[int] = set()
        for label in matched_labels:
            target_node_ids.update(self.label_index.get(label, set()))

        node_map = self._node_by_id()

        incident_edges: List[dict] = []
        neighbor_ids: Set[int] = set()
        for edge in self.edges:
            from_id = int(edge.get("from_node_id", -1))
            to_id = int(edge.get("to_node_id", -1))
            if from_id in target_node_ids or to_id in target_node_ids:
                incident_edges.append(edge)
                if from_id not in target_node_ids and from_id >= 0:
                    neighbor_ids.add(from_id)
                if to_id not in target_node_ids and to_id >= 0:
                    neighbor_ids.add(to_id)

        target_nodes = [node_map[nid] for nid in sorted(target_node_ids) if nid in node_map]
        neighbors = [node_map[nid] for nid in sorted(neighbor_ids) if nid in node_map]

        return {
            "sample_id": self.sample_id,
            "question": question,
            "matched_labels": matched_labels,
            "matched_node_ids": sorted(target_node_ids),
            "target_nodes": target_nodes,
            "incident_edges": incident_edges,
            "neighbors": neighbors,
            "stats": {
                "label_index_size": len(self.label_index),
                "target_node_count": len(target_nodes),
                "incident_edge_count": len(incident_edges),
                "neighbor_count": len(neighbors),
            },
        }

    def _chat_json(self, prompt: str) -> Dict[str, Any] | None:
        url = self.llm_config.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.llm_config.model,
            "temperature": self.llm_config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a graph QA assistant. "
                        "Use only provided evidence. "
                        "Output strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }

        req = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_config.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.llm_config.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

        try:
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
                content = "".join(text_parts)
            return json.loads(content)
        except Exception:
            return None

    def generate(self, question: str, retrieval: Dict[str, Any]) -> Dict[str, Any]:
        if not self._available():
            return {
                "answer": "LLM 未启用，返回检索结果。",
                "insufficient": True,
                "evidence": {
                    "node_ids": retrieval.get("matched_node_ids", []),
                    "edge_ids": [int(e.get("id", -1)) for e in retrieval.get("incident_edges", []) if int(e.get("id", -1)) >= 0],
                },
                "notes": "Enable --enable-llm and valid api key for generation.",
            }

        prompt = (
            "# 背景：原始图如何构建\n"
            "你在一个时空语义图系统中回答问题。图的构建过程如下：\n"
            "1) Node Process：逐帧对检测对象做去重与跨帧匹配，得到统一 node_id。\n"
            "2) Edge Process：根据 relations 与 layer_mapping 构建关系边；边按无向语义去重，存在 duplicate/conflict/new 判定。\n"
            "3) Attached/Owner：layer_mapping 构建 attached 边，owner 以规则绑定到对应节点。\n"
            "4) 每条边包含 from_node_id/to_node_id/describe/predicate/type/valid_at。\n\n"
            "# 背景：上下文如何检索\n"
            "当前检索采用关键词方案：\n"
            "1) 从问题中抽取 label 关键词。\n"
            "2) 用 label->node_id 索引定位目标节点。\n"
            "3) 返回目标节点全部关联边（incident_edges）与一跳邻域（neighbors）。\n"
            "4) 你必须严格基于这些检索证据回答，不得编造图外事实。\n\n"
            "# 你的任务\n"
            "请基于 retrieval_context 在整合推理之后回答 question。\n"
            "若证据不足，明确输出 insufficient=true 并说明缺失信息。\n\n"
            "# 输出格式（严格 JSON）\n"
            "{\n"
            "  \"answer\": string,\n"
            "  \"insufficient\": boolean,\n"
            "  \"evidence\": {\"node_ids\": int[], \"edge_ids\": int[]},\n"
            "  \"notes\": string\n"
            "}\n\n"
            f"question={json.dumps(question, ensure_ascii=False)}\n"
            f"retrieval_context={json.dumps(retrieval, ensure_ascii=False)}"
        )

        data = self._chat_json(prompt)
        if data is None:
            return {
                "answer": "LLM 调用失败，返回检索结果。",
                "insufficient": True,
                "evidence": {
                    "node_ids": retrieval.get("matched_node_ids", []),
                    "edge_ids": [int(e.get("id", -1)) for e in retrieval.get("incident_edges", []) if int(e.get("id", -1)) >= 0],
                },
                "notes": "LLM request failed.",
            }

        return {
            "answer": str(data.get("answer", "")),
            "insufficient": bool(data.get("insufficient", False)),
            "evidence": {
                "node_ids": [int(x) for x in data.get("evidence", {}).get("node_ids", []) if isinstance(x, (int, float, str)) and str(x).strip().isdigit()],
                "edge_ids": [int(x) for x in data.get("evidence", {}).get("edge_ids", []) if isinstance(x, (int, float, str)) and str(x).strip().isdigit()],
            },
            "notes": str(data.get("notes", "")),
        }

    def run(self, question: str) -> Dict[str, Any]:
        retrieval = self.retrieve(question)
        generation = self.generate(question, retrieval)
        return {
            "sample_id": self.sample_id,
            "question": question,
            "retrieval": retrieval,
            "generation": generation,
        }
