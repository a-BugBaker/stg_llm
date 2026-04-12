from __future__ import annotations

"""LLM 判定模块。

本模块只负责“问 LLM 并解析结构化 JSON 结果”，不直接改图状态。
上层处理器会根据返回结果决定是否更新节点/边。

提供四类判定：
1. 本帧去重（decide_frame_dedupe）。
2. 跨帧匹配（decide_node_match）。
3. 边动作判定（decide_edge_action）。
4. attached owner 选择（decide_attached_owner）。
"""

import json
from dataclasses import dataclass
import urllib.request
from typing import Any, Dict, List, Optional

from .candidate_resolver import CandidateResult
from .config import LLMConfig
from .models import EntityNode, EntityType, RelationEdge


@dataclass(slots=True)
class DedupeDecision:
    """本帧去重决策结果。"""
    primary_idx: Optional[int] = None
    merged_idxs: Optional[List[int]] = None
    entity_type: Optional[EntityType] = None


@dataclass(slots=True)
class MatchDecision:
    """跨帧匹配决策结果。"""
    matched_node_id: Optional[int] = None


@dataclass(slots=True)
class EdgeDecision:
    """边动作判定结果：duplicate/conflict/new。"""
    action: str = "new"  # duplicate | conflict | new


@dataclass(slots=True)
class OwnerDecision:
    """attached owner 选择结果。"""
    owner_node_id: Optional[int] = None


class LLMDecider:
    """OpenAI-compatible decision helper.

    If disabled or request fails, methods return None and caller should fallback.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    def available(self) -> bool:
        """判断当前是否具备可用的 LLM 调用条件。"""
        return self.config.enabled and bool(self.config.api_key)

    def decide_frame_dedupe(
        self,
        candidates: CandidateResult,
        objects_by_idx: Dict[int, dict],
        fallback_type: EntityType,
    ) -> Optional[DedupeDecision]:
        """对本帧候选对象做去重判定。

返回 primary_idx、merged_idxs、entity_type。
调用失败时返回 None，让上层自动回退规则。
"""
        if not self.available():
            return None

        cmp_payload = [
            {
                "idx": item.idx,
                # "iou": item.iou,
                # "distance": item.distance,
                "score": float(objects_by_idx.get(item.idx, {}).get("score", 0.0)),
                "label": str(objects_by_idx.get(item.idx, {}).get("label", "")),
                "attributes": str(objects_by_idx.get(item.idx, {}).get("attributes", "")),
            }
            for item in candidates.cur_cmp
        ]

        context_payload = [
            {
                "idx": item.idx,
                "iou": item.iou,
                "distance": item.distance,
                "score": float(objects_by_idx.get(item.idx, {}).get("score", 0.0)),
                "label": str(objects_by_idx.get(item.idx, {}).get("label", "")),
            }
            for item in candidates.cur_context
        ]

        prompt = (
            "Task background:\n"
            "- This is a spatial-temporal graph construction pipeline.\n"
            "- Current task is ONLY intra-frame deduplication for one object cluster in ONE frame.\n"
            "- You must not use any id outside the provided cmp_objects list.\n\n"
            "Input fields meaning:\n"
            "1) cmp_objects: objects that are considered duplicates of the same real entity in current frame.\n"
            "   - idx: object index in current frame (the only valid id space for primary_idx/merged_idxs).\n"
            "   - iou, distance, score, label, attributes: evidence for selecting primary object.\n"
            "2) cur_context: nearby objects in current frame for disambiguation, NOT merge targets unless they are also in cmp_objects.\n"
            "3) fallback_entity_type: rule-based fallback type, one of dynamic|static|attached.\n\n"
            "Your task:\n"
            "- Select exactly one primary_idx from cmp_objects.idx.\n"
            "- Select merged_idxs as subset of cmp_objects.idx (can include all duplicates).\n"
            "- Decide entity_type from dynamic|static|attached.\n"
            "- If uncertain, keep conservative and follow fallback_entity_type.\n\n"
            "Hard constraints (MUST):\n"
            "- primary_idx MUST be one of cmp_objects.idx.\n"
            "- merged_idxs MUST be array of unique integers, and every item MUST be in cmp_objects.idx.\n"
            "- merged_idxs MUST include primary_idx.\n"
            "- Do not output explanations or extra keys.\n\n"
            "Output format (strict JSON only):\n"
            "{\"primary_idx\": int, \"merged_idxs\": int[], \"entity_type\": \"dynamic\"|\"static\"}\n\n"
            f"cmp_objects={json.dumps(cmp_payload, ensure_ascii=False)}\n"
            f"cur_context={json.dumps(context_payload, ensure_ascii=False)}\n"
            f"fallback_entity_type={fallback_type.value}"
        )

        data = self._chat_json(prompt)
        if data is None:
            return None

        try:
            entity_type = EntityType(str(data.get("entity_type", fallback_type.value)))
        except Exception:
            entity_type = fallback_type

        merged_idxs = data.get("merged_idxs")
        if not isinstance(merged_idxs, list):
            merged_idxs = None

        primary_idx = data.get("primary_idx")
        if isinstance(primary_idx, bool):
            primary_idx = None
        if primary_idx is not None:
            try:
                primary_idx = int(primary_idx)
            except Exception:
                primary_idx = None

        return DedupeDecision(primary_idx=primary_idx, merged_idxs=merged_idxs, entity_type=entity_type)

    def decide_node_match(
        self,
        primary_obj: dict,
        candidates: CandidateResult,
        graph_nodes: Dict[int, EntityNode],
    ) -> Optional[MatchDecision]:
        """在历史候选节点中选择匹配节点。"""
        if not self.available() or not candidates.pre:
            return None

        pre_payload: List[dict] = []
        for item in candidates.pre:
            node = graph_nodes.get(item.node_id)
            if node is None:
                continue
            pre_payload.append(
                {
                    "node_id": item.node_id,
                    "iou": item.iou,
                    "distance": item.distance,
                    "node_label": node.latest_label(),
                    "node_type": node.entity_type.value,
                }
            )

        prompt = (
            "Task background:\n"
            "- This is cross-frame entity linking in a spatial-temporal graph.\n"
            "- Current task is ONLY selecting a match from the provided candidate_nodes.\n\n"
            "Input fields meaning:\n"
            "1) current_object: detection in current frame.\n"
            "2) candidate_nodes: possible historical nodes from previous frames.\n"
            "   - node_id is the only valid id space for matched_node_id.\n"
            "   - iou/distance/node_label/node_type are matching evidence.\n\n"
            "Your task:\n"
            "- Choose one matched_node_id from candidate_nodes.node_id if it is the same real-world entity.\n"
            "- Return null when no reliable match exists.\n\n"
            "Hard constraints (MUST):\n"
            "- matched_node_id must be null or one of candidate_nodes.node_id.\n"
            "- Do not output explanations or extra keys.\n\n"
            "Output format (strict JSON only):\n"
            "{\"matched_node_id\": int|null}\n\n"
            f"current_object={json.dumps(primary_obj, ensure_ascii=False)}\n"
            f"candidate_nodes={json.dumps(pre_payload, ensure_ascii=False)}"
        )

        data = self._chat_json(prompt)
        if data is None:
            return None

        val = data.get("matched_node_id")
        if val is None:
            return MatchDecision(matched_node_id=None)
        try:
            return MatchDecision(matched_node_id=int(val))
        except Exception:
            return MatchDecision(matched_node_id=None)

    def decide_edge_action(self, new_describe: str, active_edges: List[RelationEdge]) -> Optional[EdgeDecision]:
        """判定新关系与历史关系的语义关系。

返回动作：
1. duplicate: 与已有边重复，丢弃。
2. conflict: 与已有边冲突，旧边失效。
3. new: 直接新增。
"""
        if not self.available():
            return None

        old_payload = [
            {
                "edge_id": edge.id,
                "describe": edge.describe,
                "valid_at": edge.valid_at,
                "invalid_at": edge.invalid_at,
            }
            for edge in active_edges
        ]

        prompt = (
            "Task background:\n"
            "- Relation edges are treated as semantic undirected edges at decision level.\n"
            "- Current task compares one new relation text against active relations for the same endpoint pair.\n\n"
            "Input fields meaning:\n"
            "1) new_relation.describe: new relation description to insert.\n"
            "2) existing_relations: active relation descriptions for the same endpoint pair.\n"
            "   - If meaning is the same as any existing relation, action should be duplicate.\n"
            "   - If meaning contradicts an existing relation, action should be conflict.\n"
            "   - Otherwise action should be new.\n\n"
            "Hard constraints (MUST):\n"
            "- action must be exactly one of: duplicate, conflict, new.\n"
            "- Do not output explanations or extra keys.\n\n"
            "Output format (strict JSON only):\n"
            "{\"action\": \"duplicate\"|\"conflict\"|\"new\"}\n\n"
            f"new_relation={json.dumps({'describe': new_describe}, ensure_ascii=False)}\n"
            f"existing_relations={json.dumps(old_payload, ensure_ascii=False)}"
        )

        data = self._chat_json(prompt)
        if data is None:
            return None

        action = str(data.get("action", "new")).strip().lower()
        if action not in {"duplicate", "conflict", "new"}:
            action = "new"
        return EdgeDecision(action=action)

    def decide_attached_owner(
        self,
        attached_node: EntityNode,
        relation_predicate: str,
        owner_candidates: List[EntityNode],
    ) -> Optional[OwnerDecision]:
        """为 attached 节点选择 owner。"""
        if not self.available() or not owner_candidates:
            return None

        candidates_payload = [
            {
                "node_id": node.id,
                "label": node.latest_label(),
                "type": node.entity_type.value,
            }
            for node in owner_candidates
        ]

        prompt = (
            "Task background:\n"
            "- Attached object owner selection in a spatial-temporal graph.\n"
            "- Current task can only choose from provided owner_candidates.\n\n"
            "Input fields meaning:\n"
            "1) attached_node: object needing an owner.\n"
            "2) predicate: relation hint between attached object and owner.\n"
            "3) owner_candidates: allowed owner nodes; node_id is valid id space.\n\n"
            "Your task:\n"
            "- Select one owner_node_id from owner_candidates when evidence is sufficient.\n"
            "- Return null when uncertain.\n\n"
            "Hard constraints (MUST):\n"
            "- owner_node_id must be null or one of owner_candidates.node_id.\n"
            "- Do not output explanations or extra keys.\n\n"
            "Output format (strict JSON only):\n"
            "{\"owner_node_id\": int|null}\n\n"
            f"attached_node={{\"node_id\": {attached_node.id}, \"label\": {json.dumps(attached_node.latest_label(), ensure_ascii=False)}}}\n"
            f"predicate={json.dumps(relation_predicate, ensure_ascii=False)}\n"
            f"owner_candidates={json.dumps(candidates_payload, ensure_ascii=False)}"
        )

        data = self._chat_json(prompt)
        if data is None:
            return None

        val = data.get("owner_node_id")
        if val is None:
            return OwnerDecision(owner_node_id=None)
        try:
            return OwnerDecision(owner_node_id=int(val))
        except Exception:
            return OwnerDecision(owner_node_id=None)

    def _chat_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """统一的 Chat Completions JSON 调用入口。

约束：
1. 强制 response_format=json_object。
2. 任意异常返回 None，不在此层抛错。
"""
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Output must be strict JSON only."},
                {"role": "user", "content": prompt},
            ],
        }

        req = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
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
