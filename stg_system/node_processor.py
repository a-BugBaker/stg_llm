from __future__ import annotations

"""逐帧核心处理器。

这是系统最核心的业务模块，负责把“单帧对象列表”更新到语义图中。

主流程分三段：
1. 节点处理（node_process）：去重、匹配、新建、反思更新。
2. 生命周期更新：动态节点 active/inactive/disappeared 演化。
3. 边处理（edge_process）：重复/冲突/新增、owner 解析与事件记录。

LLM 与规则关系：
1. 若 LLM 可用，优先采用 LLM 判定。
2. 若 LLM 不可用或返回无效，自动回退规则。
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Set, Tuple

from .candidate_resolver import CandidateResult, get_candidates
from .config import EngineConfig
from .geometry import center, iou
from .llm_decider import LLMDecider
from .models import (
    DynamicState,
    EdgeType,
    EntityNode,
    EntityType,
    GraphState,
    IdGenerator,
    RelationEdge,
)


@dataclass(slots=True)
class FrameProcessResult:
    """单帧处理统计结果。"""
    frame_id: int
    processed_objects: int = 0
    new_nodes: int = 0
    updated_nodes: int = 0
    new_edges: int = 0
    duplicate_edges: int = 0
    conflict_edges: int = 0
    owner_assigned: int = 0


@dataclass(slots=True)
class FrameProcessor:
    config: EngineConfig
    graph: GraphState = field(default_factory=GraphState)
    ids: IdGenerator = field(default_factory=IdGenerator)
    llm_decider: Optional[LLMDecider] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """初始化阶段绑定 LLM 判定器。"""
        self.llm_decider = LLMDecider(self.config.llm)

    def process_frame(self, objects: List[dict], frame_id: int) -> FrameProcessResult:
        """处理单帧数据。

步骤：
1. 遍历对象，跳过已被合并处理的 idx。
2. 调用 get_candidates + _node_resolve 处理节点。
3. 刷新 frame_idx_map，并执行动态生命周期更新。
4. 执行边处理与 owner 解析。
"""
        result = FrameProcessResult(frame_id=frame_id)
        processed_idx: Set[int] = set()
        frame_map: Dict[int, int] = {}  # idx: node id

        for obj in objects:
            idx = int(obj.get("idx", -1))
            if idx < 0 or idx in processed_idx:
                continue

            candidates = get_candidates(idx, objects, frame_id, self.graph, self.config)
            node_id, merged_idxs = self._node_resolve(candidates, objects, idx, frame_id)

            for merged_idx in merged_idxs:
                processed_idx.add(merged_idx)
                frame_map[merged_idx] = node_id

            result.processed_objects += len(merged_idxs)
            if self.graph.nodes[node_id].first_frame == frame_id and len(self.graph.nodes[node_id].label) == 1:
                result.new_nodes += 1
            else:
                result.updated_nodes += 1

        self.graph.frame_idx_map[frame_id] = frame_map
        self._update_dynamic_lifecycle(frame_id)
        # self._rewrite_edges_for_node_changes(frame_id)
        edge_stats = self._edge_process(objects, frame_id, frame_map)
        result.new_edges = edge_stats["new_edges"]
        result.duplicate_edges = edge_stats["duplicate_edges"]
        result.conflict_edges = edge_stats["conflict_edges"]
        result.owner_assigned = edge_stats["owner_assigned"]
        return result

    def _infer_entity_type(self, obj: dict) -> EntityType:
        """实体类型初判规则。

        1. layer_id > 1 视为 attached。
        2. 命中 moving_labels 视为 dynamic。
        3. 其余视为 static。
        """
        layer_id = int(obj.get("layer_id", 1))
        label = str(obj.get("label", "")).lower()
        if layer_id > 1:
            return EntityType.ATTACHED
        if label in self.config.moving_labels:
            return EntityType.DYNAMIC
        return EntityType.STATIC
    
    def _attached_entity_type(self, obj:dict)->EntityType| None:
        """
附属实体类型判定规则：只看layer_i是否大于1，若大于1则视为attached
        """
        layer_id = int(obj.get("layer_id",1))
        if layer_id > 1:
            return EntityType.ATTACHED

    def _pick_primary_idx(self, candidates: CandidateResult, objects_by_idx: Dict[int, dict]) -> int:
        """规则回退：按 score 选主对象。"""
        best_idx = candidates.cur_cmp[0].idx
        best_score = float(objects_by_idx.get(best_idx, {}).get("score", 0.0))
        for item in candidates.cur_cmp:
            score = float(objects_by_idx.get(item.idx, {}).get("score", 0.0))
            if score > best_score:
                best_idx = item.idx
                best_score = score
        return best_idx

    def _match_existing_node(
        self,
        primary_obj: dict,
        candidates: CandidateResult,
        frame_id: int,
        current_idx: int,
    ) -> int | None:
        """跨帧匹配。

        优先 LLM，失败后回退几何+标签启发式。
        """
        if self.llm_decider is not None:
            decision = self.llm_decider.decide_node_match(primary_obj, candidates, self.graph.nodes)
            self._record_llm_match_decision(
                frame_id=frame_id,
                object_idx=current_idx,
                matched_node_id=decision.matched_node_id if decision is not None else None,
                accepted=decision is not None and decision.matched_node_id in self.graph.nodes,
            )
            if decision is not None and decision.matched_node_id in self.graph.nodes:
                return decision.matched_node_id

        # Fallback rule: geometry + label heuristic.
        if not candidates.pre:
            return None

        current_label = str(primary_obj.get("label", "")).lower()
        current_box = primary_obj.get("box", [0, 0, 0, 0])
        for item in candidates.pre:
            node = self.graph.nodes[item.node_id]
            if node.latest_label().lower() == current_label:
                return node.id

        top = candidates.pre[0]
        if top.iou >= self.config.thresholds.pre_iou_threshold:
            node = self.graph.nodes[top.node_id]
            node_box = node.latest_box()
            if node_box is not None and iou(current_box, node_box) >= self.config.thresholds.pre_iou_threshold:
                return top.node_id

        return None

    def _node_resolve(
        self,
        candidates: CandidateResult,
        objects: List[dict],
        idx: int,
        frame_id: int,
    ) -> Tuple[int, List[int]]:
        """节点解析核心逻辑（对应 design 中 node_resolve）。

        关键点：
        1. 先定主对象（LLM/规则）。
        2. 再做 entity_type 判定（LLM 可覆盖初判）。
        3. 再做历史匹配（LLM/规则）。
        4. 根据匹配结果选择新建或更新节点。
        5. 返回 node_id 与本次被合并处理的 idx 列表。
        """
        objects_by_idx = {int(obj["idx"]): obj for obj in objects}

        # 规则回退测试
        fallback_primary_idx = self._pick_primary_idx(candidates, objects_by_idx)
        primary_idx = fallback_primary_idx
        dedupe_decision = None
        fallback_type = self._infer_entity_type(objects_by_idx[fallback_primary_idx])

        # llm处理：仅在存在多个去重候选时才调用，cur_cmp==1 直接走规则
        if self.llm_decider is not None and len(candidates.cur_cmp) > 1:
            dedupe_decision = self.llm_decider.decide_frame_dedupe(candidates, objects_by_idx, fallback_type)
            if dedupe_decision is not None:
                candidate_idxs = [item.idx for item in candidates.cur_cmp]
                accepted = dedupe_decision.primary_idx in candidate_idxs
                self._record_llm_dedupe_decision(
                    frame_id=frame_id,
                    target_idx=idx,
                    candidate_idxs=candidate_idxs,
                    primary_idx=dedupe_decision.primary_idx,
                    merged_idxs=dedupe_decision.merged_idxs,
                    accepted=accepted,
                )
                if accepted:
                    primary_idx = int(dedupe_decision.primary_idx)
        primary_obj = objects_by_idx[primary_idx]
        entity_type = self._infer_entity_type(primary_obj) # 这是防止llm无返回
        if dedupe_decision is not None and dedupe_decision.entity_type is not None:
            entity_type = dedupe_decision.entity_type
        entity_type = self._attached_entity_type(primary_obj) or entity_type # 附属实体类型判定覆盖

        # 跨帧匹配
        matched_node_id = self._match_existing_node(primary_obj, candidates, frame_id, primary_idx)

        merged_idxs = [item.idx for item in candidates.cur_cmp]
        if dedupe_decision is not None and dedupe_decision.merged_idxs:
            cleaned = []
            for m in dedupe_decision.merged_idxs:
                try:
                    val = int(m)
                except Exception:
                    continue
                if val in objects_by_idx:
                    cleaned.append(val)
            if cleaned:
                merged_idxs = sorted(set(cleaned))
        if primary_idx not in merged_idxs:
            merged_idxs.append(primary_idx)

        if matched_node_id is None:
            node = self._create_node(primary_obj, frame_id, entity_type)
            return node.id, merged_idxs

        self._update_node(self.graph.nodes[matched_node_id], primary_obj, frame_id, entity_type)
        return matched_node_id, merged_idxs

    def _create_node(self, obj: dict, frame_id: int, entity_type: EntityType) -> EntityNode:
        """创建新语义节点。"""
        node_id = self.ids.node_id()
        label = str(obj.get("label", "unknown"))
        attr = str(obj.get("attributes", ""))
        box = list(obj.get("box", [0, 0, 0, 0]))

        node = EntityNode(
            id=node_id,
            entity_type=entity_type,
            label=[(label, frame_id)],
            attributes=[(attr, frame_id)],
            last_matched=frame_id,
            first_frame=frame_id,
            life_value=self.config.dynamic.initial_life_value,
        )

        if entity_type == EntityType.DYNAMIC:
            node.trajectory.append((frame_id, box, center(box)))
            node.state = DynamicState.ACTIVE
        else:
            node.position_samples.append(box)

        self.graph.nodes[node_id] = node
        return node

    def _update_node(self, node: EntityNode, obj: dict, frame_id: int, inferred_type: EntityType) -> None:
        """更新已存在节点。

        更新内容：
        1. type/label/attribute 反思流程（候选追加、确认、撤销）。
        2. 动态节点轨迹与 life_value。
        3. 静态/附属节点位置样本。
        """
        label = str(obj.get("label", "unknown"))
        attr = str(obj.get("attributes", ""))
        box = list(obj.get("box", [0, 0, 0, 0]))

        # 反思流程有待细化
        # self._handle_type_reflection(node, inferred_type, frame_id)
        # self._handle_label_reflection(node, label, frame_id)
        # self._handle_attribute_reflection(node, attr, frame_id)

        node.last_matched = frame_id

        if node.entity_type == EntityType.DYNAMIC:
            node.trajectory.append((frame_id, box, center(box)))
            node.state = DynamicState.ACTIVE
            node.life_value = min(node.life_value + 1, self.config.dynamic.max_life_value)
            node.missed_frame = None
            node.disappeared_frame = None
        else:
            node.position_samples.append(box)

    def _has_candidate(self, values: List[Tuple[str, int]], item: str) -> bool:
        for value, _ in values:
            if value == item:
                return True
        return False

    def _remove_candidate_value(self, values: List[Tuple[str, int]], item: str) -> None:
        kept: List[Tuple[str, int]] = []
        for value, frame in values:
            if value != item:
                kept.append((value, frame))
        values[:] = kept

    def _record_label_change(self, node_id: int, frame_id: int, old_label: str, new_label: str) -> None:
        if old_label == new_label:
            return
        changes = self.graph.changed_nodes.setdefault(node_id, [])
        changes.append((frame_id, old_label, new_label))

    def _handle_type_reflection(self, node: EntityNode, inferred_type: EntityType, frame_id: int) -> None:
        """类型反思。

策略：
1. static/attached 优先稳定，只记录候选不直接改。
2. dynamic 在重复证据下允许切换类型。
"""
        if node.entity_type == inferred_type:
            self._remove_candidate_value(node.candidate.type, inferred_type.value)
            return

        # Keep static/attached stable; only log possible type correction.
        if node.entity_type in (EntityType.STATIC, EntityType.ATTACHED):
            if not self._has_candidate(node.candidate.type, inferred_type.value):
                node.candidate.type.append((inferred_type.value, frame_id))
            return

        # Dynamic nodes can switch after repeated evidence.
        if self._has_candidate(node.candidate.type, inferred_type.value):
            node.entity_type = inferred_type
            self._remove_candidate_value(node.candidate.type, inferred_type.value)
            return

        node.candidate.type.append((inferred_type.value, frame_id))

    def _handle_label_reflection(self, node: EntityNode, observed_label: str, frame_id: int) -> None:
        """标签反思。

策略：
1. static/attached 倾向稳定，优先记入 candidate。
2. dynamic 在重复证据下确认写回主标签，并触发变更记录。
"""
        current = node.latest_label()
        if observed_label == current:
            # Evidence supports current label; revoke conflicting label candidates.
            node.candidate.label[:] = [
                (value, f) for value, f in node.candidate.label if value == current
            ]
            return

        # Static and attached entities are stable by default.
        if node.entity_type in (EntityType.STATIC, EntityType.ATTACHED):
            if not self._has_candidate(node.candidate.label, observed_label):
                node.candidate.label.append((observed_label, frame_id))
            return

        # Dynamic entities: confirm change after repeated evidence.
        if self._has_candidate(node.candidate.label, observed_label):
            old_label = current
            node.label.append((observed_label, frame_id))
            self._remove_candidate_value(node.candidate.label, observed_label)
            self._record_label_change(node.id, frame_id, old_label, observed_label)
            return

        node.candidate.label.append((observed_label, frame_id))

    def _handle_attribute_reflection(self, node: EntityNode, observed_attr: str, frame_id: int) -> None:
        """属性反思，规则与标签反思一致。"""
        current_attr = node.attributes[-1][0] if node.attributes else ""
        if observed_attr == current_attr:
            node.candidate.attribute[:] = [
                (value, f) for value, f in node.candidate.attribute if value == current_attr
            ]
            return

        if node.entity_type in (EntityType.STATIC, EntityType.ATTACHED):
            if not self._has_candidate(node.candidate.attribute, observed_attr):
                node.candidate.attribute.append((observed_attr, frame_id))
            return

        if self._has_candidate(node.candidate.attribute, observed_attr):
            node.attributes.append((observed_attr, frame_id))
            self._remove_candidate_value(node.candidate.attribute, observed_attr)
            return

        node.candidate.attribute.append((observed_attr, frame_id))

    # def _rewrite_edges_for_node_changes(self, frame_id: int) -> None:
    #     """节点标签变化后的边描述级联更新。"""
    #     for node_id, changes in self.graph.changed_nodes.items():
    #         for change_frame, _old_label, new_label in changes:
    #             if change_frame != frame_id:
    #                 continue

    #             for edge in self.graph.edges.values():
    #                 if edge.invalid_at is not None:
    #                     continue
    #                 if edge.from_node_id != node_id and edge.to_node_id != node_id:
    #                     continue

    #                 if edge.from_node_id == node_id:
    #                     edge.source_label = self._canonical_label(new_label)
    #                 if edge.to_node_id == node_id:
    #                     edge.target_label = self._canonical_label(new_label)
    #                 edge.describe = self._build_edge_describe(edge.source_label, edge.predicate, edge.target_label)

    def _update_dynamic_lifecycle(self, frame_id: int) -> None:
        """动态节点生命周期推进。

本帧未匹配到的动态节点：
1. life_value -= 1
2. life_value > 0 -> inactive
3. life_value <= 0 -> disappeared
"""
        matched_nodes = set(self.graph.frame_idx_map.get(frame_id, {}).values())
        for node in self.graph.nodes.values():
            if node.entity_type != EntityType.DYNAMIC:
                continue
            if node.id in matched_nodes:
                continue

            node.life_value -= 1
            node.missed_frame = frame_id
            if node.life_value > 0:
                node.state = DynamicState.INACTIVE
            else:
                node.state = DynamicState.DISAPPEARED
                node.disappeared_frame = frame_id

    def _edge_type(self, source_type: EntityType, target_type: EntityType) -> EdgeType:
        """根据端点类型推导边类型。"""
        if source_type == EntityType.ATTACHED or target_type == EntityType.ATTACHED:
            return EdgeType.ATTACHED
        if source_type == EntityType.DYNAMIC and target_type == EntityType.DYNAMIC:
            return EdgeType.DYNAMIC_DYNAMIC
        if source_type == EntityType.STATIC and target_type == EntityType.STATIC:
            return EdgeType.STATIC_STATIC
        return EdgeType.STATIC_DYNAMIC

    def _canonical_label(self, label: str) -> str:
        return re.sub(r"\d+$", "", label).strip() or label

    def _normalize_relation_text(self, text: str) -> str:
        """关系文本规范化。

主要用于去除标签尾号并压缩空白，降低文本抖动。
"""
        cleaned = re.sub(r"\b([A-Za-z_]+)\d+\b", r"\1", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _predicate_indicates_owner(self, predicate: str) -> bool:
        """判断关系文本是否具有 owner 倾向。"""
        p = predicate.lower().strip()
        keywords = ["on", "attached", "wear", "part of", "belongs", "holding", "in", "of"]
        for keyword in keywords:
            if keyword in p:
                return True
        return False

    def _build_edge_describe(self, source_label: str, source_label_frame: str, predicate: str, target_label: str,target_label_frame:str) -> str:
        """先进行边的替换，再进行描述
        
        若图中的标签与帧中的标签不一致，则先将谓词中的帧标签替换为图中标签，再进行后续逻辑。
        """
        src = self._canonical_label(source_label)
        tgt = self._canonical_label(target_label)
        pred = self._normalize_relation_text(predicate)
        src_f = self._canonical_label(source_label_frame)
        tgt_f = self._canonical_label(target_label_frame)
        
        # 若源标签发生了变化（节点合并或更新），则在谓词中进行替换
        if src != src_f:
            pred = pred.replace(src_f, src)
        
        # 若目标标签发生了变化，则在谓词中进行替换
        if tgt != tgt_f:
            pred = pred.replace(tgt_f, tgt)
        
        pred_lower = pred.lower()
        if src.lower() in pred_lower and tgt.lower() in pred_lower:
            return pred
        return f"{src} {pred} {tgt}".strip()

    def _build_attached_edge_describe(self, owner_label: str, attached_label: str) -> str:
        """构造 node-attached 静态边的标准描述。"""
        return f"{self._canonical_label(owner_label)} has attached {self._canonical_label(attached_label)}"

    def _record_edge_action(
        self,
        frame_id: int,
        action: str,
        from_node_id: int,
        to_node_id: int,
        describe: str,
        edge_id: int | None = None,
        reason: str = "",
    ) -> None:
        """记录边动作事件，供验收报告与调试分析使用。"""
        self.graph.edge_action_events.append(
            {
                "frame_id": frame_id,
                "action": action,
                "from_node_id": from_node_id,
                "to_node_id": to_node_id,
                "describe": describe,
                "edge_id": edge_id,
                "reason": reason,
            }
        )

    def _record_owner_decision(
        self,
        frame_id: int,
        attached_node_id: int,
        event_type: str,
        owner_node_id: int | None,
        owner_label: str | None,
        reason: str,
    ) -> None:
        """记录 owner 决策事件。"""
        self.graph.owner_decision_events.append(
            {
                "frame_id": frame_id,
                "attached_node_id": attached_node_id,
                "event_type": event_type,
                "owner_node_id": owner_node_id,
                "owner_label": owner_label,
                "reason": reason,
            }
        )

    def _record_llm_dedupe_decision(
        self,
        frame_id: int,
        target_idx: int,
        candidate_idxs: List[int],
        primary_idx: int | None,
        merged_idxs: List[int] | None,
        accepted: bool,
    ) -> None:
        """记录 LLM 去重判定：候选 idx 与输出 idx。"""
        self.graph.llm_dedupe_events.append(
            {
                "frame_id": frame_id,
                "target_idx": target_idx,
                "candidate_idxs": candidate_idxs,
                "llm_primary_idx": primary_idx,
                "llm_merged_idxs": merged_idxs,
                "accepted": accepted,
            }
        )

    def _record_llm_match_decision(
        self,
        frame_id: int,
        object_idx: int,
        matched_node_id: int | None,
        accepted: bool,
    ) -> None:
        """记录 LLM 跨帧匹配：idx -> node_id。"""
        self.graph.llm_match_events.append(
            {
                "frame_id": frame_id,
                "object_idx": object_idx,
                "matched_node_id": matched_node_id,
                "accepted": accepted,
            }
        )

    def _record_llm_edge_decision(
        self,
        frame_id: int,
        from_node_id: int,
        to_node_id: int,
        describe: str,
        action: str,
    ) -> None:
        """记录 LLM 边动作判定（duplicate/conflict/new）。"""
        self.graph.llm_edge_decision_events.append(
            {
                "frame_id": frame_id,
                "from_node_id": from_node_id,
                "to_node_id": to_node_id,
                "describe": describe,
                "action": action,
            }
        )

    def _edge_process(self, objects: List[dict], frame_id: int, frame_map: Dict[int, int]) -> Dict[str, int]:
        """边处理核心（对应 design 中 edge_process）。

        流程：
        1. 解析关系端点（优先 relation.idx，回退 object_tag）。
        2. 生成结构化描述 describe。
        3. 执行 duplicate/conflict/new 判定（LLM 或规则）。
        4. 写入边并记录动作事件。
        5. 对 attached 节点执行 owner 决策流程。
        """
        if not frame_map:
            return {
                "new_edges": 0,
                "duplicate_edges": 0,
                "conflict_edges": 0,
                "owner_assigned": 0,
            }

        new_edges = 0
        duplicate_edges = 0
        conflict_edges = 0
        owner_assigned = 0
        attached_edge_node_ids: Set[int] = set()
        for existing_edge in self.graph.edges.values():
            if existing_edge.invalid_at is not None:
                continue
            if not existing_edge.is_attached:
                continue
            attached_edge_node_ids.add(existing_edge.from_node_id)
            attached_edge_node_ids.add(existing_edge.to_node_id)
        objects_by_idx: Dict[int, dict] = {}
        for item in objects:
            try:
                cur_idx = int(item.get("idx", -1))
            except Exception:
                continue
            if cur_idx >= 0:
                objects_by_idx[cur_idx] = item
        # tag_to_idx = {
        #     str(obj.get("tag", "")): int(obj.get("idx", -1))
        #     for obj in objects
        #     if str(obj.get("tag", "")).strip()
        # }

        for obj in objects:
            src_idx = int(obj.get("idx", -1))
            if src_idx not in frame_map:
                continue

            src_node_id = frame_map[src_idx]
            src_node = self.graph.nodes[src_node_id]


            # layer_mapping 明确表达 owner-attached 关系：只去重，不冲突，不修改。
            for lay_map in obj.get("layer_mapping", []) or []:
                attached_idx_raw = lay_map.get("idx", -1)
                if isinstance(attached_idx_raw, int):
                    attached_idx = attached_idx_raw
                elif isinstance(attached_idx_raw, str) and attached_idx_raw.strip().isdigit():
                    attached_idx = int(attached_idx_raw.strip())
                else:
                    continue

                if attached_idx not in frame_map:
                    continue

                attached_node_id = frame_map[attached_idx]
                attached_node = self.graph.nodes[attached_node_id]

                owner_label = src_node.latest_label()
                attached_label = attached_node.latest_label()
                attached_describe = self._build_attached_edge_describe(owner_label, attached_label)

                if self._is_duplicate_edge(src_node_id, attached_node_id, attached_describe):
                    duplicate_edges += 1
                    self._record_edge_action(
                        frame_id,
                        "duplicate",
                        src_node_id,
                        attached_node_id,
                        attached_describe,
                        reason="layer_mapping_duplicate",
                    )
                else:
                    edge = RelationEdge(
                        id=self.ids.edge_id(),
                        from_node_id=src_node_id,
                        to_node_id=attached_node_id,
                        describe=attached_describe,
                        predicate="node_attached",
                        source_label=self._canonical_label(owner_label),
                        target_label=self._canonical_label(attached_label),
                        edge_type=EdgeType.ATTACHED,
                        valid_at=frame_id,
                        is_attached=True,
                    )
                    self.graph.edges[edge.id] = edge
                    new_edges += 1
                    self._record_edge_action(
                        frame_id,
                        "new",
                        src_node_id,
                        attached_node_id,
                        attached_describe,
                        edge_id=edge.id,
                        reason="layer_mapping_insert",
                    )

                attached_edge_node_ids.add(src_node_id)
                attached_edge_node_ids.add(attached_node_id)

                # owner 唯一来源：layer_mapping。已有 owner 时不覆盖。
                owner_assigned += self._resolve_attached_owner(
                    attached_node=attached_node,
                    relation_predicate="node_attached",
                    target_node=src_node,
                    frame_id=frame_id,
                )


            # 处理关系
            for rel in obj.get("relations", []) or []:
                tgt_idx = None
                rel_idx = rel.get("idx")
                if isinstance(rel_idx, int):
                    tgt_idx = rel_idx
                elif isinstance(rel_idx, str) and rel_idx.strip().isdigit():
                    tgt_idx = int(rel_idx.strip())

                if tgt_idx is None:
                    raise ValueError(f"Invalid target index: tgt_idx is None")
                if tgt_idx is None or tgt_idx not in frame_map:
                    continue

                tgt_node_id = frame_map[tgt_idx]
                tgt_node = self.graph.nodes[tgt_node_id]

                # 若两个端点都已出现在 attached 边中，则跳过普通 relation 处理。
                if src_node_id in attached_edge_node_ids and tgt_node_id in attached_edge_node_ids:
                    continue

                predicate = str(rel.get("predicate", "")).strip()
                if not predicate:
                    continue

                source_label = src_node.latest_label()
                target_label = tgt_node.latest_label()

                # 将当前帧的谓词中的两个label替换成图中对应的label（针对匹配换词的情况）
                source_label_frame = obj.get("label","")
                target_label_frame = objects_by_idx.get(tgt_idx, {}).get("label", "")

                describe = self._build_edge_describe(source_label,source_label_frame, predicate, target_label,target_label_frame)
                # edge_type = self._edge_type(src_node.entity_type, tgt_node.entity_type)

                active_edges = self._active_edges_same_endpoints(src_node_id, tgt_node_id)
                llm_action = None
                if self.llm_decider is not None and active_edges:
                    edge_decision = self.llm_decider.decide_edge_action(describe, active_edges)
                    if edge_decision is not None:
                        llm_action = edge_decision.action
                        self._record_llm_edge_decision(
                            frame_id=frame_id,
                            from_node_id=src_node_id,
                            to_node_id=tgt_node_id,
                            describe=describe,
                            action=llm_action,
                        )

                if llm_action == "duplicate":
                    duplicate_edges += 1
                    self._record_edge_action(
                        frame_id,
                        "duplicate",
                        src_node_id,
                        tgt_node_id,
                        describe,
                        reason="llm_decision",
                    )
                    continue
                if llm_action == "conflict":
                    self._invalidate_conflicting_edge(src_node_id, tgt_node_id, describe, frame_id)
                    conflict_edges += 1
                    self._record_edge_action(
                        frame_id,
                        "conflict",
                        src_node_id,
                        tgt_node_id,
                        describe,
                        reason="llm_decision",
                    )
                elif llm_action is None:
                    if self._is_duplicate_edge(src_node_id, tgt_node_id, describe):
                        duplicate_edges += 1
                        self._record_edge_action(
                            frame_id,
                            "duplicate",
                            src_node_id,
                            tgt_node_id,
                            describe,
                            reason="rule_same_describe",
                        )
                        continue
                    if self._invalidate_conflicting_edge(src_node_id, tgt_node_id, describe, frame_id):
                        conflict_edges += 1
                        self._record_edge_action(
                            frame_id,
                            "conflict",
                            src_node_id,
                            tgt_node_id,
                            describe,
                            reason="rule_same_endpoints_diff_describe",
                        )

                edge = RelationEdge(
                    id=self.ids.edge_id(),
                    from_node_id=src_node_id,
                    to_node_id=tgt_node_id,
                    describe=describe,
                    predicate=predicate,
                    source_label=self._canonical_label(source_label),
                    target_label=self._canonical_label(target_label),
                    edge_type=self._edge_type(src_node.entity_type, tgt_node.entity_type),
                    valid_at=frame_id,
                    is_attached=False,
                )
                self.graph.edges[edge.id] = edge
                new_edges += 1
                self._record_edge_action(
                    frame_id,
                    "new",
                    src_node_id,
                    tgt_node_id,
                    describe,
                    edge_id=edge.id,
                    reason="insert_new_relation",
                )

                # if src_node.entity_type == EntityType.ATTACHED:
                #     owner_assigned += self._resolve_attached_owner(
                #         attached_node=src_node,
                #         relation_predicate=predicate,
                #         target_node=tgt_node,
                #         frame_id=frame_id,
                #     )


        return {
            "new_edges": new_edges,
            "duplicate_edges": duplicate_edges,
            "conflict_edges": conflict_edges,
            "owner_assigned": owner_assigned,
        }

    def _resolve_attached_owner(
        self,
        attached_node: EntityNode,
        relation_predicate: str,
        target_node: EntityNode,
        frame_id: int,
    ) -> int:
        """attached owner 唯一绑定逻辑。

规则：
1. owner 只从 layer_mapping(node_attached) 确定。
2. 首次绑定后不再修改。
3. 不使用 owner candidate。
"""
        existing_owner_id = attached_node.owner[0] if attached_node.owner is not None else None

        if relation_predicate != "node_attached":
            self._record_owner_decision(
                frame_id,
                attached_node.id,
                "skip",
                None,
                None,
                "owner_only_from_layer_mapping",
            )
            return 0

        if existing_owner_id is None:
            attached_node.owner = (target_node.id, target_node.latest_label())
            self._record_owner_decision(
                frame_id,
                attached_node.id,
                "assign",
                target_node.id,
                target_node.latest_label(),
                "assigned_by_layer_mapping",
            )
            return 1

        if existing_owner_id == target_node.id:
            self._record_owner_decision(
                frame_id,
                attached_node.id,
                "keep",
                target_node.id,
                target_node.latest_label(),
                "owner_unchanged",
            )
            return 0

        self._record_owner_decision(
            frame_id,
            attached_node.id,
            "skip",
            target_node.id,
            target_node.latest_label(),
            "owner_locked_no_override",
        )
        return 0

    def _is_duplicate_edge(self, from_id: int, to_id: int, describe: str) -> bool:
        """检查是否存在同无向端点同描述的有效边。"""
        a, b = sorted((from_id, to_id))
        for edge in self.graph.edges.values():
            if edge.invalid_at is not None:
                continue
            x, y = sorted((edge.from_node_id, edge.to_node_id))
            if x == a and y == b and edge.describe == describe:
                return True
        return False

    def _invalidate_conflicting_edge(self, from_id: int, to_id: int, describe: str, frame_id: int) -> bool:
        """将同无向端点但不同描述的有效边标记失效。"""
        changed = False
        a, b = sorted((from_id, to_id))
        for edge in self.graph.edges.values():
            if edge.invalid_at is not None:
                continue
            x, y = sorted((edge.from_node_id, edge.to_node_id))
            if x == a and y == b and edge.describe != describe:
                edge.invalid_at = frame_id
                changed = True
        return changed

    def _active_edges_same_endpoints(self, from_id: int, to_id: int) -> List[RelationEdge]:
        """获取同无向端点的所有有效边。"""
        result: List[RelationEdge] = []
        a, b = sorted((from_id, to_id))
        for edge in self.graph.edges.values():
            if edge.invalid_at is not None:
                continue
            x, y = sorted((edge.from_node_id, edge.to_node_id))
            if x == a and y == b:
                result.append(edge)
        return result
