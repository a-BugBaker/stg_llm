from __future__ import annotations

"""核心数据模型定义。

本文件定义四层核心对象：
1. 枚举类型：实体类型、动态状态、边类型。
2. 节点模型：EntityNode + CandidatePool。
3. 边模型：RelationEdge（含时态字段）。
4. 全局图状态与 ID 生成器：GraphState/IdGenerator。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EntityType(str, Enum):
    """实体类型。

dynamic: 动态目标
static: 静态背景目标
attached: 附属目标（通常 layer_id > 1）
"""
    DYNAMIC = "dynamic"
    STATIC = "static"
    ATTACHED = "attached"


class DynamicState(str, Enum):
    """动态节点生命周期状态。"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISAPPEARED = "disappeared"


class EdgeType(str, Enum):
    """语义边类型。"""
    STATIC_STATIC = "static-static"
    STATIC_DYNAMIC = "static-dynamic"
    DYNAMIC_DYNAMIC = "dynamic-dynamic"
    ATTACHED = "attached"


@dataclass(slots=True)
class CandidatePool:
    """候选反思池。

当系统（或 LLM）对当前字段不完全确定时，先放入候选池，
后续若证据重复出现再确认写回主字段。
"""
    label: List[Tuple[str, int]] = field(default_factory=list)
    attribute: List[Tuple[str, int]] = field(default_factory=list)
    type: List[Tuple[str, int]] = field(default_factory=list)


@dataclass(slots=True)
class EntityNode:
    """语义层实体节点。

一个节点在生命周期中会持续累积：
1. label/attributes 历史。
2. candidate 反思记录。
3. 动态轨迹或静态位置样本。
4. attached 的 owner 与 owner 候选。
"""
    id: int
    entity_type: EntityType
    label: List[Tuple[str, int]] = field(default_factory=list)
    attributes: List[Tuple[str, int]] = field(default_factory=list)
    candidate: CandidatePool = field(default_factory=CandidatePool)
    last_matched: Optional[int] = None

    # static-like
    first_frame: Optional[int] = None
    position_samples: List[List[float]] = field(default_factory=list)

    # dynamic-like
    missed_frame: Optional[int] = None
    disappeared_frame: Optional[int] = None
    state: DynamicState = DynamicState.ACTIVE
    trajectory: List[Tuple[int, List[float], Tuple[float, float]]] = field(default_factory=list)
    life_value: int = 0

    # attached-like
    owner: Optional[Tuple[int, str]] = None
    owner_candidates: List[Tuple[int, str, int]] = field(default_factory=list)

    def latest_label(self) -> str:
        """返回当前最新标签。"""
        return self.label[-1][0] if self.label else "unknown"

    def latest_box(self) -> Optional[List[float]]:
        """返回用于跨帧匹配的最新几何框。"""
        if self.trajectory:
            return self.trajectory[-1][1]
        if self.position_samples:
            return self.position_samples[-1]
        return None


@dataclass(slots=True)
class RelationEdge:
    """语义层关系边。

同时保存结构化字段和描述文本：
1. predicate/source_label/target_label 用于可解释与重写。
2. describe 作为人类可读关系串。
3. valid_at/invalid_at 用于时态有效期管理。
"""
    id: int
    from_node_id: int
    to_node_id: int
    describe: str
    predicate: str
    source_label: str
    target_label: str
    edge_type: EdgeType
    valid_at: int
    is_attached: bool = False
    invalid_at: Optional[int] = None


@dataclass(slots=True)
class GraphState:
    """运行时图状态容器。

包含：
1. nodes/edges 主图。
2. frame_idx_map: 当前帧 idx 到 node_id 映射。
3. changed_nodes: 节点标签变化日志，驱动边描述级联更新。
4. edge_action_events/owner_decision_events: 可解释事件流。
"""
    nodes: Dict[int, EntityNode] = field(default_factory=dict)
    edges: Dict[int, RelationEdge] = field(default_factory=dict)
    # frame index -> object idx -> node_id
    frame_idx_map: Dict[int, Dict[int, int]] = field(default_factory=dict)
    # record node changes for edge rewrite
    changed_nodes: Dict[int, List[Tuple[int, str, str]]] = field(default_factory=dict)
    # trace edge decisions for explainability
    edge_action_events: List[Dict[str, Any]] = field(default_factory=list)
    # trace attached owner decisions for explainability
    owner_decision_events: List[Dict[str, Any]] = field(default_factory=list)
    # trace llm dedupe decisions
    llm_dedupe_events: List[Dict[str, Any]] = field(default_factory=list)
    # trace llm match decisions
    llm_match_events: List[Dict[str, Any]] = field(default_factory=list)
    # trace llm edge action decisions
    llm_edge_decision_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class IdGenerator:
    """全局自增 ID 生成器。"""
    next_node_id: int = 1
    next_edge_id: int = 1

    def node_id(self) -> int:
        """分配新的 node_id。"""
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

    def edge_id(self) -> int:
        """分配新的 edge_id。"""
        eid = self.next_edge_id
        self.next_edge_id += 1
        return eid
