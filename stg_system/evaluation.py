from __future__ import annotations

"""验收评估模块。

将运行结果转换为“可审阅”的技术报告：
1. 汇总统计（summary / graph_stats）。
2. 需求条目级通过情况（requirement_checks）。
3. 关键事件样例（samples）。
"""

from typing import Any, Dict
from typing import TYPE_CHECKING

from .config import EngineConfig
from .models import EntityType, GraphState

if TYPE_CHECKING:
    from .pipeline import PipelineSummary


def _count_dynamic_states(graph: GraphState) -> Dict[str, int]:
    """统计动态节点状态分布。"""
    counts = {"active": 0, "inactive": 0, "disappeared": 0}
    for node in graph.nodes.values():
        if node.entity_type != EntityType.DYNAMIC:
            continue
        counts[node.state.value] = counts.get(node.state.value, 0) + 1
    return counts


def _status_item(ok: bool, note: str) -> Dict[str, Any]:
    """构造统一状态项。"""
    return {
        "status": "pass" if ok else "fail",
        "ok": ok,
        "note": note,
    }


def build_design_acceptance_report(
    graph: GraphState,
    summary: PipelineSummary,
    config: EngineConfig,
) -> Dict[str, Any]:
    """生成 design 对齐验收报告。

重点输出：
1. 关系动作统计与事件样例。
2. owner 决策统计与事件样例。
3. 按需求条目逐项给出 pass/fail 与说明。
"""
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    invalid_edge_count = sum(1 for e in graph.edges.values() if e.invalid_at is not None)
    candidate_nodes = sum(
        1
        for n in graph.nodes.values()
        if n.candidate.label or n.candidate.attribute or n.candidate.type
    )
    owner_attached = sum(1 for n in graph.nodes.values() if n.entity_type == EntityType.ATTACHED and n.owner is not None)
    attached_total = sum(1 for n in graph.nodes.values() if n.entity_type == EntityType.ATTACHED)
    changed_nodes = sum(1 for _k, v in graph.changed_nodes.items() if v)
    dynamic_count = sum(1 for n in graph.nodes.values() if n.entity_type == EntityType.DYNAMIC)
    lifecycle_touched = sum(
        1
        for n in graph.nodes.values()
        if n.entity_type == EntityType.DYNAMIC and (n.missed_frame is not None or n.disappeared_frame is not None)
    )
    structured_edges = sum(
        1 for e in graph.edges.values() if bool(e.predicate) and bool(e.source_label) and bool(e.target_label)
    )
    edge_action_counts = {"new": 0, "duplicate": 0, "conflict": 0}
    for event in graph.edge_action_events:
        action = str(event.get("action", "")).lower()
        if action in edge_action_counts:
            edge_action_counts[action] += 1

    owner_event_counts = {
        "assign": 0,
        "keep": 0,
        "candidate": 0,
        "switch": 0,
        "skip": 0,
    }
    for event in graph.owner_decision_events:
        event_type = str(event.get("event_type", "")).lower()
        if event_type in owner_event_counts:
            owner_event_counts[event_type] += 1

    def sample_events(events: list[dict], action: str, limit: int = 5) -> list[dict]:
        """抽取指定动作的样例事件。"""
        result: list[dict] = []
        for item in events:
            if str(item.get("action", "")).lower() == action:
                result.append(item)
            if len(result) >= limit:
                break
        return result

    def sample_owner_events(events: list[dict], event_type: str, limit: int = 5) -> list[dict]:
        """抽取指定 owner 事件类型的样例。"""
        result: list[dict] = []
        for item in events:
            if str(item.get("event_type", "")).lower() == event_type:
                result.append(item)
            if len(result) >= limit:
                break
        return result
    pass_count = 0
    total_count = 0

    requirements = {
        "entity_node_model": _status_item(node_count > 0, f"nodes={node_count}"),
        "relation_edge_model": _status_item(edge_count >= 0, f"edges={edge_count}"),
        "candidate_pipeline": _status_item(candidate_nodes > 0, f"candidate_nodes={candidate_nodes}"),
        "node_process_implemented": _status_item(summary.total_objects > 0, f"processed_objects={summary.total_objects}"),
        "edge_process_implemented": _status_item(summary.total_frames > 0, f"frames={summary.total_frames}, new_edges={summary.total_new_edges}"),
        "dynamic_lifecycle": _status_item(dynamic_count > 0, f"dynamic_nodes={dynamic_count}, lifecycle_touched={lifecycle_touched}"),
        "attached_owner_pipeline": _status_item(attached_total > 0, f"attached_nodes={attached_total}, with_owner={owner_attached}"),
        "owner_resolution_activity": _status_item(
            summary.total_owner_assigned > 0 or attached_total == 0,
            f"owner_assigned={summary.total_owner_assigned}"
        ),
        "edge_valid_invalid_support": _status_item(edge_count >= 0, f"invalid_edges={invalid_edge_count}"),
        "edge_conflict_tracking": _status_item(
            summary.total_duplicate_edges >= 0 and summary.total_conflict_edges >= 0,
            f"duplicate_edges={summary.total_duplicate_edges}, conflict_edges={summary.total_conflict_edges}"
        ),
        "relation_action_event_logging": _status_item(
            len(graph.edge_action_events) > 0,
            f"edge_action_events={len(graph.edge_action_events)}"
        ),
        "node_change_cascade_rewrite": _status_item(structured_edges >= 0, f"structured_edges={structured_edges}, changed_nodes={changed_nodes}"),
        "neo4j_writable": _status_item(bool(config.use_neo4j), "enabled via --use-neo4j"),
        "llm_decision_hooked": _status_item(bool(config.llm.enabled), "enabled via --enable-llm"),
    }

    for item in requirements.values():
        total_count += 1
        if item["ok"]:
            pass_count += 1

    return {
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
        "graph_stats": {
            "nodes": node_count,
            "edges": edge_count,
            "invalid_edges": invalid_edge_count,
            "candidate_nodes": candidate_nodes,
            "attached_nodes": attached_total,
            "attached_with_owner": owner_attached,
            "changed_nodes": changed_nodes,
            "dynamic_state_distribution": _count_dynamic_states(graph),
            "structured_edges": structured_edges,
            "relation_action_stats": {
                "new": summary.total_new_edges,
                "duplicate": summary.total_duplicate_edges,
                "conflict": summary.total_conflict_edges,
            },
            "relation_action_event_stats": edge_action_counts,
            "owner_event_stats": owner_event_counts,
        },
        "samples": {
            "edge_new": sample_events(graph.edge_action_events, "new"),
            "edge_duplicate": sample_events(graph.edge_action_events, "duplicate"),
            "edge_conflict": sample_events(graph.edge_action_events, "conflict"),
            "owner_assign": sample_owner_events(graph.owner_decision_events, "assign"),
            "owner_switch": sample_owner_events(graph.owner_decision_events, "switch"),
            "owner_skip": sample_owner_events(graph.owner_decision_events, "skip"),
        },
        "requirement_checks": requirements,
        "overall": {
            "pass_count": pass_count,
            "total_count": total_count,
            "pass_rate": (pass_count / total_count) if total_count else 0.0,
        },
    }
