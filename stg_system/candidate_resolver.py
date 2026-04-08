from __future__ import annotations

"""候选检索模块（get_candidates）。

目标：为每个待处理对象生成三类候选：
1. cur_cmp: 本帧高重叠重复候选（用于去重）。
2. cur_context: 本帧上下文候选（辅助 LLM 判定）。
3. pre: 图中历史候选节点（用于跨帧匹配）。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import EngineConfig
from .geometry import center_distance, frame_diagonal, iou, is_small_object, size_ratio
from .models import EntityNode, GraphState


@dataclass(slots=True)
class CurrentObjectCandidate:
    """本帧对象候选记录。"""
    idx: int
    iou: float
    distance: float
    width: float
    height: float


@dataclass(slots=True)
class PreviousNodeCandidate:
    """历史语义节点候选记录。"""
    node_id: int
    iou: float
    distance: float
    width: float
    height: float


@dataclass(slots=True)
class CandidateResult:
    """get_candidates 统一返回结构。"""
    cur_cmp: List[CurrentObjectCandidate]
    cur_context: List[CurrentObjectCandidate]
    pre: Optional[List[PreviousNodeCandidate]]


def _wh(box: List[float]) -> tuple[float, float]:
    """返回检测框宽高。"""
    return float(box[2]) - float(box[0]), float(box[3]) - float(box[1])


def _distance_threshold(box: List[float], objects: List[dict], cfg: EngineConfig) -> float:
    """按目标尺度返回中心距离阈值。"""
    diag = frame_diagonal(objects)
    if is_small_object(box, diag):
        return cfg.thresholds.center_dist_threshold_small
    return cfg.thresholds.center_dist_threshold_large


def get_candidates(
    idx: int,
    objects: List[dict],
    frame_id: int,
    graph: GraphState,
    cfg: EngineConfig,
) -> CandidateResult:
    """为当前 idx 构建候选集合。

实现逻辑：
1. 在当前帧中找重复候选（cur_cmp）和上下文候选（cur_context）。
2. 若是首帧，pre 为空。
3. 非首帧时，按尺寸比 + IoU/距离从图中召回历史节点（pre）。
"""
    target = objects[idx]
    target_box = target["box"]
    dist_thr = _distance_threshold(target_box, objects, cfg)

    cur_cmp: List[CurrentObjectCandidate] = []
    cur_context: List[CurrentObjectCandidate] = []

    for other in objects:
        other_idx = int(other["idx"])
        if other_idx == idx:
            continue

        other_box = other["box"]
        cur_iou = iou(target_box, other_box)
        dist = center_distance(target_box, other_box)
        w, h = _wh(other_box)

        if cur_iou >= cfg.thresholds.cmp_iou_threshold:
            cur_cmp.append(CurrentObjectCandidate(other_idx, cur_iou, dist, w, h))

        if dist <= dist_thr:
            cur_context.append(CurrentObjectCandidate(other_idx, cur_iou, dist, w, h))

    cur_cmp.append(CurrentObjectCandidate(idx, 1.0, 0.0, *_wh(target_box)))
    cur_cmp.sort(key=lambda x: x.iou, reverse=True)

    cur_context.sort(key=lambda x: x.iou, reverse=True)
    cur_context = cur_context[: cfg.thresholds.cur_context_limit]

    if frame_id == 0:
        return CandidateResult(cur_cmp=cur_cmp, cur_context=cur_context, pre=None)

    pre: List[PreviousNodeCandidate] = []
    target_w, target_h = _wh(target_box)

    for node in graph.nodes.values():
        node_box = node.latest_box()
        if not node_box:
            continue

        ratio = size_ratio(target_box, node_box)
        if ratio <= 0.0:
            continue
        if ratio < cfg.thresholds.size_ratio_min or ratio > cfg.thresholds.size_ratio_max:
            continue

        cand_iou = iou(target_box, node_box)
        dist = center_distance(target_box, node_box)
        node_w, node_h = _wh(node_box)

        if cand_iou >= cfg.thresholds.pre_iou_threshold or dist <= dist_thr:
            pre.append(PreviousNodeCandidate(node.id, cand_iou, dist, node_w, node_h))

    pre.sort(key=lambda x: (x.iou, -x.distance), reverse=True)
    return CandidateResult(cur_cmp=cur_cmp, cur_context=cur_context, pre=pre)
