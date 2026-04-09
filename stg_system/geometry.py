from __future__ import annotations

"""几何工具函数。

该模块只做与检测框相关的纯数学计算，不依赖图状态：
1. 基本尺寸（宽/高/面积/中心）。
2. 框间关系（中心距离、IoU、尺寸比）。
3. 目标尺度判定（是否小目标）与帧对角线估计。
"""

from typing import List, Sequence, Tuple

Box = Sequence[float]


def box_width(box: Box) -> float:
    """计算检测框宽度，自动裁剪为非负值。"""
    return max(0.0, float(box[2]) - float(box[0]))


def box_height(box: Box) -> float:
    """计算检测框高度，自动裁剪为非负值。"""
    return max(0.0, float(box[3]) - float(box[1]))


def area(box: Box) -> float:
    """计算检测框面积。"""
    return box_width(box) * box_height(box)


def center(box: Box) -> Tuple[float, float]:
    """计算检测框中心点坐标。"""
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def center_distance(box_a: Box, box_b: Box) -> float:
    """计算两个检测框中心点的欧氏距离。"""
    ax, ay = center(box_a)
    bx, by = center(box_b)
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def iou(box_a: Box, box_b: Box) -> float:
    """计算两个检测框的 IoU。

返回值范围 [0, 1]。当并集为 0 时返回 0。
"""
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = area(box_a) + area(box_b) - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def size_ratio(box_a: Box, box_b: Box) -> tuple[float, float]:
    """计算两个框长宽比（node/target）。
    用于跨帧匹配时过滤尺度差异过大的候选。
    """
    w_ratio = box_width(box_b) / box_width(box_a) if box_width(box_a) > 0 else 0.0
    h_ratio = box_height(box_b) / box_height(box_a) if box_height(box_a) > 0 else 0.0
    
    return w_ratio, h_ratio






