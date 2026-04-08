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


def size_ratio(box_a: Box, box_b: Box) -> float:
    """计算两个框面积比（小面积/大面积）。

用于跨帧匹配时过滤尺度差异过大的候选。
"""
    a = area(box_a)
    b = area(box_b)
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return min(a, b) / max(a, b)


def is_small_object(box: Box, frame_diag: float) -> bool:
    """判断目标是否为小目标。

规则：目标对角线 < 帧对角线 10%。
"""
    w = box_width(box)
    h = box_height(box)
    diag = (w * w + h * h) ** 0.5
    if frame_diag <= 0.0:
        return False
    return diag < 0.10 * frame_diag


def frame_diagonal(objects: List[dict]) -> float:
    """估计当前帧的几何尺度（对角线）。

简化实现：取所有框的最大 x2/y2 作为画幅边界。
"""
    max_x2 = 0.0
    max_y2 = 0.0
    for obj in objects:
        box = obj.get("box", [0, 0, 0, 0])
        max_x2 = max(max_x2, float(box[2]))
        max_y2 = max(max_y2, float(box[3]))
    return (max_x2 * max_x2 + max_y2 * max_y2) ** 0.5
