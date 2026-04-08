from __future__ import annotations

"""系统配置定义。

本文件集中管理三类配置：
1. 几何候选检索阈值（ThresholdConfig）。
2. 动态节点生命周期规则（DynamicConfig）。
3. LLM 与整体引擎行为（LLMConfig/EngineConfig）。
"""

from dataclasses import dataclass, field
import os
from typing import Set


@dataclass(slots=True)
class ThresholdConfig:
    """候选检索阈值（首版固定阈值）。

字段含义：
1. cmp_iou_threshold: 本帧重复候选判定的 IoU 阈值。
2. pre_iou_threshold: 跨帧图中候选节点召回的 IoU 阈值。
3. center_dist_threshold_small/large: 小目标/大目标中心距离阈值。
4. size_ratio_min/max: 跨帧尺寸比过滤范围。
5. cur_context_limit: 提供给 LLM 的本帧上下文候选数上限。
"""

    cmp_iou_threshold: float = 0.85
    pre_iou_threshold: float = 0.30
    center_dist_threshold_small: float = 40.0
    center_dist_threshold_large: float = 120.0
    size_ratio_min: float = 0.35
    size_ratio_max: float = 2.8
    cur_context_limit: int = 8


@dataclass(slots=True)
class DynamicConfig:
    """动态节点生命周期规则。"""

    initial_life_value: int = 3
    max_life_value: int = 8


@dataclass(slots=True)
class LLMConfig:
    """LLM 配置（OpenAI 兼容接口）。

说明：
1. enabled=False 时，系统自动回退到规则判定。
2. api_key 默认读 OPENAI_API_KEY 环境变量。
3. model/base_url 可用于切换兼容网关。
"""

    enabled: bool = False
    base_url: str = "https://api.openai.com/v1"
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = "gpt-4o-mini"
    timeout_seconds: int = 40
    temperature: float = 0.1


@dataclass(slots=True)
class EngineConfig:
    """离线流水线总配置。

除阈值/生命周期/LLM 外，还包含：
1. moving_labels: 用于静态/动态初判的先验标签集合。
2. use_neo4j 与连接参数: 控制是否写入图数据库。
3. group_id: 预留分组字段，便于后续多视频隔离。
"""

    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    dynamic: DynamicConfig = field(default_factory=DynamicConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    moving_labels: Set[str] = field(
        default_factory=lambda: {
            "person",
            "man",
            "woman",
            "boy",
            "girl",
            "car",
            "bus",
            "truck",
            "bike",
            "bicycle",
            "motorcycle",
            "dog",
            "cat",
        }
    )
    use_neo4j: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "stg.llm"
    group_id: str = "default_video"
