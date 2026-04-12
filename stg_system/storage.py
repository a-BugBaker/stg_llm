from __future__ import annotations

"""Neo4j 持久化模块。

职责：
1. 连接与关闭数据库。
2. 初始化约束与索引。
3. 批量写入节点与边。

说明：
本模块是薄封装，业务判定逻辑全部留在处理器层。
"""

from dataclasses import dataclass
import importlib
import json
from typing import Iterable, Optional,Any
from .models import EntityNode, RelationEdge


@dataclass(slots=True)
class Neo4jConfig:
    """Neo4j 连接参数。"""
    uri: str
    user: str
    password: str
    database : str
    sample_id: str


class Neo4jStorage:
    """将语义图写入 Neo4j 的存储层。"""

    def __init__(self, config: Neo4jConfig):
        self._config = config
        self._driver = None

    def connect(self) -> None:
        """建立数据库连接。"""
        try:
            neo4j_module = importlib.import_module("neo4j")
            graph_database = getattr(neo4j_module, "GraphDatabase")
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("neo4j package is required for Neo4jStorage") from exc

        self._driver = graph_database.driver(
            self._config.uri,
            auth=(self._config.user, self._config.password),
        )

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _get_session(self) :
        """获取配置中指定数据库的会话。"""
        if self._driver is None:
            self.connect()
        return self._driver.session(database=self._config.database)
       

    def ensure_schema(self) -> None:
        """创建必要约束与索引。"""
        if self._driver is None:
            return

        queries = [
            "CREATE CONSTRAINT entity_node_sample_node IF NOT EXISTS FOR (n:EntityNode) REQUIRE (n.sample_id, n.node_id) IS UNIQUE",
            "CREATE CONSTRAINT relation_edge_sample_edge IF NOT EXISTS FOR ()-[e:SEMANTIC_REL]-() REQUIRE (e.sample_id, e.edge_id) IS UNIQUE",
            "CREATE INDEX entity_state IF NOT EXISTS FOR (n:EntityNode) ON (n.state)",
            "CREATE INDEX entity_sample_id IF NOT EXISTS FOR (n:EntityNode) ON (n.sample_id)",
            "CREATE INDEX edge_valid_at IF NOT EXISTS FOR ()-[e:SEMANTIC_REL]-() ON (e.valid_at)",
            "CREATE INDEX edge_sample_id IF NOT EXISTS FOR ()-[e:SEMANTIC_REL]-() ON (e.sample_id)",
        ]

        with self._get_session() as session:  # 修改这里
            self._drop_legacy_constraints(session)
            for query in queries:
                session.run(query)

    def _drop_legacy_constraints(self, session) -> None:
        """删除旧版仅按 node_id/edge_id 的唯一约束，避免 sample_id 分片写入冲突。"""
        # Best-effort fixed-name cleanup.
        session.run("DROP CONSTRAINT entity_node_id IF EXISTS")
        session.run("DROP CONSTRAINT relation_edge_id IF EXISTS")

        # Robust cleanup for unknown legacy names.
        rows = session.run(
            "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties"
        )
        for row in rows:
            name = row.get("name")
            ctype = str(row.get("type", ""))
            labels_or_types = row.get("labelsOrTypes") or []
            properties = row.get("properties") or []

            is_unique = "UNIQUE" in ctype.upper()
            if not is_unique or not name:
                continue

            drop_needed = False
            if labels_or_types == ["EntityNode"] and properties == ["node_id"]:
                drop_needed = True
            if labels_or_types == ["SEMANTIC_REL"] and properties == ["edge_id"]:
                drop_needed = True

            if drop_needed:
                escaped_name = str(name).replace("`", "``")
                session.run(f"DROP CONSTRAINT `{escaped_name}` IF EXISTS")
        

    def upsert_nodes(self, nodes: Iterable[EntityNode]) -> None:
        """写入/更新节点。"""
        if self._driver is None:
            return

        query = (
            "MERGE (n:EntityNode {sample_id: $sample_id, node_id: $node_id}) "
            "SET n.entity_type=$entity_type, "
            "n.latest_label=$latest_label, "
            "n.latest_attributes=$latest_attributes, "
            "n.last_matched=$last_matched, "
            "n.first_frame=$first_frame, "
            "n.missed_frame=$missed_frame, "
            "n.disappeared_frame=$disappeared_frame, "
            "n.state=$state, "
            "n.life_value=$life_value, "
            "n.owner_node_id=$owner_node_id, "
            "n.owner_label=$owner_label, "
            "n.latest_box=$latest_box, "
            "n.label_history_json=$label_history_json, "
            "n.attributes_history_json=$attributes_history_json, "
            "n.candidate_label_json=$candidate_label_json, "
            "n.candidate_attribute_json=$candidate_attribute_json, "
            "n.candidate_type_json=$candidate_type_json, "
            "n.position_samples_json=$position_samples_json, "
            "n.trajectory_json=$trajectory_json, "
            "n.owner_candidates_json=$owner_candidates_json"
        )

        with self._get_session() as session:  # 修改这里
            for node in nodes:
                latest_attributes = node.attributes[-1][0] if node.attributes else ""
                owner_node_id = node.owner[0] if node.owner is not None else None
                owner_label = node.owner[1] if node.owner is not None else None
                latest_box = node.latest_box()
                session.run(
                    query,
                    sample_id=self._config.sample_id,
                    node_id=node.id,
                    entity_type=node.entity_type.value,
                    latest_label=node.latest_label(),
                    latest_attributes=latest_attributes,
                    last_matched=node.last_matched,
                    first_frame=node.first_frame,
                    missed_frame=node.missed_frame,
                    disappeared_frame=node.disappeared_frame,
                    state=node.state.value,
                    life_value=node.life_value,
                    owner_node_id=owner_node_id,
                    owner_label=owner_label,
                    latest_box=latest_box,
                    label_history_json=json.dumps(node.label, ensure_ascii=False),
                    attributes_history_json=json.dumps(node.attributes, ensure_ascii=False),
                    candidate_label_json=json.dumps(node.candidate.label, ensure_ascii=False),
                    candidate_attribute_json=json.dumps(node.candidate.attribute, ensure_ascii=False),
                    candidate_type_json=json.dumps(node.candidate.type, ensure_ascii=False),
                    position_samples_json=json.dumps(node.position_samples, ensure_ascii=False),
                    trajectory_json=json.dumps(node.trajectory, ensure_ascii=False),
                    owner_candidates_json=json.dumps(node.owner_candidates, ensure_ascii=False),
                )

    def upsert_edges(self, edges: Iterable[RelationEdge]) -> None:
        """写入/更新边。"""
        if self._driver is None:
            return

        query = (
            "MATCH (a:EntityNode {sample_id: $sample_id, node_id: $from_id}) "
            "MATCH (b:EntityNode {sample_id: $sample_id, node_id: $to_id}) "
            "MERGE (a)-[e:SEMANTIC_REL {sample_id: $sample_id, edge_id: $edge_id}]->(b) "
            "SET e.describe=$describe, "
            "e.predicate=$predicate, "
            "e.source_label=$source_label, "
            "e.target_label=$target_label, "
            "e.edge_type=$edge_type, "
            "e.is_attached=$is_attached, "
            "e.valid_at=$valid_at, "
            "e.invalid_at=$invalid_at"
        )

        with self._get_session() as session:  # 修改这里
            for edge in edges:
                session.run(
                    query,
                    sample_id=self._config.sample_id,
                    from_id=edge.from_node_id,
                    to_id=edge.to_node_id,
                    edge_id=edge.id,
                    describe=edge.describe,
                    predicate=edge.predicate,
                    source_label=edge.source_label,
                    target_label=edge.target_label,
                    edge_type=edge.edge_type.value,
                    is_attached=edge.is_attached,
                    valid_at=edge.valid_at,
                    invalid_at=edge.invalid_at,
                )
