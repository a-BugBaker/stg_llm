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
from typing import Iterable, Optional,Any
from .models import EntityNode, RelationEdge


@dataclass(slots=True)
class Neo4jConfig:
    """Neo4j 连接参数。"""
    uri: str
    user: str
    password: str
    database : str


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
            "CREATE CONSTRAINT entity_node_id IF NOT EXISTS FOR (n:EntityNode) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT relation_edge_id IF NOT EXISTS FOR ()-[e:SEMANTIC_REL]-() REQUIRE e.edge_id IS UNIQUE",
            "CREATE INDEX entity_state IF NOT EXISTS FOR (n:EntityNode) ON (n.state)",
            "CREATE INDEX edge_valid_at IF NOT EXISTS FOR ()-[e:SEMANTIC_REL]-() ON (e.valid_at)",
        ]

        with self._get_session() as session:  # 修改这里
            for query in queries:
                session.run(query)
        

    def upsert_nodes(self, nodes: Iterable[EntityNode]) -> None:
        """写入/更新节点。"""
        if self._driver is None:
            return

        query = (
            "MERGE (n:EntityNode {node_id: $node_id}) "
            "SET n.entity_type=$entity_type, "
            "n.latest_label=$latest_label, "
            "n.last_matched=$last_matched, "
            "n.state=$state, "
            "n.life_value=$life_value"
        )

        with self._get_session() as session:  # 修改这里
            for node in nodes:
                session.run(
                    query,
                    node_id=node.id,
                    entity_type=node.entity_type.value,
                    latest_label=node.latest_label(),
                    last_matched=node.last_matched,
                    state=node.state.value,
                    life_value=node.life_value,
                )

    def upsert_edges(self, edges: Iterable[RelationEdge]) -> None:
        """写入/更新边。"""
        if self._driver is None:
            return

        query = (
            "MATCH (a:EntityNode {node_id: $from_id}) "
            "MATCH (b:EntityNode {node_id: $to_id}) "
            "MERGE (a)-[e:SEMANTIC_REL {edge_id: $edge_id}]->(b) "
            "SET e.describe=$describe, "
            "e.predicate=$predicate, "
            "e.source_label=$source_label, "
            "e.target_label=$target_label, "
            "e.edge_type=$edge_type, "
            "e.valid_at=$valid_at, "
            "e.invalid_at=$invalid_at"
        )

        with self._get_session() as session:  # 修改这里
            for edge in edges:
                session.run(
                    query,
                    from_id=edge.from_node_id,
                    to_id=edge.to_node_id,
                    edge_id=edge.id,
                    describe=edge.describe,
                    predicate=edge.predicate,
                    source_label=edge.source_label,
                    target_label=edge.target_label,
                    edge_type=edge.edge_type.value,
                    valid_at=edge.valid_at,
                    invalid_at=edge.invalid_at,
                )
