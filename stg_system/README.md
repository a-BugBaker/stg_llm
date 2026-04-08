# STG 系统（离线构建版）

本模块用于按帧处理场景图数据，构建时空语义图，并输出快照与验收报告。

## 当前已实现能力

- 语义节点与语义边核心模型（含生命周期字段）。
- 基于几何信息的候选检索（IoU、中心距离、尺寸约束）。
- 节点处理中的 candidate 反思流程（追加、确认、撤销）。
- 边处理中的动作跟踪（new、duplicate、conflict）。
- attached 节点 owner 解析流程（含候选确认）。
- 边与 owner 的可解释事件日志（用于调试与验收）。
- 可选写入 Neo4j。
- 离线 CLI：输入 JSON，输出图快照和验收报告。

## 快速运行

```bash
python -m stg_system.cli --input data/less_move.json --max-frames 20
```

默认输出到 `output` 目录：

- `output/semantic_graph_snapshot.json`
- `output/design_acceptance_report.json`

## 启用 LLM

```bash
python -m stg_system.cli \
  --input data/less_move.json \
  --max-frames 20 \
  --enable-llm \
  --llm-model gpt-4o-mini \
  --llm-api-key "$OPENAI_API_KEY"
```

## 启用 Neo4j 写入

```bash
python -m stg_system.cli \
  --input data/less_move.json \
  --use-neo4j \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password password
```

## 说明

- 未启用 LLM 或密钥不可用时，会自动回退到规则策略。
- 验收报告包含逐条需求状态、关系动作统计、边/owner 决策样例。
- 详细联调步骤见 [docs/运行测试文档.md](docs/%E8%BF%90%E8%A1%8C%E6%B5%8B%E8%AF%95%E6%96%87%E6%A1%A3.md)。
