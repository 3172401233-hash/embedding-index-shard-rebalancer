# embedding-index-shard-rebalancer

规划向量索引分片迁移、副本一致性和查询连续性。

> **注意**：本库只做规划与仿真，**不调用任何真实的向量数据库**。
> `SimulatedCluster` 只是内存中的向量 ID 集合，用于验证迁移计划的安全性。

## 背景

向量索引扩缩容时，分片迁移会让部分查询暂时缺少数据；而直接复制索引又可能
因为构建时间长、内存不足、删除向量不同步而降低召回率。本库根据分片元数据
生成分阶段的迁移计划：拆分（split）→ 复制/构建（copy）→ 校验（verify）→
双层查询（dual query）→ 退休（retire）。

## 安全不变量

- 迁移期间查询必须同时覆盖旧分片和新分片（双层查询窗口）。
- 新分片构建完成并通过数量与摘要（SHA-1 digest）校验后，才允许退休旧分片。
- 热分片迁移不得超过内存上限与查询 QPS 预算，否则标记为不可迁移并给出原因。
- 低频分片也必须保留足够副本（`min_replicas`），副本不足的分片不会被迁移。
- 构建失败或迁移中断时回滚子分片，旧分片保持服务，查询覆盖不丢失。

## 用法

```python
from shard_rebalancer import (
    FaultInjection, PlanExecutor, PlannerConfig, RebalancePlanner,
    ShardSpec, SimulatedCluster,
)

shards = [
    ShardSpec(shard_id="s0", size_bytes=4000, vector_count=400,
              centroids={"c0": 3.0, "c1": 1.0}, replicas=2, qps=200.0,
              build_time_seconds=120.0, deleted_ids={"s0-000001"}),
    # ...更多分片
]
config = PlannerConfig(target_shard_count=4, memory_limit_bytes=8_000,
                       query_budget_qps=1_000.0, min_replicas=2)

plan = RebalancePlanner(config).plan(shards)
for stage in plan.stages:          # 每阶段的分片状态、双查窗口、召回风险
    print(stage.sequence, stage.kind.value, stage.shard_states,
          stage.dual_query_window, stage.recall_risk)
print(plan.unmigratable)           # 无法迁移的分片及原因

cluster = SimulatedCluster.from_specs(shards)
result = PlanExecutor(cluster, FaultInjection()).execute(plan)
```

## 输入与输出

- **输入**：每个分片的大小、向量数量、质心分布、副本数、查询 QPS、预计构建
  时间、待删除向量 ID，以及集群级的内存上限、查询预算、目标分片数。
- **输出**：`MigrationPlan`，包含每个迁移任务的可行性、阻塞原因、估计召回
  风险（删除未同步 + 质心分裂倾斜的启发式加权）、双查窗口，以及按阶段排列
  的分片状态快照。

## 测试

```bash
python -m unittest
```

覆盖均匀分裂、热点分片（QPS/内存预算）、删除记录、副本不足、构建失败和
迁移中断六类场景。兼容 Python 3.8+。
