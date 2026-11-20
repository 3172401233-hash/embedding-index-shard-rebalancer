"""Vector index shard rebalancing planner (simulation only).

This library plans shard split / copy / dual-query / retire steps from
shard metadata.  It never talks to a real vector database.
"""

from .models import (
    ChildShardSpec,
    MigrationPlan,
    MigrationTask,
    PlannerConfig,
    ShardSpec,
    ShardState,
    StageKind,
    StageRecord,
)
from .planner import RebalancePlanner, estimate_recall_risk
from .executor import (
    ExecutionResult,
    FaultInjection,
    PlanExecutor,
    SimulatedCluster,
    digest_ids,
)

__all__ = [
    "ChildShardSpec",
    "ExecutionResult",
    "FaultInjection",
    "MigrationPlan",
    "MigrationTask",
    "PlanExecutor",
    "PlannerConfig",
    "RebalancePlanner",
    "ShardSpec",
    "ShardState",
    "SimulatedCluster",
    "StageKind",
    "StageRecord",
    "digest_ids",
    "estimate_recall_risk",
]
