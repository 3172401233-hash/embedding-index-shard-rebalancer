"""Data models for the shard rebalancing planner.

All quantities are pure metadata: the planner never touches a real
vector database, it only reasons about shard statistics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class ShardState(Enum):
    """Lifecycle states a shard can be in during a migration stage."""

    ACTIVE = "active"                # serving queries normally
    SPLITTING = "splitting"          # split plan computed, copy pending
    BUILDING = "building"            # new index is being built
    VERIFYING = "verifying"          # count + digest validation running
    DUAL_QUERY = "dual_query"        # serving together with its replacement
    RETIRING = "retiring"            # drain in progress, no new queries
    RETIRED = "retired"              # fully removed from the serving set
    FAILED = "failed"                # migration failed, shard kept active
    BLOCKED = "blocked"              # cannot be migrated, see reasons


class StageKind(Enum):
    COPY = "copy"
    VERIFY = "verify"
    DUAL_QUERY = "dual_query"
    RETIRE = "retire"


@dataclass
class ShardSpec:
    """Static description of one existing index shard."""

    shard_id: str
    size_bytes: int
    vector_count: int
    centroids: Dict[str, float]          # centroid id -> routing weight
    replicas: int
    qps: float
    build_time_seconds: float            # estimated time to rebuild this shard
    deleted_ids: Set[str] = field(default_factory=set)  # pending deletes

    @property
    def live_count(self) -> int:
        """Vectors that should survive in a rebuilt index."""
        return max(0, self.vector_count - len(self.deleted_ids))


@dataclass
class ChildShardSpec:
    """A shard produced by splitting a source shard."""

    shard_id: str
    source_id: str
    size_bytes: int
    vector_count: int                    # live vectors after deletes applied
    centroids: Dict[str, float]
    replicas: int
    build_time_seconds: float


@dataclass
class PlannerConfig:
    """Cluster level budgets and the desired topology."""

    target_shard_count: int
    memory_limit_bytes: int              # max bytes one in-flight build may use
    query_budget_qps: float              # max total QPS allowed during dual query
    min_replicas: int = 2                # replicas every serving shard must keep
    max_recall_risk: float = 0.05        # warn above this estimated risk
    dual_query_overlap_seconds: float = 30.0  # safety margin after verify
    verify_seconds: float = 5.0


@dataclass
class MigrationTask:
    """Plan for replacing one source shard by one or more children."""

    source: ShardSpec
    children: List[ChildShardSpec]
    feasible: bool
    blocked_reasons: List[str]
    recall_risk: float
    warnings: List[str] = field(default_factory=list)
    dual_query_window: Optional[Tuple[float, float]] = None


@dataclass
class StageRecord:
    """One observable stage of the executed plan."""

    sequence: int
    kind: StageKind
    shard_states: Dict[str, ShardState]
    dual_query_window: Optional[Tuple[float, float]]
    recall_risk: float
    notes: List[str] = field(default_factory=list)


@dataclass
class MigrationPlan:
    """Full planner output."""

    tasks: List[MigrationTask]
    stages: List[StageRecord]
    unmigratable: Dict[str, List[str]]
    target_shard_count: int
    achieved_shard_count: int
