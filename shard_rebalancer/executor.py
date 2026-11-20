"""Simulated plan executor.

The executor runs a MigrationPlan against an in-memory fake cluster.
It enforces the safety invariants the planner promises:

* during migration, queries are answered by the union of the old shard
  and its children (dual query);
* the old shard is retired only after every child passed count and
  digest validation;
* on build failure or interruption the old shard stays active and any
  partial children are rolled back, so no query ever loses coverage.

No real vector database is involved; SimulatedCluster just holds
sets of vector ids.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .models import (
    MigrationPlan,
    MigrationTask,
    ShardState,
    StageKind,
    StageRecord,
)


def digest_ids(ids) -> str:
    """Stable content digest used by the verify step."""
    return hashlib.sha1(",".join(sorted(ids)).encode("utf-8")).hexdigest()


class SimulatedCluster:
    """In-memory stand-in for a vector index cluster."""

    def __init__(self):
        # shard_id -> {"ids": set, "state": ShardState, "replicas": int}
        self.shards: Dict[str, Dict] = {}

    @classmethod
    def from_specs(cls, specs) -> "SimulatedCluster":
        cluster = cls()
        for spec in specs:
            ids = {"%s-%06d" % (spec.shard_id, i) for i in range(spec.vector_count)}
            cluster.shards[spec.shard_id] = {
                "ids": ids,
                "state": ShardState.ACTIVE,
                "replicas": spec.replicas,
            }
        return cluster

    def serving_ids(self) -> Set[str]:
        """Ids visible to queries right now (union of serving shards)."""
        visible = set()
        for entry in self.shards.values():
            if entry["state"] in (ShardState.ACTIVE, ShardState.DUAL_QUERY,
                                  ShardState.VERIFYING, ShardState.SPLITTING):
                visible |= entry["ids"]
        return visible


@dataclass
class FaultInjection:
    """Deterministic faults for testing the failure paths."""

    fail_build_for: Set[str] = field(default_factory=set)  # source shard ids
    interrupt_after_stages: Optional[int] = None           # stop the plan early


class MigrationInterrupted(Exception):
    pass


@dataclass
class ExecutionResult:
    stages: List[StageRecord]
    failed_tasks: Dict[str, List[str]]
    interrupted: bool
    final_states: Dict[str, ShardState]


class PlanExecutor:
    """Executes a plan on a SimulatedCluster."""

    def __init__(self, cluster: SimulatedCluster, faults: Optional[FaultInjection] = None):
        self.cluster = cluster
        self.faults = faults or FaultInjection()

    def _expected_child_ids(self, source_ids, deleted_ids, child_index, child_count):
        live = sorted(set(source_ids) - set(deleted_ids))
        return {vid for i, vid in enumerate(live) if i % child_count == child_index}

    def _rollback(self, task: MigrationTask) -> None:
        for child in task.children:
            self.cluster.shards.pop(child.shard_id, None)
        self.cluster.shards[task.source.shard_id]["state"] = ShardState.ACTIVE

    def execute(self, plan: MigrationPlan) -> ExecutionResult:
        stages: List[StageRecord] = []
        failed: Dict[str, List[str]] = {}
        interrupted = False
        seq = 0

        def record(kind, task, notes):
            nonlocal seq
            states = {sid: e["state"] for sid, e in self.cluster.shards.items()}
            window = task.dual_query_window if task else None
            risk = task.recall_risk if task else 0.0
            stages.append(StageRecord(seq, kind, states, window, risk, list(notes)))
            seq += 1
            if self.faults.interrupt_after_stages is not None and \
                    seq >= self.faults.interrupt_after_stages:
                raise MigrationInterrupted()

        try:
            for task in plan.tasks:
                if not task.feasible:
                    failed[task.source.shard_id] = list(task.blocked_reasons)
                    record(StageKind.COPY, task,
                           ["blocked: " + r for r in task.blocked_reasons])
                    continue
                self._run_task(task, record, failed)
        except MigrationInterrupted:
            interrupted = True
            # Roll back anything still in flight so coverage is never lost.
            for task in plan.tasks:
                src = self.cluster.shards.get(task.source.shard_id)
                if src is not None and src["state"] != ShardState.ACTIVE:
                    self._rollback(task)
                    failed.setdefault(task.source.shard_id, []).append(
                        "migration interrupted; rolled back to source shard")

        final = {sid: e["state"] for sid, e in self.cluster.shards.items()}
        return ExecutionResult(stages, failed, interrupted, final)

    def _run_task(self, task: MigrationTask, record, failed) -> None:
        src_id = task.source.shard_id
        source_entry = self.cluster.shards[src_id]

        # COPY / BUILD children.
        source_entry["state"] = ShardState.SPLITTING
        if src_id in self.faults.fail_build_for:
            source_entry["state"] = ShardState.FAILED
            record(StageKind.COPY, task, ["build failed for %s" % src_id])
            self._rollback(task)
            failed.setdefault(src_id, []).append("child index build failed")
            return
        for i, child in enumerate(task.children):
            ids = self._expected_child_ids(
                source_entry["ids"], task.source.deleted_ids, i, len(task.children))
            self.cluster.shards[child.shard_id] = {
                "ids": ids,
                "state": ShardState.BUILDING,
                "replicas": child.replicas,
            }
        record(StageKind.COPY, task,
               ["built %d children for %s" % (len(task.children), src_id)])

        # VERIFY counts and digests before any traffic shift.
        source_entry["state"] = ShardState.VERIFYING
        expected_live = sorted(set(source_entry["ids"]) - set(task.source.deleted_ids))
        rebuilt = set()
        for child in task.children:
            rebuilt |= self.cluster.shards[child.shard_id]["ids"]
        for child in task.children:
            self.cluster.shards[child.shard_id]["state"] = ShardState.VERIFYING
        if len(rebuilt) != len(expected_live) or \
                digest_ids(rebuilt) != digest_ids(expected_live):
            record(StageKind.VERIFY, task, ["count/digest mismatch after rebuild"])
            self._rollback(task)
            failed.setdefault(src_id, []).append("count/digest mismatch after rebuild")
            return
        record(StageKind.VERIFY, task, ["verified %d live vectors" % len(rebuilt)])

        # DUAL QUERY: old and new serve together.
        source_entry["state"] = ShardState.DUAL_QUERY
        for child in task.children:
            self.cluster.shards[child.shard_id]["state"] = ShardState.ACTIVE
        record(StageKind.DUAL_QUERY, task,
               ["dual query window %.1f..%.1f" % task.dual_query_window])

        # RETIRE the source only after verification and overlap.
        source_entry["state"] = ShardState.RETIRED
        record(StageKind.RETIRE, task, ["retired %s" % src_id])
