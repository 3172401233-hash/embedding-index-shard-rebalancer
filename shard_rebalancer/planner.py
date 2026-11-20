"""Rebalance planning: split, copy, dual-query and retire scheduling.

The planner is deterministic and side-effect free.  It receives shard
metadata and cluster budgets and returns a :class:`MigrationPlan`
describing every stage, the dual-query windows, the estimated recall
risk and the reasons why a shard cannot be migrated.
"""

from typing import Dict, List, Tuple

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

# Recall-risk heuristic weights.  Deletes that are still pending when a
# naive copy is made are the dominant recall killer; centroid imbalance
# between the children adds a smaller routing-miss component.
_DELETE_WEIGHT = 0.6
_IMBALANCE_WEIGHT = 0.4


def _split_centroids(centroids: Dict[str, float], parts: int) -> List[Dict[str, float]]:
    """Greedily assign centroids to ``parts`` buckets, balancing weight."""
    buckets = [dict() for _ in range(parts)]
    loads = [0.0] * parts
    for cid, weight in sorted(centroids.items(), key=lambda kv: (-kv[1], kv[0])):
        idx = min(range(parts), key=lambda i: (loads[i], i))
        buckets[idx][cid] = weight
        loads[idx] += weight
    return buckets


def _largest_remainder(total: int, weights: List[float]) -> List[int]:
    """Distribute ``total`` items proportionally to ``weights``."""
    weight_sum = sum(weights) or 1.0
    raw = [total * w / weight_sum for w in weights]
    shares = [int(r) for r in raw]
    remainder = total - sum(shares)
    order = sorted(range(len(weights)), key=lambda i: (raw[i] - shares[i], -i), reverse=True)
    for i in order[:remainder]:
        shares[i] += 1
    return shares


def _centroid_imbalance(children: List[ChildShardSpec]) -> float:
    shares = [c.vector_count for c in children]
    if not shares or sum(shares) == 0:
        return 0.0
    mean = sum(shares) / len(shares)
    if mean == 0:
        return 0.0
    spread = (max(shares) - min(shares)) / 2.0
    return min(1.0, spread / mean)


def estimate_recall_risk(source: ShardSpec, children: List[ChildShardSpec]) -> float:
    """Heuristic recall risk in ``[0, 1]`` for one migration task.

    * pending deletes that a plain copy would resurrect dominate the risk;
    * skewed centroid splits make query routing more likely to miss a
      child during the dual-query window.
    """
    delete_ratio = len(source.deleted_ids) / max(1, source.vector_count)
    imbalance = _centroid_imbalance(children)
    return min(1.0, _DELETE_WEIGHT * delete_ratio + _IMBALANCE_WEIGHT * imbalance)


class RebalancePlanner:
    """Computes a staged migration plan from shard metadata."""

    def __init__(self, config: PlannerConfig):
        self.config = config

    # -- split-factor decision -------------------------------------------------

    def _split_factors(self, shards: List[ShardSpec]) -> Dict[str, int]:
        """Decide how many children each shard should be split into."""
        factors = {s.shard_id: 1 for s in shards}
        extra = self.config.target_shard_count - len(shards)
        if extra <= 0:
            return factors
        total_size = sum(s.size_bytes for s in shards) or 1
        target_size = total_size / float(self.config.target_shard_count)
        # Hottest (qps-weighted) and largest shards are split first.
        order = sorted(shards, key=lambda s: (-s.qps * s.size_bytes, s.shard_id))
        for shard in order:
            if extra <= 0:
                break
            want = max(2, round(shard.size_bytes / target_size))
            want = min(want, extra + 1)
            if want > 1:
                factors[shard.shard_id] = want
                extra -= want - 1
        # If budget is still unspent, keep splitting the hottest splittable shard.
        while extra > 0:
            candidates = [s for s in order if factors[s.shard_id] > 1] or order
            pick = candidates[0]
            factors[pick.shard_id] += 1
            extra -= 1
        return factors

    # -- per-shard task construction -------------------------------------------

    def _build_children(self, source: ShardSpec, parts: int) -> List[ChildShardSpec]:
        buckets = _split_centroids(source.centroids, parts)
        weights = [sum(b.values()) for b in buckets]
        live_shares = _largest_remainder(source.live_count, weights)
        size_shares = _largest_remainder(source.size_bytes, weights)
        children = []
        for i in range(parts):
            children.append(
                ChildShardSpec(
                    shard_id="%s.part%d" % (source.shard_id, i),
                    source_id=source.shard_id,
                    size_bytes=size_shares[i],
                    vector_count=live_shares[i],
                    centroids=buckets[i],
                    replicas=source.replicas,
                    build_time_seconds=source.build_time_seconds / parts,
                )
            )
        return children

    def _check_feasibility(self, source: ShardSpec,
                           children: List[ChildShardSpec]) -> List[str]:
        reasons = []
        if source.replicas < self.config.min_replicas:
            reasons.append(
                "replicas %d below minimum %d: cannot retire safely"
                % (source.replicas, self.config.min_replicas)
            )
        peak_child = max((c.size_bytes for c in children), default=0)
        if peak_child > self.config.memory_limit_bytes:
            reasons.append(
                "child build needs %d bytes, memory limit is %d"
                % (peak_child, self.config.memory_limit_bytes)
            )
        # Dual query doubles the shard's query fan-out.
        if source.qps * 2 > self.config.query_budget_qps:
            reasons.append(
                "dual query needs %.1f qps, budget is %.1f"
                % (source.qps * 2, self.config.query_budget_qps)
            )
        return reasons

    # -- public API --------------------------------------------------------------

    def plan(self, shards: List[ShardSpec]) -> MigrationPlan:
        factors = self._split_factors(shards)
        tasks: List[MigrationTask] = []
        unmigratable: Dict[str, List[str]] = {}
        clock = 0.0

        for source in shards:
            parts = factors[source.shard_id]
            if parts <= 1:
                continue  # shard keeps its shape; nothing to migrate
            children = self._build_children(source, parts)
            reasons = self._check_feasibility(source, children)
            risk = estimate_recall_risk(source, children)
            warnings = []
            if risk > self.config.max_recall_risk:
                warnings.append(
                    "estimated recall risk %.3f exceeds budget %.3f"
                    % (risk, self.config.max_recall_risk)
                )
            if reasons:
                unmigratable[source.shard_id] = reasons
                tasks.append(MigrationTask(source, children, False, reasons, risk, warnings))
                continue
            # Timeline: build children, verify, dual-query overlap, retire.
            build_end = clock + max(c.build_time_seconds for c in children)
            verify_end = build_end + self.config.verify_seconds
            retire_at = verify_end + self.config.dual_query_overlap_seconds
            window = (build_end, retire_at)
            tasks.append(
                MigrationTask(source, children, True, [], risk, warnings,
                              dual_query_window=window)
            )
            clock = retire_at

        stages = self._render_stages(shards, tasks)
        achieved = len(shards) + sum(
            len(t.children) - 1 for t in tasks if t.feasible
        )
        return MigrationPlan(
            tasks=tasks,
            stages=stages,
            unmigratable=unmigratable,
            target_shard_count=self.config.target_shard_count,
            achieved_shard_count=achieved,
        )

    # -- stage rendering ---------------------------------------------------------

    def _render_stages(self, shards: List[ShardSpec],
                       tasks: List[MigrationTask]) -> List[StageRecord]:
        states = {s.shard_id: ShardState.ACTIVE for s in shards}
        stages: List[StageRecord] = []
        seq = 0
        task_by_source = {t.source.shard_id: t for t in tasks}

        def snapshot(kind, task, notes):
            nonlocal seq
            window = task.dual_query_window if task else None
            risk = task.recall_risk if task else 0.0
            stages.append(StageRecord(seq, kind, dict(states), window, risk, notes))
            seq += 1

        for source in shards:
            task = task_by_source.get(source.shard_id)
            if task is None:
                continue
            if not task.feasible:
                states[source.shard_id] = ShardState.BLOCKED
                snapshot(StageKind.COPY, task,
                         ["blocked: " + r for r in task.blocked_reasons])
                states[source.shard_id] = ShardState.ACTIVE
                continue
            states[source.shard_id] = ShardState.SPLITTING
            for child in task.children:
                states[child.shard_id] = ShardState.BUILDING
            snapshot(StageKind.COPY, task,
                     ["copying %s into %d children" % (source.shard_id, len(task.children))])
            states[source.shard_id] = ShardState.VERIFYING
            for child in task.children:
                states[child.shard_id] = ShardState.VERIFYING
            snapshot(StageKind.VERIFY, task,
                     ["verifying live count %d and digest" % source.live_count])
            states[source.shard_id] = ShardState.DUAL_QUERY
            for child in task.children:
                states[child.shard_id] = ShardState.ACTIVE
            snapshot(StageKind.DUAL_QUERY, task,
                     ["dual query window %.1f..%.1f" % task.dual_query_window])
            states[source.shard_id] = ShardState.RETIRED
            snapshot(StageKind.RETIRE, task,
                     ["retired %s" % source.shard_id])
        return stages
