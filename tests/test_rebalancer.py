import unittest

from shard_rebalancer import (
    FaultInjection,
    PlanExecutor,
    PlannerConfig,
    RebalancePlanner,
    ShardSpec,
    ShardState,
    SimulatedCluster,
    StageKind,
)


def make_shard(shard_id, size=1000, vectors=100, qps=10.0, replicas=2,
               build_time=60.0, deleted=(), centroids=None):
    return ShardSpec(
        shard_id=shard_id,
        size_bytes=size,
        vector_count=vectors,
        centroids=centroids if centroids is not None else {"c0": 1.0, "c1": 1.0},
        replicas=replicas,
        qps=qps,
        build_time_seconds=build_time,
        deleted_ids=set(deleted),
    )


def make_config(target=4, memory=10_000, qps=1_000.0, min_replicas=2):
    return PlannerConfig(
        target_shard_count=target,
        memory_limit_bytes=memory,
        query_budget_qps=qps,
        min_replicas=min_replicas,
    )


def live_ids(spec):
    return {"%s-%06d" % (spec.shard_id, i) for i in range(spec.vector_count)} \
        - set(spec.deleted_ids)


class UniformSplitTest(unittest.TestCase):
    def test_two_even_shards_split_to_four(self):
        specs = [make_shard("s0"), make_shard("s1")]
        plan = RebalancePlanner(make_config(target=4)).plan(specs)

        self.assertEqual(plan.achieved_shard_count, 4)
        self.assertEqual(plan.unmigratable, {})
        for task in plan.tasks:
            self.assertTrue(task.feasible)
            self.assertEqual(len(task.children), 2)
            counts = [c.vector_count for c in task.children]
            self.assertEqual(sum(counts), task.source.live_count)
            self.assertLessEqual(max(counts) - min(counts), 1)
            self.assertIsNotNone(task.dual_query_window)

        # Stage stream contains copy -> verify -> dual query -> retire.
        kinds = [s.kind for s in plan.stages]
        for expected in (StageKind.COPY, StageKind.VERIFY,
                         StageKind.DUAL_QUERY, StageKind.RETIRE):
            self.assertIn(expected, kinds)

        cluster = SimulatedCluster.from_specs(specs)
        result = PlanExecutor(cluster).execute(plan)
        self.assertFalse(result.interrupted)
        self.assertEqual(result.failed_tasks, {})
        active = [sid for sid, st in result.final_states.items()
                  if st == ShardState.ACTIVE]
        self.assertEqual(len(active), 4)
        # No live vector was lost.
        expected = live_ids(specs[0]) | live_ids(specs[1])
        self.assertEqual(cluster.serving_ids(), expected)


class HotShardTest(unittest.TestCase):
    def test_hot_shard_split_within_budgets(self):
        hot = make_shard("hot", size=4000, vectors=400, qps=200.0,
                         centroids={"c%d" % i: float(10 - i) for i in range(4)})
        cold = make_shard("cold", size=200, vectors=20, qps=1.0)
        plan = RebalancePlanner(make_config(target=3, memory=5000, qps=1000.0)) \
            .plan([hot, cold])

        hot_task = next(t for t in plan.tasks if t.source.shard_id == "hot")
        self.assertTrue(hot_task.feasible)
        self.assertEqual(len(hot_task.children), 2)
        # Hottest shard is migrated first so its dual-query window opens first.
        self.assertEqual(plan.tasks[0].source.shard_id, "hot")
        # Low-frequency shard keeps enough replicas on its children.
        for task in plan.tasks:
            for child in task.children:
                self.assertGreaterEqual(child.replicas, 2)

    def test_query_budget_blocks_too_hot_shard(self):
        hot = make_shard("hot", qps=600.0)  # dual query needs 1200 > 1000 budget
        other = make_shard("other", qps=1.0)
        plan = RebalancePlanner(make_config(target=3, qps=1000.0)).plan([hot, other])

        self.assertIn("hot", plan.unmigratable)
        self.assertTrue(any("qps" in r for r in plan.unmigratable["hot"]))
        hot_task = next(t for t in plan.tasks if t.source.shard_id == "hot")
        self.assertFalse(hot_task.feasible)

    def test_memory_limit_blocks_oversized_child(self):
        big = make_shard("big", size=9000, vectors=900)
        other = make_shard("other", size=100, vectors=10)
        plan = RebalancePlanner(make_config(target=3, memory=4000)).plan([big, other])

        self.assertIn("big", plan.unmigratable)
        self.assertTrue(any("memory" in r for r in plan.unmigratable["big"]))


class DeletedRecordsTest(unittest.TestCase):
    def test_pending_deletes_excluded_and_scored(self):
        deleted = {"s0-%06d" % i for i in range(20)}
        spec = make_shard("s0", vectors=100, deleted=deleted)
        filler = make_shard("s1")
        plan = RebalancePlanner(make_config(target=3)).plan([spec, filler])

        task = next(t for t in plan.tasks if t.source.shard_id == "s0")
        self.assertTrue(task.feasible)
        self.assertEqual(sum(c.vector_count for c in task.children), 80)
        # Pending deletes raise the estimated recall risk above zero.
        self.assertGreater(task.recall_risk, 0.0)
        self.assertTrue(task.warnings, "risk warning should be recorded")

        cluster = SimulatedCluster.from_specs([spec, filler])
        result = PlanExecutor(cluster).execute(plan)
        self.assertEqual(result.failed_tasks, {})
        # Deleted vectors never reappear in the rebuilt index.
        self.assertFalse(cluster.serving_ids() & deleted)
        self.assertEqual(len(cluster.serving_ids()), 80 + filler.vector_count)


class InsufficientReplicasTest(unittest.TestCase):
    def test_single_replica_shard_is_not_migrated(self):
        weak = make_shard("weak", replicas=1, qps=50.0)
        healthy = make_shard("healthy", replicas=3, qps=1.0)
        plan = RebalancePlanner(make_config(target=3, min_replicas=2)) \
            .plan([weak, healthy])

        self.assertIn("weak", plan.unmigratable)
        self.assertTrue(any("replicas" in r for r in plan.unmigratable["weak"]))

        cluster = SimulatedCluster.from_specs([weak, healthy])
        result = PlanExecutor(cluster).execute(plan)
        # The blocked shard keeps serving; nothing was retired into it.
        self.assertEqual(result.final_states["weak"], ShardState.ACTIVE)
        self.assertIn("weak", result.failed_tasks)
        self.assertEqual(cluster.serving_ids(),
                         live_ids(weak) | live_ids(healthy))


class BuildFailureTest(unittest.TestCase):
    def test_failed_build_keeps_source_active(self):
        specs = [make_shard("s0"), make_shard("s1")]
        plan = RebalancePlanner(make_config(target=4)).plan(specs)
        cluster = SimulatedCluster.from_specs(specs)
        faults = FaultInjection(fail_build_for={"s0"})
        result = PlanExecutor(cluster, faults).execute(plan)

        self.assertIn("s0", result.failed_tasks)
        self.assertEqual(result.final_states["s0"], ShardState.ACTIVE)
        # Failed children were rolled back.
        self.assertNotIn("s0.part0", result.final_states)
        # The healthy shard still migrated.
        self.assertEqual(result.final_states["s1"], ShardState.RETIRED)
        self.assertEqual(cluster.serving_ids(),
                         live_ids(specs[0]) | live_ids(specs[1]))


class InterruptionTest(unittest.TestCase):
    def test_interrupt_rolls_back_in_flight_task(self):
        specs = [make_shard("s0"), make_shard("s1")]
        plan = RebalancePlanner(make_config(target=4)).plan(specs)
        cluster = SimulatedCluster.from_specs(specs)
        # Stop right after the first COPY stage of the first task.
        faults = FaultInjection(interrupt_after_stages=1)
        result = PlanExecutor(cluster, faults).execute(plan)

        self.assertTrue(result.interrupted)
        # In-flight task rolled back: source active, children gone.
        self.assertEqual(result.final_states["s0"], ShardState.ACTIVE)
        self.assertNotIn("s0.part0", result.final_states)
        # Second task never started; both sources still serve everything.
        self.assertEqual(cluster.serving_ids(),
                         live_ids(specs[0]) | live_ids(specs[1]))
        self.assertTrue(any("interrupted" in reason
                            for reason in result.failed_tasks["s0"]))


if __name__ == "__main__":
    unittest.main()
