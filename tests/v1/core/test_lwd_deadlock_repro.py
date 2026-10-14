# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reproduction of the tail-of-load notify deadlock (issue #1, 10-11).

Replays the scheduler-decision state observed in the field log. The head
notify used to be held by preflight forever (R mid-prompt, decode_ready=0,
free never changing) because preflight compared the real multi-group need
against a free reading that ignored the blocks allocate_slots reclaims
first. The reclaim fix (_lwd_reclaim_hidden_prefill_blocks, called at the
top of _schedule_impl) removes the false hold for the field state; this
file now proves both sides:

  - k=2910 (field): reclaim frees 490 -> free 3634->4124 >= need 3782,
    the batch dispatches within a bounded number of steps;
  - k=607: reclaim frees only 96 < the structural gap 148 -> hold
    persists. This is a TRUE hold (the real allocator cannot fit the
    batch either); it stays until the capacity guarantee (edge-side
    block-budget batching / larger pool) lands, and this test is the
    reminder: it must flip when that lands.

Field numbers mirrored here (see issue #1 comment "v2 table"):
  pool 4143 (4142 allocatable), head notify seqno=127 = [R's continuation
  (R computed=2910/3504, 594 tokens) + 5 fresh 3504 items + 1 truncated
  2304 item], admission need = 3782, gap = 148 (structural constant,
  independent of k), decode_ready = 0.

What this covers: the scheduler decision loop (queue/reclaim/preflight/
decode gate). What it does NOT cover: tensor transport and real allocator
internals -- those need the NPU environment.
"""

import ast
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS


def cdiv(a, b):
    return -(-a // b)


SCHED_PATH = (
    Path(__file__).resolve().parents[3] / "vllm/v1/lwd_control/control_cloud_scheduler/"
    "lwd_cloud_mixed_scheduler.py"
)

# DSV4 six-group density table (tokens per block), from the field
# blocks_to_alloc log line (7+1+28+28+438+110 = 612 for 3504 tokens).
GROUP_BLOCK_SIZES = (512, 4096, 128, 128, 8, 32)

# Sliding window per group (tokens). g0/g1 keep full history (no reclaim);
# g2/g3 SWA and the c128 compressor state (g5) use W=128, the c4 compressor
# state (g4) uses W=8 (compressor.py: coff*ratio). The 32k NPU run
# reproduced the derived caps 162/2562/645 exactly.
GROUP_WINDOWS = (None, None, 128, 128, 8, 128)

WANTED_METHODS = (
    "_schedule_impl",
    "_lwd_is_decode",
    "_lwd_collect_decode_requests",
    "_lwd_classify_items",
    "_lwd_notify_items",
    "_lwd_preflight_ok",
    "_lwd_item_block_cost",
    "_lwd_decode_blocks_per_req",
    "_lwd_reclaim_hidden_prefill_blocks",
)


def load_methods():
    tree = ast.parse(SCHED_PATH.read_text(encoding="utf-8"))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LwdCloudMixedScheduler"
    )
    methods = [
        node
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef) and node.name in WANTED_METHODS
    ]
    assert {m.name for m in methods} == set(WANTED_METHODS), (
        "missing scheduler methods under test"
    )
    header = ast.parse("from __future__ import annotations")
    header.body.extend(methods)
    namespace = {"cdiv": cdiv, "logger": NS(info=lambda *a, **k: None)}
    exec(compile(header, str(SCHED_PATH), "exec"), namespace)
    return namespace


class Sched:
    """Fake scheduler binding the extracted methods (no vllm needed)."""


_ns = load_methods()
for _name in WANTED_METHODS:
    setattr(Sched, _name, _ns[_name])


class FakeBlockPool:
    def __init__(self, free):
        self.free = free

    def get_num_free_blocks(self):
        return self.free


class FakeCoordinator:
    """Stateful density-table coordinator: subtracts blocks already held
    for partially computed requests (mirrors manager behavior). The
    subtraction keys off the BLOCK TABLE LENGTH, which reclaim does not
    change (freed blocks become null but stay in the table) -- so this
    class intentionally ignores reclaim state, exactly like production."""

    def __init__(self, computed_of):
        self.computed_of = computed_of
        self.single_type_managers = [NS(block_size=bs) for bs in GROUP_BLOCK_SIZES]

    def get_num_blocks_to_allocate(
        self,
        request_id,
        num_tokens,
        new_computed_blocks,
        num_encoder_tokens,
        total_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=False,
    ):
        held_tokens = self.computed_of.get(request_id, 0)
        return sum(
            max(
                cdiv(num_tokens, m.block_size) - cdiv(held_tokens, m.block_size),
                0,
            )
            for m in self.single_type_managers
        )


class FakeReclaimModel:
    """Faithful port of the production reclaim semantics
    (SingleTypeKVCacheManager.remove_skipped_blocks):
      skipped = max(0, computed - W + 1); blocks to null = skipped // bs,
      capped at the table length; the downward walk stops at the first
      already-null block, so repeated calls free nothing (idempotent).
    Full-history groups (W is None) never skip (base class returns 0).
    Only free rises; need-side state is untouched (see FakeCoordinator).
    """

    def __init__(self, pool, computed_of):
        self.pool = pool
        # rid -> per-group [table_length, nulled_prefix]
        self.tables = {
            rid: [[cdiv(c, bs), 0] for bs in GROUP_BLOCK_SIZES]
            for rid, c in computed_of.items()
        }
        self.calls = []

    def remove_skipped_blocks(self, request_id, total_computed_tokens):
        self.calls.append((request_id, total_computed_tokens))
        table = self.tables.get(request_id)
        if table is None:
            raise KeyError(request_id)  # production would too
        for gi, (bs, win) in enumerate(zip(GROUP_BLOCK_SIZES, GROUP_WINDOWS)):
            if win is None:
                continue
            length, nulled = table[gi]
            skipped = max(0, total_computed_tokens - win + 1)
            want = min(skipped // bs, length)
            freed = max(0, want - nulled)
            if freed:
                table[gi][1] = nulled + freed
                self.pool.free += freed


def held_blocks(computed):
    return sum(cdiv(computed, bs) for bs in GROUP_BLOCK_SIZES)


def item(rid, num_tokens, offset=0):
    return NS(
        request_id=rid,
        num_tokens=num_tokens,
        offset=offset,
        has_mrope=False,
        mrope_positions=b"",
    )


def make_deadlock_sched(computed=2910):
    """Rebuild the 10-11 field state: R mid-prompt (first chunk =
    `computed` tokens) + head notify 127 (R's continuation + 5 fresh
    3504 + 1 truncated 2304). free = 4142 - held(computed)."""
    r_id = "chatcmpl-97272205e306db6b7-958a475c"
    computed_of = {r_id: computed}
    sched = Sched()
    pool = FakeBlockPool(4142 - held_blocks(computed))
    reclaim = FakeReclaimModel(pool, computed_of)
    sched.kv_cache_manager = NS(
        block_pool=pool,
        coordinator=FakeCoordinator(computed_of),
        empty_kv_cache_blocks=NS(blocks=((),) * len(GROUP_BLOCK_SIZES)),
        remove_skipped_blocks=reclaim.remove_skipped_blocks,
    )
    sched.reclaim = reclaim
    sched._coordinator_computed = computed_of

    def req(rid, prompt, comp):
        return NS(
            request_id=rid,
            num_prompt_tokens=prompt,
            num_computed_tokens=comp,
            num_tokens=prompt,  # no output tokens yet
        )

    sched.r_req = req(r_id, 3504, computed)
    sched.requests = {r_id: sched.r_req}
    # R is the only running request and is NOT decode-eligible
    # (computed < 3504) -> decode_ready stays 0.
    sched.running = [sched.r_req]
    sched.waiting = []
    sched.skipped_waiting = []
    sched._lwd_spec_factor = 2
    sched.max_num_scheduled_tokens = 20480
    sched.max_num_running_reqs = 128
    sched.max_model_len = 131072

    # seqno=127: R's continuation + 5 fresh 3504 + truncated 2304.
    fresh = [f"fresh-{i}" for i in range(5)]
    for rid in fresh:
        sched.requests[rid] = req(rid, 3504, 0)
    trunc = "fresh-trunc"
    sched.requests[trunc] = req(trunc, 3504, 0)
    sched.notify = NS(
        seqno=127,
        items=[
            item(r_id, 3504 - computed, offset=computed),
            *[item(rid, 3504) for rid in fresh],
            item(trunc, 2304),
        ],
        request_id=r_id,
        num_tokens=3504 - computed,
    )
    sched.prefill_notify_queue = deque([sched.notify])

    # Protocol/tensor layers are out of scope: stub them.
    sched._lwd_pre_recv_ready = lambda notify: True
    sched.mixed_calls = []
    sched.decode_steps = []

    def _fake_mixed(notify, items, decode_ids):
        sched.mixed_calls.append(notify.seqno)
        queue = sched.prefill_notify_queue
        if queue and queue[0] is notify:
            queue.popleft()
        # Emulate success: R finishes its prompt.
        sched.r_req.num_computed_tokens = sched.r_req.num_prompt_tokens
        return NS(mixed=True)

    def _fake_visible(decode_ids):
        sched.decode_steps.append(list(decode_ids))
        return NS(mixed=False)

    sched._schedule_mixed = _fake_mixed
    sched._lwd_schedule_for_visible_reqs = _fake_visible
    return sched


class TestReclaimFix(unittest.TestCase):
    """k=2910 (field state): the early reclaim must break the deadlock."""

    STEPS = 200

    def test_k2910_reclaim_unlocks_deadlock(self):
        """REGRESSION (was red test_deadlock_must_be_resolved): with the
        field state, reclaim frees 490 blocks (free 3634->4124 >= 3782)
        and the head notify dispatches on the first step."""
        sched = make_deadlock_sched(computed=2910)
        for _ in range(self.STEPS):
            out = sched._schedule_impl()
            if sched.mixed_calls:
                break
            self.assertFalse(getattr(out, "mixed", False))
        self.assertEqual(sched.mixed_calls, [127], "batch must dispatch")
        self.assertEqual(sched.prefill_notify_queue, deque())
        self.assertEqual(sched.r_req.num_computed_tokens, 3504)
        # free trajectory: 3634 -reclaim-> 4124, then the mixed step runs.
        self.assertEqual(sched.kv_cache_manager.block_pool.free, 4124)

    def test_k2910_reclaim_frees_exactly_490(self):
        """The faithful reclaim model must reproduce the W=128 field
        arithmetic: 21+21+362+86 = 490 (g2/g3/g4/g5)."""
        sched = make_deadlock_sched(computed=2910)
        sched._lwd_reclaim_hidden_prefill_blocks()
        self.assertEqual(sched.kv_cache_manager.block_pool.free, 3634 + 490)
        # Window-independent lower bound (g4+g5 only): 362+86 = 448.
        self.assertGreaterEqual(sched.kv_cache_manager.block_pool.free - 3634, 448)

    def test_reclaim_is_idempotent(self):
        """Second call frees nothing (production walks stop at null
        prefixes) -- no double counting across decode-only steps."""
        sched = make_deadlock_sched(computed=2910)
        sched._lwd_reclaim_hidden_prefill_blocks()
        free_after_first = sched.kv_cache_manager.block_pool.free
        sched._lwd_reclaim_hidden_prefill_blocks()
        sched._lwd_reclaim_hidden_prefill_blocks()
        self.assertEqual(sched.kv_cache_manager.block_pool.free, free_after_first)
        self.assertEqual(len(sched.reclaim.calls), 3)

    def test_reclaim_does_not_change_need(self):
        """The need side must be reclaim-invariant: the cost predicates
        subtract by block-table LENGTH (nulls still count), so preflight
        need is identical before and after the early reclaim. NPU field
        evidence: need=2137 both pre-reclaim (fixed branch preflight)
        and post-reclaim (real chunk-alloc)."""
        sched = make_deadlock_sched(computed=2910)
        items = sched._lwd_notify_items(sched.notify)
        before = sum(sched._lwd_item_block_cost(i) for i in items)
        sched._lwd_reclaim_hidden_prefill_blocks()
        after = sum(sched._lwd_item_block_cost(i) for i in items)
        self.assertEqual(before, 3782)
        self.assertEqual(after, before)

    def test_need_3782_and_gap_148_reproduced(self):
        """The fake coordinator must reproduce the field need exactly,
        and the gap 148 is the structural constant (6x613+612-4142),
        independent of k."""
        for computed in (2910, 607):
            sched = make_deadlock_sched(computed=computed)
            items = sched._lwd_notify_items(sched.notify)
            need = sum(sched._lwd_item_block_cost(i) for i in items)
            self.assertEqual(
                need - sched.kv_cache_manager.block_pool.get_num_free_blocks(),
                148,
                f"gap must stay 148 for k={computed}",
            )
            if computed == 2910:
                self.assertEqual(need, 3782)


class TestTrueHoldRemains(unittest.TestCase):
    """k=607: reclaim alone cannot (and must not) dispatch this batch --
    the real allocator would fail it too. Stays until the capacity
    guarantee (edge-side block-budget batching / larger pool) lands;
    flip this test when that lands."""

    STEPS = 200

    def test_k607_true_hold_persists(self):
        sched = make_deadlock_sched(computed=607)
        for _ in range(self.STEPS):
            sched._schedule_impl()
        self.assertEqual(sched.mixed_calls, [], "k=607 is a true hold: no dispatch")
        self.assertIs(sched.prefill_notify_queue[0], sched.notify)
        # Reclaim fired once (freed 96: 3+3+75+15), then free is stable.
        self.assertEqual(
            sched.kv_cache_manager.block_pool.free,
            (4142 - held_blocks(607)) + 96,
        )
        self.assertEqual(sched.r_req.num_computed_tokens, 607)
        # 148 - 96 = 52 blocks still missing: capacity guarantee territory.

    def test_k607_reclaim_frees_only_96(self):
        """Short first chunk -> little outside the windows: 3+3+75+15
        = 96 < the structural gap 148."""
        sched = make_deadlock_sched(computed=607)
        sched._lwd_reclaim_hidden_prefill_blocks()
        self.assertEqual(
            sched.kv_cache_manager.block_pool.free - (4142 - held_blocks(607)),
            96,
        )


class TestCounterfactuals(unittest.TestCase):
    def test_one_fewer_full_item_unlocks(self):
        """Dropping one fresh 3504 item (-613 blocks) admits the batch
        even without reclaim: the deadlock margin was exactly one full
        first-chunk request."""
        sched = make_deadlock_sched(computed=2910)
        items = sched.notify.items
        # keep [R's continuation, fresh0..3, truncated]: drop fresh item 5
        sched.notify.items = items[:5] + items[6:]
        # need = 105 + 4x613 + 612 = 3169 <= 3634 -> admits.
        sched._schedule_impl()
        self.assertEqual(sched.mixed_calls, [127])
        self.assertEqual(sched.prefill_notify_queue, deque())
        self.assertEqual(sched.r_req.num_computed_tokens, 3504)

    def test_freed_pool_unlocks(self):
        """With free back at 4143 (R's blocks released / richer pool),
        the same batch admits immediately."""
        sched = make_deadlock_sched(computed=2910)
        sched.kv_cache_manager.block_pool.free = 4143
        sched._schedule_impl()
        self.assertEqual(sched.mixed_calls, [127])

    def test_decode_ready_is_zero_in_deadlock_state(self):
        """R is the only running request and is mid-prompt: no decode
        drain path exists (the second pillar of the deadlock)."""
        sched = make_deadlock_sched(computed=2910)
        self.assertEqual(sched._lwd_collect_decode_requests(), [])


if __name__ == "__main__":
    unittest.main()
