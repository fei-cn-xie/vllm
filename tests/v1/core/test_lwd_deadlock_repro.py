# SPDX-License-Identifier: Apache-2.0
"""CPU reproduction of the tail-of-load notify deadlock (issue #1, 10-11).

Replays the exact scheduler-decision state observed in the field log and
asserts the deadlock loop: the head notify is held by preflight forever,
R (mid-prompt request) never progresses, free blocks never change, and no
fail-fast fires. Counterfactual sub-tests prove the two unlock paths.

Field numbers mirrored here (see issue #1 comment "v2 table"):
  pool free = 3634, head notify seqno=127 = [R's 594-token continuation
  (R computed=2910/3504) + 5 fresh 3504 items + 1 truncated 2304 item],
  admission need = 3782 (7+1+28+28+438+110 density table reproduces it
  exactly), decode_ready = 0.

What this covers: the scheduler decision loop (queue/preflight/decode
gate). What it does NOT cover: tensor transport and real allocator
internals — those need the NPU environment.
"""

import ast
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS


def cdiv(a, b):
    return -(-a // b)


SCHED_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm/v1/lwd_control/control_cloud_scheduler/"
    "lwd_cloud_mixed_scheduler.py"
)

# DSV4 six-group density table (tokens per block), from the field
# blocks_to_alloc log line (7+1+28+28+438+110 = 612 for 3504 tokens).
GROUP_BLOCK_SIZES = (512, 4096, 128, 128, 8, 32)

WANTED_METHODS = (
    "_schedule_impl",
    "_lwd_is_decode",
    "_lwd_collect_decode_requests",
    "_lwd_classify_items",
    "_lwd_notify_items",
    "_lwd_preflight_ok",
    "_lwd_item_block_cost",
    "_lwd_decode_blocks_per_req",
)


def load_methods():
    tree = ast.parse(SCHED_PATH.read_text(encoding="utf-8"))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LwdCloudMixedScheduler"
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
    for partially computed requests (mirrors manager behavior)."""

    def __init__(self, computed_of):
        self.computed_of = computed_of
        self.single_type_managers = [
            NS(block_size=bs) for bs in GROUP_BLOCK_SIZES
        ]

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
                cdiv(num_tokens, m.block_size)
                - cdiv(held_tokens, m.block_size),
                0,
            )
            for m in self.single_type_managers
        )


def item(rid, num_tokens, offset=0):
    return NS(
        request_id=rid,
        num_tokens=num_tokens,
        offset=offset,
        has_mrope=False,
        mrope_positions=b"",
    )


def make_deadlock_sched(free=3634):
    """Rebuild the 10-11 field state: R mid-prompt + head notify 127."""
    r_id = "chatcmpl-97272205e306db6b7-958a475c"
    computed_of = {r_id: 2910}  # R's chunk1 from seqno=126
    sched = Sched()
    sched.kv_cache_manager = NS(
        block_pool=FakeBlockPool(free),
        coordinator=FakeCoordinator(computed_of),
        empty_kv_cache_blocks=NS(blocks=((),) * len(GROUP_BLOCK_SIZES)),
    )
    sched._coordinator_computed = computed_of

    def req(rid, prompt, computed):
        return NS(
            request_id=rid,
            num_prompt_tokens=prompt,
            num_computed_tokens=computed,
            num_tokens=prompt,  # no output tokens yet
        )

    sched.r_req = req(r_id, 3504, 2910)
    sched.requests = {r_id: sched.r_req}
    # R is the only running request and is NOT decode-eligible
    # (2910 < 3504) -> decode_ready stays 0.
    sched.running = [sched.r_req]
    sched.waiting = []
    sched.skipped_waiting = []
    sched._lwd_spec_factor = 2
    sched.max_num_scheduled_tokens = 20480
    sched.max_num_running_reqs = 128
    sched.max_model_len = 131072

    # seqno=127: R's 594 continuation + 5 fresh 3504 + truncated 2304.
    fresh = [f"fresh-{i}" for i in range(5)]
    for rid in fresh:
        sched.requests[rid] = req(rid, 3504, 0)
    trunc = "fresh-trunc"
    sched.requests[trunc] = req(trunc, 3504, 0)
    sched.notify = NS(
        seqno=127,
        items=[
            item(r_id, 594, offset=2910),
            *[item(rid, 3504) for rid in fresh],
            item(trunc, 2304),
        ],
        request_id=r_id,
        num_tokens=594,
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


class TestDeadlockRepro(unittest.TestCase):
    DEADLOCK_STEPS = 200

    def test_deadlock_loop_persists(self):
        """Characterization: with the field state, the head notify is
        held forever, R never progresses, free never changes, and the
        fail-fast verify never fires (preflight defers before it)."""
        sched = make_deadlock_sched()
        for _ in range(self.DEADLOCK_STEPS):
            out = sched._schedule_impl()
            self.assertFalse(getattr(out, "mixed", False))
        self.assertEqual(sched.mixed_calls, [], "batch must never dispatch")
        self.assertEqual(len(sched.decode_steps), self.DEADLOCK_STEPS)
        self.assertTrue(all(ids == [] for ids in sched.decode_steps))
        self.assertIs(sched.prefill_notify_queue[0], sched.notify)
        self.assertEqual(sched.r_req.num_computed_tokens, 2910)
        self.assertEqual(sched.kv_cache_manager.block_pool.free, 3634)

    def test_need_3782_reproduced_from_density_table(self):
        """The fake coordinator must reproduce the field need exactly."""
        sched = make_deadlock_sched()
        items = sched._lwd_notify_items(sched.notify)
        need = sum(sched._lwd_item_block_cost(i) for i in items)
        self.assertEqual(need, 3782)
        # And it exceeds free by the field gap.
        self.assertEqual(
            need - sched.kv_cache_manager.block_pool.get_num_free_blocks(),
            148,
        )

    def test_counterfactual_one_fewer_full_item_unlocks(self):
        """Dropping one fresh 3504 item (-613 blocks) admits the batch:
        the deadlock margin was exactly one full first-chunk request."""
        sched = make_deadlock_sched()
        items = sched.notify.items
        # keep [R's continuation, fresh0..3, truncated]: drop fresh item 5
        sched.notify.items = items[:5] + items[6:]
        # need = 105 + 4x613 + 612 = 3169 <= 3634 -> admits.
        sched._schedule_impl()
        self.assertEqual(sched.mixed_calls, [127])
        self.assertEqual(sched.prefill_notify_queue, deque())
        self.assertEqual(sched.r_req.num_computed_tokens, 3504)

    def test_counterfactual_freed_pool_unlocks(self):
        """With free back at 4143 (R's blocks released / richer pool),
        the same batch admits immediately."""
        sched = make_deadlock_sched()
        sched.kv_cache_manager.block_pool.free = 4143
        sched._schedule_impl()
        self.assertEqual(sched.mixed_calls, [127])

    def test_decode_ready_is_zero_in_deadlock_state(self):
        """R is the only running request and is mid-prompt: no decode
        drain path exists (the second pillar of the deadlock)."""
        sched = make_deadlock_sched()
        self.assertEqual(sched._lwd_collect_decode_requests(), [])


if __name__ == "__main__":
    unittest.main()
