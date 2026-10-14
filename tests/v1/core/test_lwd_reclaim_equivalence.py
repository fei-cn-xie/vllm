# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence lock for the early reclaim (issue #1, root cause 3).

The cloud scheduler now calls remove_skipped_blocks for hidden mid-prefill
requests at the top of _schedule_impl, with the exact argument
allocate_slots would later pass for the same running request. This file
drives the REAL production reclaim code (AST-extracted from
single_type_kv_cache_manager.py; this Windows host has no torch, so real
imports are impossible -- same pattern as test_lwd_deadlock_repro.py) and
locks three properties:

1. Equivalence: "early reclaim (scheduler's own call) + native-time
   reclaim" leaves the same block tables and free counter as "native-time
   reclaim alone", and the native-time reclaim becomes a no-op.
2. Argument lock: the scheduler passes exactly
   min(num_computed_tokens, max_model_len) -- the CURRENT watermark.
3. Failure mode of the wrong argument (computed + next chunk): it nulls
   blocks inside the window the next chunk's early queries still read
   (tokens [computed-W+1, computed)), which another request could then
   grab -- silent corruption. Documents why the lock exists.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS


def cdiv(a, b):
    return -(-a // b)


REPO = Path(__file__).resolve().parents[3]
SCHED_PATH = (
    REPO / "vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_mixed_scheduler.py"
)
STM_PATH = REPO / "vllm/v1/core/single_type_kv_cache_manager.py"


def extract_functions(path, class_name, names):
    """Pull verbatim methods out of a production class (no vllm import)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    funcs = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    missing = set(names) - {f.name for f in funcs}
    assert not missing, f"missing methods: {class_name}.{sorted(missing)}"
    header = ast.parse("from __future__ import annotations")
    header.body.extend(funcs)
    namespace = {"logger": NS(info=lambda *a, **k: None)}
    exec(compile(header, str(path), "exec"), namespace)
    return namespace


_BASE = extract_functions(
    STM_PATH,
    "SingleTypeKVCacheManager",
    {"remove_skipped_blocks", "get_num_skipped_tokens"},
)
_SW = extract_functions(STM_PATH, "SlidingWindowManager", {"get_num_skipped_tokens"})
_SCHED = extract_functions(
    SCHED_PATH,
    "LwdCloudMixedScheduler",
    {"_lwd_reclaim_hidden_prefill_blocks", "_lwd_is_decode"},
)


class FullMgr:
    """Full-history group (g0/g1-like): base semantics, never skips."""

    remove_skipped_blocks = _BASE["remove_skipped_blocks"]
    get_num_skipped_tokens = _BASE["get_num_skipped_tokens"]


class SWMgr(FullMgr):
    """Sliding-window group: the only override is get_num_skipped_tokens
    = max(0, computed - W + 1)."""

    get_num_skipped_tokens = _SW["get_num_skipped_tokens"]


class Sched:
    _lwd_reclaim_hidden_prefill_blocks = _SCHED["_lwd_reclaim_hidden_prefill_blocks"]
    _lwd_is_decode = _SCHED["_lwd_is_decode"]


class FakePool:
    def __init__(self):
        self.freed = 0

    def free_blocks(self, blocks, prepend=False):
        self.freed += len(blocks)

    def get_num_free_blocks(self):
        return self.freed


class Block:
    def __init__(self):
        self.block_hash = None  # uncached scratch block


NULL = Block()  # sentinel: identity comparison, as in production


R_ID = "R"
K = 2910  # first-chunk watermark (issue #1 field value)
NEXT_CHUNK = 594
PROMPT = K + NEXT_CHUNK  # 3504


def make_managers():
    """g4-like (bs=8, W=8) + g5-like (bs=32, W=128) + one full-history
    group, sharing one pool; tables hold K allocated tokens."""
    pool = FakePool()
    managers = []
    for bs, win in ((8, 8), (32, 128)):
        mgr = SWMgr()
        mgr.block_size = bs
        mgr.sliding_window = win
        mgr.req_to_blocks = {R_ID: [Block() for _ in range(cdiv(K, bs))]}
        mgr._null_block = NULL
        mgr.block_pool = pool
        managers.append(mgr)
    full = FullMgr()
    full.block_size = 512
    full.req_to_blocks = {R_ID: [Block() for _ in range(cdiv(K, 512))]}
    full._null_block = NULL
    full.block_pool = pool
    managers.append(full)
    return managers, pool


def null_prefix(mgr):
    blocks = mgr.req_to_blocks[R_ID]
    n = 0
    for b in blocks:
        if b is NULL:
            n += 1
        else:
            break
    return n


def snapshot(managers, pool):
    return (
        tuple(
            (m.block_size, null_prefix(m), len(m.req_to_blocks[R_ID])) for m in managers
        ),
        pool.freed,
    )


def make_sched(managers, pool):
    args_seen = []

    def remove_skipped_blocks(request_id, total_computed_tokens):
        args_seen.append((request_id, total_computed_tokens))
        for m in managers:
            m.remove_skipped_blocks(request_id, total_computed_tokens)

    sched = Sched()
    sched.running = [
        NS(
            request_id=R_ID,
            num_prompt_tokens=PROMPT,
            num_computed_tokens=K,
        )
    ]
    sched.max_model_len = 131072
    sched.kv_cache_manager = NS(
        block_pool=pool, remove_skipped_blocks=remove_skipped_blocks
    )
    return sched, args_seen


class TestEquivalence(unittest.TestCase):
    def test_scheduler_reclaim_equals_native_reclaim(self):
        """Early reclaim (scheduler) + native-time reclaim must leave
        exactly the same tables/free as native-time reclaim alone, with
        the native-time pass a complete no-op."""
        managers_a, pool_a = make_managers()
        managers_b, pool_b = make_managers()
        sched, _ = make_sched(managers_a, pool_a)

        # Path A: scheduler's early reclaim, then the reclaim that
        # allocate_slots would do for the running request's next chunk.
        sched._lwd_reclaim_hidden_prefill_blocks()
        freed_after_early = pool_a.freed
        for m in managers_a:
            m.remove_skipped_blocks(R_ID, K)

        # Path B: native-time reclaim only.
        for m in managers_b:
            m.remove_skipped_blocks(R_ID, K)

        self.assertEqual(
            pool_a.freed,
            freed_after_early,
            "native-time reclaim must be a no-op after the early one",
        )
        self.assertEqual(snapshot(managers_a, pool_a), snapshot(managers_b, pool_b))

    def test_scheduler_passes_exactly_current_watermark(self):
        """The argument must be min(num_computed_tokens, max_model_len)
        -- the CURRENT watermark, never computed + next chunk."""
        managers, pool = make_managers()
        sched, args_seen = make_sched(managers, pool)
        sched._lwd_reclaim_hidden_prefill_blocks()
        self.assertEqual(args_seen, [(R_ID, K)])

    def test_full_history_group_never_reclaims(self):
        managers, pool = make_managers()
        sched, _ = make_sched(managers, pool)
        sched._lwd_reclaim_hidden_prefill_blocks()
        full = managers[-1]
        self.assertEqual(null_prefix(full), 0)
        self.assertEqual(len(full.req_to_blocks[R_ID]), cdiv(K, 512))


class TestWrongArgument(unittest.TestCase):
    def test_computed_plus_chunk_nulls_live_window_blocks(self):
        """The dangerous argument (computed + next chunk) nulls blocks
        the next chunk's earliest queries still read (window
        [computed-W+1, computed)) -- g4 (bs=8, W=8) demonstrates it."""
        correct, pool_c = make_managers()
        wrong, pool_w = make_managers()
        for m in correct:
            m.remove_skipped_blocks(R_ID, K)
        for m in wrong:
            m.remove_skipped_blocks(R_ID, K + NEXT_CHUNK)

        g4_correct, g4_wrong = correct[0], wrong[0]
        kept_from = null_prefix(g4_correct)
        nulled_wrong = null_prefix(g4_wrong)
        # Correct: skip (2910-8+1)//8 = 362 of 364 blocks, keep the 2
        # blocks covering tokens 2896..2909 (window + alignment).
        self.assertEqual(kept_from, 362)
        # Wrong: wants to skip (3504-8+1)//8 = 437 blocks, capped at the
        # table length -> the ENTIRE table (364) is nulled, including
        # every window block the next chunk's early queries read.
        self.assertEqual(nulled_wrong, len(g4_wrong.req_to_blocks[R_ID]))
        self.assertGreater(nulled_wrong, kept_from)
        self.assertTrue(
            g4_wrong.req_to_blocks[R_ID][kept_from] is NULL,
            "wrong arg must be shown to destroy a kept-window block",
        )
        self.assertFalse(g4_correct.req_to_blocks[R_ID][kept_from] is NULL)
        self.assertGreater(pool_w.freed, pool_c.freed)


if __name__ == "__main__":
    unittest.main()
