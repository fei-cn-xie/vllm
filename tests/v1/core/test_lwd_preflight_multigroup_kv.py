# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for issue #1: preflight KV 估算必须是多组真实口径。

第一类(seqno=446,1010_lwd_repro_clean.log):
  DSV4-Flash 六组异构 KV cache(组 block_size 512/4096/128/128/8/32,
  由日志 blocks_to_alloc 反推:3504 token -> 7+1+28+28+438+110=612 块,
  uncapped need_slot 口径 613)。旧单组公式 cdiv(num_tokens, 128) 只估
  28 块,偏低 21.9 倍:free=1731 时 3 个 prefill + 116 decode 被假放行
  (200<=1731),第 3 个 item 在 full_sequence_must_fit 中途被拒 ->
  batch admission mismatch 崩溃。真实需求 3*613+116*6=2535>1731。

第二类(seqno=129,1011 真机验证日志):
  边侧组批预算填尾截断的 chunk(3504 prompt / 1008 notify)——
  full_sequence_must_fit 的准入计量单位是"全量剩余序列",按 chunk
  估算只算 178 块,实为 612 块。修复:两道谓词(full-fit 带 cap /
  chunk 扣减无 cap)取最大。
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS


def cdiv(a, b):
    return -(-a // b)


SCHED_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm/v1/lwd_control/control_cloud_scheduler/"
    "lwd_cloud_mixed_scheduler.py"
)

# DSV4-Flash 六组 block_size(与复现日志逐位对齐)
DSV4_GROUP_BLOCK_SIZES = (512, 4096, 128, 128, 8, 32)

WANTED_METHODS = (
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
        "缺少被测方法(修复需引入 _lwd_item_block_cost/"
        "_lwd_decode_blocks_per_req)"
    )
    # 补 future import 使注解惰性求值,免依赖模块级名字
    header = ast.parse("from __future__ import annotations")
    header.body.extend(methods)
    namespace = {"cdiv": cdiv, "logger": NS(info=lambda *a, **k: None)}
    exec(compile(header, str(SCHED_PATH), "exec"), namespace)
    return namespace


class Sched:
    """绑定被测方法的假调度器(不依赖 vllm 安装)。"""


for _name in WANTED_METHODS:
    setattr(Sched, _name, load_methods()[_name])


class FakeCoordinator:
    """多组求和假 coordinator:逐组 cdiv(num_tokens, 组 block_size),
    按 total_computed 近似扣除已持有块(无 SWA 窗口扣减,精度足够)。"""

    def __init__(self, block_sizes):
        self.single_type_managers = [NS(block_size=bs) for bs in block_sizes]
        self.calls = []

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
        self.calls.append(
            (num_tokens, total_computed_tokens, num_tokens_main_model,
             apply_admission_cap)
        )
        return sum(
            max(
                cdiv(num_tokens, m.block_size)
                - cdiv(total_computed_tokens, m.block_size),
                0,
            )
            for m in self.single_type_managers
        )


def make_sched(group_sizes, free_blocks):
    sched = Sched()
    sched.kv_cache_manager = NS(
        block_pool=NS(get_num_free_blocks=lambda: free_blocks),
        coordinator=FakeCoordinator(group_sizes),
        empty_kv_cache_blocks=NS(blocks=((),) * len(group_sizes)),
    )
    sched.requests = {}
    sched.running = []
    sched._lwd_spec_factor = 2
    sched.max_num_scheduled_tokens = 20480
    sched.max_num_running_reqs = 128
    sched.max_model_len = 131072
    return sched


def item(rid, num_tokens=3504):
    return NS(
        request_id=rid,
        num_tokens=num_tokens,
        offset=0,
        has_mrope=False,
        mrope_positions=b"",
    )


class TestPreflightMultiGroupKV(unittest.TestCase):
    @staticmethod
    def add_requests(sched, items, computed=0):
        for it in items:
            sched.requests[it.request_id] = NS(
                request_id=it.request_id,
                num_computed_tokens=computed,
                num_tokens=computed + it.num_tokens,
            )

    def test_issue1_dsv4_batch_must_defer(self):
        # free=1731: 旧单组公式 3*28+116*1=200 -> 假放行(崩溃路径);
        # 多组真实口径 3*613+116*6=2535 -> 必须整批推迟
        sched = make_sched(DSV4_GROUP_BLOCK_SIZES, free_blocks=1731)
        items = [item(f"r{i}") for i in range(3)]
        self.add_requests(sched, items)
        self.assertFalse(sched._lwd_preflight_ok(items, 116))

    def test_issue1_boundary_between_pass_and_defer(self):
        # 真实需求恰 2535:free=2535 放行(检查是 >),2534 推迟
        items = [item(f"r{i}") for i in range(3)]
        sched = make_sched(DSV4_GROUP_BLOCK_SIZES, free_blocks=2535)
        self.add_requests(sched, items)
        self.assertTrue(sched._lwd_preflight_ok(items, 116))
        sched = make_sched(DSV4_GROUP_BLOCK_SIZES, free_blocks=2534)
        self.add_requests(sched, items)
        self.assertFalse(sched._lwd_preflight_ok(items, 116))

    def test_single_group_degenerates_to_legacy_formula(self):
        # 单组模型(如 Qwen-MM):slot 3506 -> cdiv(3506,128)=28,
        # decode cdiv(2,128)=1,总量 3*28+116=200,与旧公式一致
        items = [item(f"r{i}") for i in range(3)]
        sched = make_sched((128,), free_blocks=200)
        self.add_requests(sched, items)
        self.assertTrue(sched._lwd_preflight_ok(items, 116))
        sched = make_sched((128,), free_blocks=199)
        self.add_requests(sched, items)
        self.assertFalse(sched._lwd_preflight_ok(items, 116))

    def test_item_cost_uses_need_slot_with_lookahead(self):
        # 续传 chunk:两道谓词——full-fit 按全量剩余序列(9504,带
        # cap),chunk 扣减按 need_slot(4506,无 cap),取大者
        sched = make_sched(DSV4_GROUP_BLOCK_SIZES, free_blocks=10**6)
        coord = sched.kv_cache_manager.coordinator
        it = item("cont", num_tokens=1000)
        sched.requests["cont"] = NS(
            request_id="cont",
            num_computed_tokens=3504,
            num_tokens=9504,
        )
        cost = sched._lwd_item_block_cost(it)
        full_call, chunk_call = coord.calls[-2], coord.calls[-1]
        self.assertEqual(full_call, (9504, 3504, 9504, True))
        self.assertEqual(chunk_call, (4506, 3504, 4504, False))
        # full-fit: Σ(cdiv(9504,bs)-cdiv(3504,bs)) = 12+2+47+47+750+187
        # chunk:    Σ(cdiv(4506,bs)-cdiv(3504,bs)) = 2+1+8+8+126+31
        self.assertEqual(cost, max(1045, 176))

    def test_issue1_truncated_chunk_charged_at_full_prompt(self):
        # issue #1 第二类:边侧预算填尾截断的 chunk——prompt=3504 而
        # notify=1008,准入按全量剩余序列计费(612),不是按 chunk(178)
        sched = make_sched(DSV4_GROUP_BLOCK_SIZES, free_blocks=10**6)
        it = item("trunc", num_tokens=1008)
        sched.requests["trunc"] = NS(
            request_id="trunc",
            num_computed_tokens=0,
            num_tokens=3504,
        )
        self.assertEqual(sched._lwd_item_block_cost(it), 612)

    def test_issue1_mixed_batch_with_truncated_tail(self):
        # 崩溃场景缩小版:[全量3504, 全量3504, 截断1008(prompt=3504)]
        # + 116 decode:need = 613+613+612+116*6 = 2534,边界含等号
        items = [item("a"), item("b"), item("t", num_tokens=1008)]
        sched = make_sched(DSV4_GROUP_BLOCK_SIZES, free_blocks=2534)
        self.add_requests(sched, items[:2])
        sched.requests["t"] = NS(
            request_id="t", num_computed_tokens=0, num_tokens=3504
        )
        self.assertTrue(sched._lwd_preflight_ok(items, 116))
        sched = make_sched(DSV4_GROUP_BLOCK_SIZES, free_blocks=2533)
        self.add_requests(sched, items[:2])
        sched.requests["t"] = NS(
            request_id="t", num_computed_tokens=0, num_tokens=3504
        )
        self.assertFalse(sched._lwd_preflight_ok(items, 116))

    def test_decode_cost_sums_across_groups(self):
        # spec_factor=2:每组 cdiv(2, bs)=1,六组共 6(旧公式只算 1)
        sched = make_sched(DSV4_GROUP_BLOCK_SIZES, free_blocks=10**6)
        self.assertEqual(sched._lwd_decode_blocks_per_req(), 6)


if __name__ == "__main__":
    unittest.main()
