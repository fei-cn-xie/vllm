# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""边侧 C2E 行交付裁定测试(移植自 mm 分支 7ff2ddc3e pad-and-skip)。

修复前形态:token 行缺席且未完结一律 ERROR 杀请求——混排步的零接受
活行(prefill 请求在采样布局只占预留槽,收集器计得 accepted=0、未完结)
是协议伪影,不是 unembed 失败,被误杀即丢请求。

_lwd_row_disposition 是纯函数(staticmethod,零依赖),用 AST 抽取源码
exec 执行做表驱动行为测试(任何主机可跑);交付循环的接线(裁定先行、
skip 不打调试税、error 分支保留 eos-trace B 点)用源码扫描守卫。
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEDULER = (
    REPO_ROOT / "vllm" / "v1" / "lwd_control" / "control_edge_scheduler"
    / "lwd_edge_scheduler.py"
)


def _load_disposition():
    """AST 抽取 _lwd_row_disposition 源码独立执行(绕开模块级 torch 依赖)。"""
    tree = ast.parse(SCHEDULER.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_lwd_row_disposition"
    )
    src = ast.get_source_segment(SCHEDULER.read_text(encoding="utf-8"), fn)
    # 剥掉 staticmethod 装饰器语义:直接以普通函数执行
    ns: dict = {}
    exec("from __future__ import annotations\n" + src, ns)
    return ns["_lwd_row_disposition"]


# (accepted, has_tokens, finished, want, 说明)
CASES = [
    (0, False, False, "skip", "零接受活行:混排伪影,跳过不杀(修复前=ERROR)"),
    (0, False, True, "deliver", "零接受收尾行:终结信号必须透传,吞掉丢完结"),
    (1, False, False, "error", "声明接受却无 token:unembed 真失败(原语义)"),
    (3, True, False, "deliver", "正常 token 行"),
    (2, True, True, "deliver", "正常完结行"),
    (0, True, False, "skip", "accepted=0 且带 token:两条线均不可达"
                             "(accepted 按 token 有无计),行为与 mm 同构"),
]


def test_row_disposition_matrix():
    disposition = _load_disposition()
    for accepted, has_tokens, finished, want, note in CASES:
        got = disposition(accepted, has_tokens, finished)
        assert got == want, f"({accepted},{has_tokens},{finished}): {got} != {want} — {note}"


def test_delivery_loop_gated_by_disposition():
    """交付循环必须先裁定再行动:skip 不交付不打调试税,error 才走 B 点。"""
    src = SCHEDULER.read_text(encoding="utf-8")
    assert "def _lwd_row_disposition(" in src
    loop = src.index("def _lwd_deliver_unembed(")
    body = src[loop:src.index("def _lwd_finish_code(", loop)]
    assert "disposition = self._lwd_row_disposition(" in body
    # skip 分支在调试打点之前(零接受活行不打调试税)
    assert body.index('if disposition == "skip":') < body.index(
        "LwdControlDiag.edge_tokens_delivered"
    )
    # error 分支保留 eos-trace B 点与 ERROR 语义
    assert body.index('if disposition == "error":') < body.index(
        "trace_edge_kill_row"
    )
    assert "FinishReason.ERROR" in body
    # 修复前的无差别杀行条件不得回归
    assert "if not sampled_token_ids and not finished_now:" not in body


def test_module_still_compiles():
    """语法完整性(py_compile 级,不触发重依赖执行)。"""
    import py_compile

    py_compile.compile(str(SCHEDULER), doraise=True)


# 轻量跑器支持:仓级 conftest 依赖 torch 时手工执行本文件用例
if __name__ == "__main__":  # pragma: no cover
    test_row_disposition_matrix()
    test_delivery_loop_gated_by_disposition()
    test_module_still_compiles()
    print("ALL ROW-DISPOSITION CHECKS PASSED")
