# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source guard: the LWD edge disables prefix caching at config time.

The prefill-only edge runs no transformer layers, so it holds zero cacheable
KV groups; with prefix caching left on (the default), the KV cache
coordinator (upstream hybrid and the Ascend patch alike) asserts "requires
at least one cacheable KV cache group" during scheduler construction, as
seen on the 2026-10-19 bring-up. vllm/config/vllm.py must keep the
role=="edge" disable next to the encoder-only disable.

vllm/config/vllm.py imports torch, hence the AST source-scan form; it runs
on any host and fails the moment the block is dropped.
"""

from __future__ import annotations

import ast
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[3] / "vllm" / "config" / "vllm.py"


def _edge_disable_present() -> bool:
    tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if not ("lwd_config" in test and "edge" in test):
            continue
        for stmt in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if (
                isinstance(stmt, ast.Assign)
                and any(
                    isinstance(t, ast.Attribute) and t.attr == "enable_prefix_caching"
                    for t in stmt.targets
                )
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is False
            ):
                return True
    return False


def test_lwd_edge_disables_prefix_caching_at_config_time() -> None:
    assert _edge_disable_present(), (
        "vllm/config/vllm.py lost the LWD edge prefix-caching disable: the "
        "edge holds zero cacheable KV groups and the coordinator asserts "
        "'requires at least one cacheable KV cache group' at scheduler "
        "construction"
    )
