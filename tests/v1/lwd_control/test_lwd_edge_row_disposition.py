# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Edge-side C2E row delivery disposition tests (ported from mm branch
7ff2ddc3e pad-and-skip).

Pre-fix shape: a token row that was absent and not finished was
unconditionally ERROR-killed — but the zero-accepted live row of a mixed
step (a prefill request only occupies a reserved slot in the sampling
layout, so the collector counts accepted=0 and not finished) is a
protocol artifact, not an unembed failure; killing it wrongly dropped
the request.

_lwd_row_disposition is a pure function (staticmethod, zero
dependencies); its source is extracted via AST and exec'd for
table-driven behavior tests (runnable on any host); the delivery-loop
wiring (disposition first, skip pays no debug tax, error branch keeps
the eos-trace B point) is guarded by source scanning.
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
    """Extract the _lwd_row_disposition source via AST and execute it standalone (bypasses module-level torch dependency)."""
    tree = ast.parse(SCHEDULER.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_lwd_row_disposition"
    )
    src = ast.get_source_segment(SCHEDULER.read_text(encoding="utf-8"), fn)
    # Strip the staticmethod decorator semantics: execute directly as a plain function
    ns: dict = {}
    exec("from __future__ import annotations\n" + src, ns)
    return ns["_lwd_row_disposition"]


# (accepted, has_tokens, finished, want, note)
CASES = [
    (0, False, False, "skip", "zero-accepted live row: mixed-step artifact, skip instead of kill (pre-fix=ERROR)"),
    (0, False, True, "deliver", "zero-accepted finishing row: the finish signal must pass through; swallowing it loses the finish"),
    (1, False, False, "error", "claims accepted but no token: real unembed failure (original semantics)"),
    (3, True, False, "deliver", "normal token row"),
    (2, True, True, "deliver", "normal finished row"),
    (0, True, False, "skip", "accepted=0 yet carries tokens: unreachable on both paths"
                             " (accepted is counted by token presence), behavior is isomorphic to mm"),
]


def test_row_disposition_matrix():
    disposition = _load_disposition()
    for accepted, has_tokens, finished, want, note in CASES:
        got = disposition(accepted, has_tokens, finished)
        assert got == want, f"({accepted},{has_tokens},{finished}): {got} != {want} — {note}"


def test_delivery_loop_gated_by_disposition():
    """The delivery loop must disposition first, then act: skip neither delivers nor pays the debug tax; only error reaches the B point."""
    src = SCHEDULER.read_text(encoding="utf-8")
    assert "def _lwd_row_disposition(" in src
    loop = src.index("def _lwd_deliver_unembed(")
    body = src[loop:src.index("def _lwd_finish_code(", loop)]
    assert "disposition = self._lwd_row_disposition(" in body
    # The skip branch comes before the debug instrumentation (zero-accepted live rows pay no debug tax)
    assert body.index('if disposition == "skip":') < body.index(
        "LwdControlDiag.edge_tokens_delivered"
    )
    # The error branch keeps the eos-trace B point and ERROR semantics
    assert body.index('if disposition == "error":') < body.index(
        "trace_edge_kill_row"
    )
    assert "FinishReason.ERROR" in body
    # The pre-fix indiscriminate row-kill condition must not regress
    assert "if not sampled_token_ids and not finished_now:" not in body


def test_module_still_compiles():
    """Syntax integrity (py_compile level, no heavy-dependency execution)."""
    import py_compile

    py_compile.compile(str(SCHEDULER), doraise=True)


# Lightweight runner support: run this file's cases manually when the repo-level conftest pulls in torch
if __name__ == "__main__":  # pragma: no cover
    test_row_disposition_matrix()
    test_delivery_loop_gated_by_disposition()
    test_module_still_compiles()
    print("ALL ROW-DISPOSITION CHECKS PASSED")
