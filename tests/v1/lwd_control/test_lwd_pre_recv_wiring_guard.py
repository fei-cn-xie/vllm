# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source guard: the executor-side pre-recv MQ wiring stays in place.

Issue #18 breakpoint 5 (2026-10-19 hang): the engine and the Ascend cloud
worker were fully wired for pre-recv, but the executor never created the
hint MQ nor attached the done MQ, so the receive watermark never advanced,
the pre-fill gate never opened and the cloud spun empty steps forever.
The engine contract is getattr(executor, "lwd_recv_hint_mq" /
"lwd_irecv_done_mq"). This guard pins the four executor wiring points.
AST source-scan form, no torch required.
"""

from __future__ import annotations

import ast
from pathlib import Path

EXECUTOR = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "v1"
    / "executor"
    / "multiproc_executor.py"
)


def _source() -> str:
    return EXECUTOR.read_text(encoding="utf-8")


def test_worker_proc_handle_carries_irecv_done_mq() -> None:
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "WorkerProcHandle":
            anns = {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
            }
            assert "irecv_done_mq" in anns, (
                "WorkerProcHandle lost irecv_done_mq: the engine's done-MQ "
                "drain (lwd_irecv_done_mq) would silently stay None and the "
                "pre-recv gate would never open"
            )
            return
    raise AssertionError("WorkerProcHandle not found")


def test_ready_payload_exports_irecv_done_handle() -> None:
    src = _source()
    assert '"irecv_done_handle"' in src, (
        "worker_main READY payload no longer exports irecv_done_handle; "
        "the executor cannot attach the done-MQ reader"
    )
    assert 'handles.get("irecv_done_handle")' in src, (
        "wait_for_response_handle_ready no longer reconstructs the done MQ"
    )


def test_executor_creates_hint_mq_for_cloud_role_only() -> None:
    src = _source()
    assert "create_hint_mq()" in src, (
        "executor no longer creates the pre-recv hint MQ; hints have no "
        "sideband channel and the gate never opens"
    )
    assert 'lwd_config.role == "cloud"' in src, (
        "hint MQ creation must stay gated on the cloud role (mm parity)"
    )
    assert "self.lwd_irecv_done_mq = next(" in src, (
        "executor no longer attaches lwd_irecv_done_mq after wait_for_ready"
    )
