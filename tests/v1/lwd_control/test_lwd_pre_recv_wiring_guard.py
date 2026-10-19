# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source guard: the pre-recv sideband MQ wiring stays in place.

Issue #18 breakpoint 5 (2026-10-19 hang): without the wiring, the receive
watermark never advanced, the pre-recv gate never opened and the cloud
spun empty steps forever while the edge waited for the DOWN tensor.

The non-invasive split being pinned here:
- upstream executor file carries only three additive pass-through hunks:
  the WorkerProcHandle.irecv_done_mq field, the READY payload's
  "irecv_done_handle" key (read off the inner worker), and the
  wait_for_ready post-hoc reader attach;
- everything else lives in our package: the engine creates the hint MQ in
  __init__ BEFORE super() spawns workers (the handle rides the
  environment), registers the sender against its own MQ, and collects the
  done MQ from the executor's public worker handles.

AST source-scan form, no torch required.
"""

from __future__ import annotations

import ast
from pathlib import Path

VLLM_ROOT = Path(__file__).resolve().parents[3]
EXECUTOR = VLLM_ROOT / "vllm" / "v1" / "executor" / "multiproc_executor.py"
ENGINE = (
    VLLM_ROOT
    / "vllm"
    / "v1"
    / "lwd_control"
    / "control_cloud_scheduler"
    / "lwd_cloud_engine.py"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_executor_carries_the_done_mq_pass_through() -> None:
    tree = ast.parse(_source(EXECUTOR))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "WorkerProcHandle":
            anns = {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
            }
            assert "irecv_done_mq" in anns, (
                "WorkerProcHandle lost irecv_done_mq: the engine cannot "
                "collect the done MQ and the pre-recv gate never opens"
            )
            break
    src = _source(EXECUTOR)
    assert '"irecv_done_handle"' in src, (
        "worker_main READY payload no longer exports irecv_done_handle"
    )
    assert "worker.worker.irecv_done_mq.export_handle()" in src, (
        "the READY export must read the inner (Ascend) worker directly, "
        "keeping WorkerProc free of LWD knowledge"
    )
    assert 'response.get("irecv_done_handle")' in src, (
        "wait_for_ready no longer attaches the done-MQ reader post-hoc"
    )


def test_executor_stays_logic_free() -> None:
    """The LWD orchestration must stay in our package, not the executor."""
    src = _source(EXECUTOR)
    assert "create_hint_mq" not in src, (
        "hint MQ creation moved back into the executor; it belongs to the "
        "engine (before super() spawns workers)"
    )
    assert "lwd_recv_hint_mq" not in src and "lwd_irecv_done_mq = " not in src, (
        "executor-owned LWD MQ attributes crept back in"
    )


def test_engine_owns_hint_and_done_mqs() -> None:
    src = _source(ENGINE)
    assert "self._lwd_hint_mq = create_hint_mq()" in src, (
        "the engine must create the hint MQ in __init__"
    )
    super_pos = src.index("super().__init__(")
    hint_pos = src.index("self._lwd_hint_mq = create_hint_mq()")
    assert hint_pos < super_pos, (
        "the hint MQ must be created BEFORE super().__init__: the handle "
        "rides the environment and only workers spawned afterwards see it"
    )
    assert "handle.irecv_done_mq" in src and "model_executor.workers" in src, (
        "the engine must collect the done MQ from the executor's public "
        "worker handles"
    )
    assert 'getattr(self, "_lwd_done_mq", None)' in src, (
        "the drain must read the engine-owned done MQ"
    )
