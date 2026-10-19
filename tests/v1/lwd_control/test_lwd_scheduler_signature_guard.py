# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Signature guard: LWD scheduler overrides stay call-compatible with the
native Scheduler.schedule.

Upstream step_with_batch_queue passes throttle_prefills positionally; the
2026-10-19 bring-up died with TypeError on both sides because the migrated
overrides predated that parameter. Compare the call-relevant parameter list
(names + optionality) of each LWD override against the native one, so any
future upstream signature change fails here instead of at first request.

AST source-scan form: runs on any host, no torch required.
"""

from __future__ import annotations

import ast
from pathlib import Path

VLLM_ROOT = Path(__file__).resolve().parents[3]
NATIVE = VLLM_ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py"
LWD_SCHEDULERS = {
    "LwdEdgeScheduler": VLLM_ROOT
    / "vllm"
    / "v1"
    / "lwd_control"
    / "control_edge_scheduler"
    / "lwd_edge_scheduler.py",
    "LwdCloudScheduler": VLLM_ROOT
    / "vllm"
    / "v1"
    / "lwd_control"
    / "control_cloud_scheduler"
    / "lwd_cloud_scheduler.py",
}


def _schedule_params(path: Path, class_name: str) -> list[tuple[str, bool]]:
    """(name, has_default) for the call-relevant positional params of
    <class_name>.schedule, self excluded."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "schedule":
                    args = stmt.args
                    names = [a.arg for a in [*args.posonlyargs, *args.args]][1:]
                    n_defaults = len(args.defaults)
                    return [
                        (name, idx >= len(names) - n_defaults)
                        for idx, name in enumerate(names)
                    ]
    raise AssertionError(f"{class_name}.schedule not found in {path}")


def test_lwd_schedules_match_native_signature() -> None:
    native = _schedule_params(NATIVE, "Scheduler")
    assert native, "native Scheduler.schedule params disappeared?"
    for class_name, path in LWD_SCHEDULERS.items():
        assert _schedule_params(path, class_name) == native, (
            f"{class_name}.schedule is not call-compatible with the native "
            f"Scheduler.schedule{native}: upstream passes these positionally "
            f"from step_with_batch_queue and the engine dies with TypeError "
            f"at first request"
        )
