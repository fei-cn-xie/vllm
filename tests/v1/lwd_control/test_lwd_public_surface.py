# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public surface guard tests + intra-package dependency direction guards.

core.py reads vllm_config.lwd_config inline to select the engine class;
serve.py no longer touches lwd_control; heavyweight modules (torch-
requiring engines/schedulers) have their existence asserted via
find_spec, without triggering execution.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

try:
    from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
except Exception:  # pragma: no cover
    pytest.skip("vllm package entry not importable", allow_module_level=True)

assert callable(LwdControlDiag.token_passthrough_enabled)
assert callable(LwdControlDiag.disable_down_enabled)
assert callable(LwdControlDiag.bypass_unembed_rows)

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = REPO_ROOT / "vllm"


def _spec(name: str):
    return importlib.util.find_spec(name)


def test_module_inventory():
    """Core module existence (does not trigger torch-dependent execution)."""
    for name in [
        "vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler",
        "vllm.v1.lwd_control.control_scheduler.lwd_chunk_align",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler",
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine",
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler",
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefill_notify_queue",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_picker",
    ]:
        assert _spec(name) is not None, name


def test_deleted_modules_gone():
    """PD-disaggregation phase machine and legacy config/ledger modules are deleted and must not be importable."""
    for name in [
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_phase_scheduler",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_awaiting_tracker",
        "vllm.v1.lwd_control.lwd_config",
    ]:
        assert _spec(name) is None, name


def test_external_entry_sites():
    """core.py keeps the inline selection block; serve.py has zero lwd_control references."""
    core = (VLLM_DIR / "v1" / "engine" / "core.py").read_text()
    assert "vllm_config.lwd_config" in core
    assert "LwdEdgeEngineCore" in core and "LwdCloudEngineCore" in core
    serve = (VLLM_DIR / "entrypoints" / "cli" / "serve.py").read_text()
    assert "lwd_control" not in serve


def test_cloud_scheduler_keeps_debug_static_method():
    """Static method names referenced by lwd_debug still exist (text
    guard, torch-free); core.py has no LWD model output hooks (c2e is
    folded into scheduler update_from_output + engine post_step)."""
    source = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_cloud_scheduler"
        / "lwd_cloud_scheduler.py"
    ).read_text()
    assert "def _lwd_c2e_finish_reasons(" in source
    core = (VLLM_DIR / "v1" / "engine" / "core.py").read_text()
    assert "lwd_process_model_output" not in core
    assert "lwd_handle_model_output" not in core


def test_cloud_package_never_imports_edge_package():
    """Dependency direction guard: the cloud package must not import the edge package."""
    cloud_dir = VLLM_DIR / "v1" / "lwd_control" / "control_cloud_scheduler"
    offenders = [
        py.name
        for py in cloud_dir.glob("*.py")
        if "control_edge_scheduler" in py.read_text()
    ]
    assert offenders == []


def test_diag_package_purity():
    """Diagnostics package purity: zero functional dependencies (no
    scheduler/engine/comm imports), no execution adapters (future/stub
    output belong to the engine); a violation means diagnostics and
    functionality are re-entangled."""
    diag_dir = VLLM_DIR / "v1" / "lwd_control" / "lwd_control_diag"
    files = list(diag_dir.glob("*.py"))
    assert files, "lwd_control_diag package missing"
    banned = (
        "control_edge_scheduler",
        "control_cloud_scheduler",
        "lwd_comm",
        "control_scheduler",
        "LwdBypassFuture",
        "LwdStubModelOutput",
        "LWD_EMPTY_MODEL_OUTPUT",
    )
    for py in files:
        text = py.read_text()
        for marker in banned:
            assert marker not in text, (py.name, marker)


def test_no_scheduler_mode_strings_left():
    """PD-disaggregation discriminator strings are cleared from the package (PD_mix is the only form)."""
    for py in (VLLM_DIR / "v1" / "lwd_control").rglob("*.py"):
        text = py.read_text()
        assert "prefill_first" not in text, py
        assert "decode_first" not in text, py


# ------------------------------------------------------------------ #
# Review-loop guards (structural and interface constraints settled)  #
# ------------------------------------------------------------------ #
def test_lwd_control_uses_lwd_logger_only():
    """Unified logging interface: zero direct vllm.logger usage inside
    the lwd_control package; everything goes through vllm.lwd_logger
    (messages automatically carry the [lwd] prefix, VLLM_LWD_LOG_LEVEL
    controls the level independently); the two log sites this PR adds
    to native files (sched dbg / has_work) follow the same policy."""
    for py in (VLLM_DIR / "v1" / "lwd_control").rglob("*.py"):
        text = py.read_text()
        assert "from vllm.logger import" not in text, py
        assert "init_logger" not in text, py
    sched = (VLLM_DIR / "v1" / "core" / "sched" / "scheduler.py").read_text()
    assert "lwd_logger.info" in sched
    base_engine = (
        VLLM_DIR / "v1" / "lwd_control" / "control_scheduler" / "lwd_base_engine.py"
    ).read_text()
    assert "lwd_logger.info" in base_engine


def test_has_work_override_lives_in_base_engine():
    """has_work state-transition logging folded into the base class:
    the native core.py check stays untouched, the override lives in
    LwdBaseEngineCore (review: common logic goes in lwd_base_engine)."""
    core = (VLLM_DIR / "v1" / "engine" / "core.py").read_text()
    assert "_last_has_work_state" not in core
    base_engine = (
        VLLM_DIR / "v1" / "lwd_control" / "control_scheduler" / "lwd_base_engine.py"
    ).read_text()
    assert "def has_work" in base_engine


def test_message_dispatch_skeleton_in_base_engine():
    """Receive-thread message entry unified in the base class: the
    _lwd_on_payload entry + unknown-frame warn-and-drop live in the
    base class; both engines only implement the _lwd_dispatch_message
    hook (review: extracted into the base class together with cloud's
    dispatch)."""
    base_engine = (
        VLLM_DIR / "v1" / "lwd_control" / "control_scheduler" / "lwd_base_engine.py"
    ).read_text()
    assert "def _lwd_on_payload" in base_engine
    assert "drop unexpected frame" in base_engine
    for sub in (
        "control_edge_scheduler/lwd_edge_engine.py",
        "control_cloud_scheduler/lwd_cloud_engine.py",
    ):
        text = (VLLM_DIR / "v1" / "lwd_control" / sub).read_text()
        assert "def _lwd_dispatch_message" in text, sub
        assert "def _lwd_on_payload" not in text, sub


def test_review_deleted_code_stays_deleted():
    """Code trimmed during review must not flow back (it comes back as
    a whole with the features when they land later):
    - zero-accept live-row removal (_filter_deliver_rows, the 4-card
      issue fix, to be brought back later);
    - edge scheduler request-entry orchestration lwd_edge_add_request
      (folded into the engine's add_request);
    - cloud-side hash-chain prefix caching (_lwd_build_hashed_request /
      local_hasher / request_block_hasher, the prefix caching feature
      lands separately);
    - abort notification wiring (feature not landed yet, the wire
      struct LwdAbortNotify is kept)."""
    cloud_sched = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_cloud_scheduler"
        / "lwd_cloud_scheduler.py"
    ).read_text()
    assert "_filter_deliver_rows" not in cloud_sched
    edge_sched = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_edge_scheduler"
        / "lwd_edge_scheduler.py"
    ).read_text()
    assert "def lwd_edge_add_request" not in edge_sched
    cloud_engine = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_cloud_scheduler"
        / "lwd_cloud_engine.py"
    ).read_text()
    for banned in (
        "_lwd_build_hashed_request",
        "local_hasher",
        "request_block_hasher",
        "LwdAbortNotify",
    ):
        assert banned not in cloud_engine, banned
    assert "LwdAbortNotify" not in edge_sched
    edge_engine = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_edge_scheduler"
        / "lwd_edge_engine.py"
    ).read_text()
    assert "LwdAbortNotify" not in edge_engine


def test_edge_add_request_communication_in_engine():
    """Request-entry division of labor (review: communication is best
    done in the engine): the engine's add_request orchestrates
    admission (lwd_admit_request) -> cloud notify (_lwd_notify_request)
    -> native super().add_request (abort_immediately goes through
    native abort_requests)."""
    edge_engine = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_edge_scheduler"
        / "lwd_edge_engine.py"
    ).read_text()
    assert "def add_request" in edge_engine
    assert "self.scheduler.lwd_admit_request(request)" in edge_engine
    assert "self._lwd_notify_request(request)" in edge_engine
    assert "super().add_request(request)" in edge_engine
    edge_sched = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_edge_scheduler"
        / "lwd_edge_scheduler.py"
    ).read_text()
    assert "def lwd_admit_request" in edge_sched
    # notify publishing (communication) no longer lives in the scheduler
    assert "def lwd_edge_notify_request" not in edge_sched


def test_edge_batch_queue_depth_is_two():
    """Batch queue depth = 2 (settled in review; performance validation
    to be done on a real machine environment)."""
    import regex as re

    edge_engine = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_edge_scheduler"
        / "lwd_edge_engine.py"
    ).read_text()
    match = re.search(r"^LWD_EDGE_BATCH_QUEUE_DEPTH\s*=\s*(\d+)", edge_engine, re.M)
    assert match is not None
    assert int(match.group(1)) == 2
