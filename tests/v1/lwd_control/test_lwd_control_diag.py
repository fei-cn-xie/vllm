"""LwdControlDiag tests: switch semantics, bypass payload extraction, spec observation accumulation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag


def test_switches_strict_one_and_live_read(monkeypatch):
    """Switches are strictly == "1", read live per call (in-process env changes take effect immediately)."""
    monkeypatch.delenv("VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_LWD_DISABLE_DOWN", raising=False)
    assert LwdControlDiag.token_passthrough_enabled() is False
    assert LwdControlDiag.disable_down_enabled() is False
    # anything other than "1" is always False
    monkeypatch.setenv("VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE", "true")
    assert LwdControlDiag.token_passthrough_enabled() is False
    # live-read semantics + OR relationship
    monkeypatch.setenv("VLLM_ASCEND_LWD_DISABLE_DOWN", "1")
    assert LwdControlDiag.disable_down_enabled() is True
    assert LwdControlDiag.token_passthrough_enabled() is True
    monkeypatch.setenv("VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE", "1")
    assert LwdControlDiag.token_passthrough_enabled() is True


def _outputs(rows: dict[str, list[int]], finished=()):
    """Fake engine_core_outputs: {0: EngineCoreOutputs(outputs, finished)}."""
    outputs = [
        SimpleNamespace(request_id=rid, new_token_ids=toks, finish_reason=None)
        for rid, toks in rows.items()
    ]
    return {0: SimpleNamespace(outputs=outputs, finished_requests=list(finished))}


def test_bypass_token_payload_aligned_rows():
    """SKIP_SAMPLE payload: aligned to the c2e row set, missing rows padded empty."""
    outputs = _outputs({"a": [1, 2], "b": [3]})
    payload = LwdControlDiag.bypass_token_payload(["a", "zzz"], outputs)
    assert payload == [[1, 2], []]


def test_disable_down_rows_pads_finished():
    """DISABLE_DOWN row set: this step's tokens + finished padded as empty rows; an empty step is empty."""
    outputs = _outputs({"a": [7]}, finished=["b"])
    rows = LwdControlDiag.disable_down_rows(outputs)
    assert rows == {"a": [7], "b": []}
    assert LwdControlDiag.disable_down_rows(_outputs({})) == {}


def test_spec_observation_accumulates_native_formula():
    """Spec observation accumulates with the native formula:
    draft += len(spec_ids), accepted += len(generated) - 1;
    unknown/finished requests are skipped."""
    diag = LwdControlDiag()
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"a": [1, 2, 3], "gone": [9], "fin": [9]}
    )
    model_output = SimpleNamespace(
        sampled_token_ids=[[0, 4, 5, 6]],
        req_id_to_index={"a": 0},
    )
    requests = {
        "a": SimpleNamespace(is_finished=lambda: False),
        "fin": SimpleNamespace(is_finished=lambda: True),
    }
    diag.observe_spec_acceptance(scheduler_output, model_output, requests)
    # a: draft 3, accepted = len([0,4,5,6]) - 1 = 3; gone unknown / fin finished are not counted
    assert diag._spec_draft_tokens == 3
    assert diag._spec_accepted == 3
    # step without spec tokens: zero-cost return
    diag.observe_spec_acceptance(
        SimpleNamespace(scheduled_spec_decode_tokens={}), model_output, requests
    )
    assert diag._spec_draft_tokens == 3


def test_bypass_unembed_rows_gating(monkeypatch):
    """Passthrough decision: only an UNEMBED batch + the switch on +
    tokens carried in the notify yields a passthrough row set; it only
    produces (req_ids, bypass token) data, execution adapters belong
    to the edge engine."""
    try:
        from vllm.v1.core.sched.output import LwdBatchType
    except Exception:  # pragma: no cover - lightweight env with py<3.10
        pytest.skip("sched.output not importable (requires Python>=3.10)")

    batch = SimpleNamespace(
        batch_type=LwdBatchType.LWD_UNEMBED,
        batch_meta=SimpleNamespace(req_ids=["a"], token_ids=[[1, 2]]),
    )
    monkeypatch.delenv("VLLM_ASCEND_LWD_DISABLE_DOWN", raising=False)
    assert LwdControlDiag.bypass_unembed_rows(batch) is None
    assert LwdControlDiag.bypass_unembed_rows(None) is None
    monkeypatch.setenv("VLLM_ASCEND_LWD_DISABLE_DOWN", "1")
    assert LwdControlDiag.bypass_unembed_rows(batch) == (["a"], [[1, 2]])
    # UNEMBED batch without bypass tokens (a normal batch) is not bypassed
    empty = SimpleNamespace(
        batch_type=LwdBatchType.LWD_UNEMBED,
        batch_meta=SimpleNamespace(req_ids=["a"], token_ids=[]),
    )
    assert LwdControlDiag.bypass_unembed_rows(empty) is None


def test_debug_forwarding_degrades_to_noop_without_lwd_debug_package():
    """Breakpoint-4 regression (issue #18): when vllm.v1.lwd_debug is
    not migrated into main, probe forwarding degrades to a no-op
    instead of raising ModuleNotFoundError -- the first request / first
    step inevitably passes through these forwarding points. Once the
    package is migrated in, the same path automatically resumes real
    probing."""
    from vllm.v1.lwd_control.lwd_control_diag.lwd_control_diag import (
        _lwd_debug_mods,
    )

    control_log, debug, log_base = _lwd_debug_mods()
    # whether degraded or not, all three entries must be callable and must not raise
    log_base.set_debug(True)
    control_log.flight(3, 1)
    debug.edge_tokens_delivered("req-1", [1, 2, 3], None, None)
    debug.cloud_step(None, None, None)
    debug.cloud_request_admitted(None, None)
    # functional-side forwarding points likewise must not raise
    LwdControlDiag.set_debug(True)
    LwdControlDiag.edge_step_activity(3, 1)
    LwdControlDiag.cloud_request_admitted(None, None)
