# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observation-only guard for the 2026-10-20 aivec investigation.

Two rules after the reverts: the cloud must NOT neutralize penalties (the
earlier fix was pushed on a since-disproven mechanism — restore behavior
and OBSERVE instead), and the sampler probe must stay installed so the
next fault run reports the kernel inputs first-hand. Pure source scan.
"""

from __future__ import annotations

from pathlib import Path

ENGINE = (
    Path(__file__).resolve().parents[3]
    / "vllm" / "v1" / "lwd_control" / "control_cloud_scheduler"
    / "lwd_cloud_engine.py"
)


def test_penalties_are_forwarded_not_neutralized() -> None:
    src = ENGINE.read_text(encoding="utf-8")
    assert 'values["repetition_penalty"] = 1.0' not in src, (
        "penalty neutralization crept back; the aivec root cause is not "
        "proven - observe with the probe instead of changing behavior"
    )
    assert "wire penalties" in src, "the wire-penalty observation log is gone"


def test_sampler_probe_stays_installed() -> None:
    src = ENGINE.read_text(encoding="utf-8")
    assert "def lwd_install_sampler_probe() -> None:" in src
    assert "lwd_install_sampler_probe()" in src
    assert "bincount launch" in src and "OUT-OF-RANGE" in src
