# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""chunk alignment profile tests (2026-10 lwd-debug-1019 INJECT window
mismatch regression): the adoption/validation logic runs on any
interpreter; the struct field table, the engine wiring, and the upstream
split attribute-read surface are frozen by AST source guards; the real
`_mamba_block_aligned_split` is reproduced equivalently as table-driven
execution cases (needs torch + vllm importable; lightweight environments
skip conditionally and re-run in the real environment).

Incident shape: the edge side is the chunk partitioning authority and the
cloud side reproduces it from the clamping parameters; but the cloud-native
schedule() mamba block alignment truncated 1561 down to 1536 (block-128
aligned) while the edge side sent embeds for 1561 -> the cloud INJECT
window validation crashed on rows=1561 != scheduled=1536. Fix:
LwdCloudParamsNotify ships the cloud alignment profile, the edge side
adopts it into the scheduler, and the edge-native schedule() runs the
same upstream code to perform the identical truncation.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.v1.lwd_control.control_scheduler.lwd_chunk_align import (
    LWD_CHUNK_ALIGN_ADOPT_ATTRS,
    lwd_apply_chunk_align_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MESSAGES = REPO_ROOT / "vllm" / "v1" / "lwd_control" / "lwd_comm" / "lwd_messages.py"
SCHEDULER = REPO_ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py"
CLOUD_ENGINE = (
    REPO_ROOT / "vllm" / "v1" / "lwd_control" / "control_cloud_scheduler"
    / "lwd_cloud_engine.py"
)
EDGE_ENGINE = (
    REPO_ROOT / "vllm" / "v1" / "lwd_control" / "control_edge_scheduler"
    / "lwd_edge_engine.py"
)


def make_profile(**overrides) -> SimpleNamespace:
    """Stub of real cloud-side values (Qwen3.6-27B + qwen3_5_mtp deployment shape)."""
    fields = {
        "block_size": 128,
        "hash_block_size": 128,
        "need_mamba_block_aligned_split": True,
        "mamba_prefill_checkpoint_alignment": None,
        "mamba_has_prefill_checkpoint_blocks": False,
        "mamba_partial_cache_hit": False,
        "mamba_fine_grained_prefix_cache": False,
        "use_eagle_block_drop": True,
        "num_prefill_lookahead": 3,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_edge_scheduler() -> SimpleNamespace:
    """Edge scheduler stub: init-derived values are all in the "no mamba / no spec" edge shape."""
    sched = SimpleNamespace(
        hash_block_size=128,
        need_mamba_block_aligned_split=False,
        mamba_prefill_checkpoint_alignment=None,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_partial_cache_hit=False,
        mamba_fine_grained_prefix_cache=False,
        use_eagle_block_drop=False,
        num_prefill_lookahead=0,
    )
    return sched


def make_cache(block_size: int = 128) -> SimpleNamespace:
    return SimpleNamespace(block_size=block_size)


def test_apply_stamps_all_adopt_attrs():
    """Adoption: all 7 attributes are stamped to cloud values; hash size keeps the edge's original value (validation only)."""
    sched = make_edge_scheduler()
    profile = make_profile()
    lwd_apply_chunk_align_profile(sched, make_cache(), profile)
    for name in LWD_CHUNK_ALIGN_ADOPT_ATTRS:
        assert getattr(sched, name) == getattr(profile, name), name
    assert sched.hash_block_size == 128


def test_apply_block_size_mismatch_raises():
    """Block size mismatch = deployment misalignment; fail fast instead of silently adopting wrong values."""
    sched = make_edge_scheduler()
    with pytest.raises(RuntimeError, match="block size mismatch"):
        lwd_apply_chunk_align_profile(sched, make_cache(16), make_profile())
    # The failure path must not partially stamp values
    assert sched.need_mamba_block_aligned_split is False


def test_apply_hash_block_size_mismatch_raises():
    sched = make_edge_scheduler()
    sched.hash_block_size = 64
    with pytest.raises(RuntimeError, match="hash block size mismatch"):
        lwd_apply_chunk_align_profile(sched, make_cache(), make_profile())


# --------------------------------------------------------------------- #
# AST source guards: runnable on any host                                 #
# --------------------------------------------------------------------- #

def _struct_fields(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    return {
        stmt.target.id
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }


def test_struct_fields_match_adopt_attrs():
    """Struct fields == validation bits (block/hash) + adoption bits; the two sides must not drift."""
    fields = _struct_fields(MESSAGES, "LwdChunkAlignProfile")
    assert fields == {"block_size", "hash_block_size"} | set(LWD_CHUNK_ALIGN_ADOPT_ATTRS)


def test_cloud_params_chunk_align_defaults_to_none():
    """chunk_align defaults to None: payloads encoded by older clouds stay decodable, and the edge reports version skew accordingly."""
    tree = ast.parse(MESSAGES.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LwdCloudParamsNotify"
    )
    fields = {
        stmt.target.id: stmt for stmt in node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    assert "chunk_align" in fields
    assert fields["chunk_align"].value is not None, "chunk_align must carry a default value"


def test_engines_wire_profile():
    """Wiring guard: the cloud must fill it in, the edge must adopt it and fail fast on None."""
    cloud = CLOUD_ENGINE.read_text(encoding="utf-8")
    assert "lwd_build_chunk_align_profile(" in cloud
    assert "self.scheduler" in cloud and "cache_config" in cloud
    edge = EDGE_ENGINE.read_text(encoding="utf-8")
    assert "lwd_apply_chunk_align_profile(" in edge
    assert re.search(r"chunk_align is None", edge), "missing None fail-fast"


def test_split_attr_reads_covered_by_profile():
    """Drift tripwire: any new self attribute read by the upstream split
    must enter the profile surface.

    `_mamba_block_aligned_split` may only read: context (cache_config/
    scheduler_config/max_num_scheduled_tokens), the validation bit
    hash_block_size, and the adoption bits. A new attribute = the cloud
    gained one more truncation input the edge does not know about; this
    guard raises the alarm before the on-site crash.
    """
    tree = ast.parse(SCHEDULER.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_mamba_block_aligned_split"
    )
    body = ast.get_source_segment(SCHEDULER.read_text(encoding="utf-8"), fn)
    reads = set(re.findall(r"self\.(\w+)", body))
    allowed = (
        {"cache_config", "scheduler_config", "max_num_scheduled_tokens",
         "hash_block_size"}
        | set(LWD_CHUNK_ALIGN_ADOPT_ATTRS)
    )
    assert reads <= allowed, f"split newly reads attributes {reads - allowed}: profile must be extended"


# --------------------------------------------------------------------- #
# Real split equivalence reproduction (needs torch; skipped in lightweight envs, re-run in the real env) #
# --------------------------------------------------------------------- #

try:
    from vllm.v1.core.sched.scheduler import Scheduler as _Scheduler
    SCHEDULER_CLS = _Scheduler
except Exception:  # pragma: no cover
    SCHEDULER_CLS = None  # type: ignore[assignment]

needs_scheduler = pytest.mark.skipif(
    SCHEDULER_CLS is None,
    reason="re-run in the real environment (Python>=3.10 + torch, vllm.v1.core.sched importable)",
)


def _split_self(profile: SimpleNamespace, threshold: int = 8192) -> SimpleNamespace:
    """Stub self carrying the full split attribute-read surface (attribute set frozen by the guard test)."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=profile.block_size),
        hash_block_size=profile.hash_block_size,
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(
            long_prefill_token_threshold=threshold
        ),
        **{name: getattr(profile, name) for name in LWD_CHUNK_ALIGN_ADOPT_ATTRS},
    )


@needs_scheduler
@pytest.mark.parametrize(
    ("prompt_len", "start", "num_new", "want", "profile_overrides"),
    [
        # Incident record: mid-prompt budget-truncated chunk 1561 -> cut to 1536 by 128 alignment
        (4096, 0, 1561, 1536, {}),
        # An already-aligned chunk passes through unchanged
        (4096, 0, 2048, 2048, {}),
        # Start inside a block: stop at the next block boundary first (shape where a prefix hit leaves the start mid-block)
        (4096, 100, 2048, 28, {}),
        # Final block is exempt from alignment truncation, but the stop point meshes with
        # eagle's one-block drop of last_cache_position (1561%128=25: last=1536, eagle drops one block -> 1408)
        (1561, 0, 1561, 1408, {}),
        # Same as above with eagle block-drop off: stops at 1536, the remaining 25 finishes
        # block by block (locks in the block-drop delta)
        (1561, 0, 1561, 1536, {"use_eagle_block_drop": False}),
        # Final block resumed from the dropped-block position: the stop point does not
        # strictly fall inside the segment, so the whole segment is admitted
        (1600, 1408, 192, 192, {}),
        # Final block with an in-block start: steps block by block up to the boundary
        (1600, 1440, 160, 96, {}),
        # Remaining budget cannot cross a block: aligned result is 0 (skipped this step; the whole block is scheduled next step)
        (4096, 0, 100, 0, {}),
    ],
)
def test_split_reproduces_cloud_cuts(
    prompt_len, start, num_new, want, profile_overrides
):
    """Cloud-native values vs edge-adopted values run the same upstream
    split; the outputs must be equal and match the table values — the
    edge now cuts the window the cloud will schedule before sending embeds."""
    profile = make_profile(**profile_overrides)
    request = SimpleNamespace(
        num_computed_tokens=start,
        num_tokens=prompt_len,
        num_prompt_tokens=prompt_len,
        shared_prefix_boundary=0,
    )
    cloud_sched = _split_self(profile)
    edge_sched = make_edge_scheduler()
    lwd_apply_chunk_align_profile(edge_sched, make_cache(), profile)
    edge_sched.cache_config = SimpleNamespace(block_size=profile.block_size)
    edge_sched.max_num_scheduled_tokens = 8192
    edge_sched.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=8192
    )
    cloud_cut = SCHEDULER_CLS._mamba_block_aligned_split(
        cloud_sched, request, num_new
    )
    edge_cut = SCHEDULER_CLS._mamba_block_aligned_split(
        edge_sched, request, num_new
    )
    assert cloud_cut == want, (cloud_cut, want)
    assert edge_cut == cloud_cut


@needs_scheduler
@pytest.mark.parametrize(
    ("num_tokens", "computed", "num_new", "lookahead", "want"),
    [
        # Multi-module MTP: chunk endpoint is 2 from the prefill tail, 2 < lookahead 3 -> trim 1
        (1538, 0, 1536, 3, 1535),
        # Endpoint is exactly the tail (remaining=0) -> admitted as-is
        (1538, 0, 1538, 3, 1538),
        # Edge without spec (lookahead=0, pre-adoption shape) -> admitted as-is;
        # after adoption it trims the same way as the cloud
        (1538, 0, 1536, 0, 1536),
    ],
)
def test_reserve_prefill_lookahead_reproduces(
    num_tokens, computed, num_new, lookahead, want
):
    """lookahead reservation is adopted with the profile: edge and cloud trim the chunk the same way."""
    request = SimpleNamespace(
        num_tokens=num_tokens,
        num_computed_tokens=computed,
        num_prompt_tokens=num_tokens,
        shared_prefix_boundary=0,
    )
    sched = SimpleNamespace(num_prefill_lookahead=lookahead)
    assert (
        SCHEDULER_CLS._reserve_prefill_lookahead(sched, request, computed, num_new)
        == want
    )
