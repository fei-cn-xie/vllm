# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build and adoption of the prefill chunk alignment profile (the
completion piece of the edge/cloud chunk-division reproduction
contract).

Background (issue #18 follow-up): the edge is the chunk-division
authority and the cloud reproduces the in-batch boundaries through the
clamp parameters; but the cloud's native schedule() still has a set of
truncation branches the edge cannot derive locally — mamba block
alignment for GDN hybrid models (`_mamba_block_aligned_split`; the
edge kv cache has no mamba groups) and the prefill lookahead
reservation for multimodal MTP (the edge has no speculative config).
When the two sides' profiles disagree, the cloud silently re-cuts the
chunk, the edge sends embeds at the original size, and the cloud's
INJECT window validation crashes (observed live: chunk rows=1561 !=
scheduled=1536).

Fix: the cloud sends the scheduler's live attributes down via
LwdCloudParamsNotify and the edge adopts them verbatim onto its own
scheduler — the edge's native schedule() then runs the very same
upstream split code, zero logic duplication, upstream changes are
followed for free.

This module stays pure logic (no message-struct imports, attributes
are read duck-typed) so it is testable under any interpreter; the
consistency between struct fields and this module's attribute table is
frozen by a test AST guard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vllm.config import CacheConfig

try:  # testable in lightweight environments: log when the logger chain is available, stay silent otherwise
    from vllm.lwd_logger import lwd_logger
except Exception:  # pragma: no cover
    lwd_logger = None  # type: ignore[assignment]


class LwdChunkAlignSource(Protocol):
    """Profile read-side protocol: satisfied by both the message
    struct and test stubs."""

    @property
    def block_size(self) -> int: ...
    @property
    def hash_block_size(self) -> int: ...
    @property
    def need_mamba_block_aligned_split(self) -> bool: ...
    @property
    def mamba_prefill_checkpoint_alignment(self) -> int | None: ...
    @property
    def mamba_has_prefill_checkpoint_blocks(self) -> bool: ...
    @property
    def mamba_partial_cache_hit(self) -> bool: ...
    @property
    def mamba_fine_grained_prefix_cache(self) -> bool: ...
    @property
    def use_eagle_block_drop(self) -> bool: ...
    @property
    def num_prefill_lookahead(self) -> int: ...


# Scheduler attributes the edge has no legitimate local source for and
# adopts verbatim (one-to-one with the self attributes read at runtime
# by `_mamba_block_aligned_split` and `_reserve_prefill_lookahead`).
LWD_CHUNK_ALIGN_ADOPT_ATTRS: tuple[str, ...] = (
    "need_mamba_block_aligned_split",
    "mamba_prefill_checkpoint_alignment",
    "mamba_has_prefill_checkpoint_blocks",
    "mamba_partial_cache_hit",
    "mamba_fine_grained_prefix_cache",
    "use_eagle_block_drop",
    "num_prefill_lookahead",
)


def lwd_build_chunk_align_profile(scheduler, cache_config: CacheConfig):
    """Cloud-side fill: build the profile from the scheduler's live
    attributes (single source of truth, no secondary derivation).

    block_size comes from cache_config (same read semantics as
    `_mamba_block_aligned_split`); the rest come from the scheduler's
    init-derived attributes.
    """
    from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdChunkAlignProfile

    return LwdChunkAlignProfile(
        block_size=cache_config.block_size,
        hash_block_size=scheduler.hash_block_size,
        **{name: getattr(scheduler, name) for name in LWD_CHUNK_ALIGN_ADOPT_ATTRS},
    )


def lwd_apply_chunk_align_profile(
    scheduler, cache_config: CacheConfig, profile: LwdChunkAlignSource
) -> None:
    """Edge-side adoption: after verifying both sides' block sizes
    match, stamp the profile attributes onto the scheduler.

    Adoption happens during engine startup (before the first prefill
    batch is dispatched); from then on the edge's native schedule()
    truncates chunks in lockstep with the cloud. The block-size check
    is fail-fast: sizes have a local source (both sides derive the
    same platform default), so a mismatch means a deployment mismatch,
    and silently adopting would only make every reproduction wrong.
    """
    if profile.block_size != cache_config.block_size:
        raise RuntimeError(
            f"[LWD] edge/cloud block size mismatch: edge="
            f"{cache_config.block_size} cloud={profile.block_size}; the "
            "chunk alignment profile assumes equal block sizes, start both "
            "sides with the same value"
        )
    if profile.hash_block_size != scheduler.hash_block_size:
        raise RuntimeError(
            f"[LWD] edge/cloud hash block size mismatch: edge="
            f"{scheduler.hash_block_size} cloud={profile.hash_block_size}; "
            "start both sides with the same value"
        )
    for name in LWD_CHUNK_ALIGN_ADOPT_ATTRS:
        setattr(scheduler, name, getattr(profile, name))
    if lwd_logger is not None:
        lwd_logger.info(
            "[cloud-params] chunk align profile adopted: block=%d hash=%d "
            "mamba_split=%s eagle_drop=%s lookahead=%d",
            profile.block_size,
            profile.hash_block_size,
            profile.need_mamba_block_aligned_split,
            profile.use_eagle_block_drop,
            profile.num_prefill_lookahead,
        )
