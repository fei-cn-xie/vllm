# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-hash chain for the embeds-only cloud request (mm parity).

The PR #3 review dropped the hash chain from the cloud engine pending a
separate prefix-caching design, and froze the old identifiers out of that
file. The 2026-10-20 multi-request run proved the gap fatal on its own:
with no hasher attached, the cloud request's hash list stays empty and
the first full block crashes cache_full_blocks with IndexError. This
module restores the chain OUTSIDE the frozen file so the freeze stands:

- first build uses the edge-computed wire chain (hashes over the REAL
  token ids; the cloud's local fallback hashes zero placeholders and can
  never match another request's entries);
- continuation, missing or length-mismatched chains fall back to the
  standard local hasher (fail-open), which also keeps the list long
  enough for the block indexing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import VllmConfig
    from vllm.v1.core.kv_cache_utils import KVCacheConfig
    from vllm.v1.request import Request


def lwd_cloud_block_hasher(
    vllm_config: "VllmConfig",
    kv_cache_config: "KVCacheConfig",
    wire_hashes: list[bytes],
) -> Callable[["Request"], list[Any]] | None:
    """Build the per-request hasher closure; None disables hashing."""
    if not vllm_config.cache_config.enable_prefix_caching:
        return None
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import (
        get_request_block_hasher,
        init_none_hash,
        resolve_kv_cache_block_sizes,
    )

    _, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    caching_hash_fn = get_hash_fn_by_name(
        vllm_config.cache_config.prefix_caching_hash_algo
    )
    init_none_hash(caching_hash_fn)
    fallback = get_request_block_hasher(hash_block_size, caching_hash_fn)
    wire = list(wire_hashes)

    def hasher(request: "Request") -> list[Any]:
        if len(request.block_hashes) == 0 and request.num_output_tokens == 0:
            if len(wire) == request.num_prompt_tokens // hash_block_size:
                return wire
        return fallback(request)

    return hasher
