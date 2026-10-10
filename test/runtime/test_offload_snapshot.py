# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Mixed authoritative history survives snapshot and fresh-page restore."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest
from tokenspeed.runtime.cache.transfer.layout import layout_from_cache_arena
from tokenspeed.runtime.cache.transfer.ops import (
    CacheTransfer,
    HostTier,
    RestoreOp,
    SnapshotOp,
)
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.offload_config import KVOffloadConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.attention.kv_cache.recipes.storage import (
    plan_cache_storage,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def make_arena(prefix, *, queries):
    group = CacheGroupSpec(
        group_id="history",
        retention="full_history",
        rows_per_page=8,
        entry_stride_tokens=1,
        transfer_policy="full_suffix",
        replayable=False,
    )
    names = ("layer.0.latent_kv", "layer.1.latent_kv")
    plan = pack(
        ((group, tuple(CacheFieldSpec(n, n, (8, 1, 32), "bfloat16") for n in names)),),
        prefix_granularity=prefix,
        cache_blocks_per_lcm_block={"history": prefix // 8},
        alignment=256,
        max_padding_fraction=0.25,
    ).bind(8)
    config = KVOffloadConfig(
        field_ids=(names[1],),
        hot_tokens=32,
        reserved_tokens=16 if queries == 4 else 8,
        request_slots=4,
        topk=16,
        queries=queries,
        host_budget_bytes=1 << 20,
        overlap=True,
        cyclic_tokens=12 if queries == 4 else 0,
        selection_consumers=(),
        device_rows=192 if queries == 4 else 160,
    )
    return (
        CacheArena(
            plan,
            "cuda",
            cache_group_specs=(group,),
            storage_plan=plan_cache_storage(plan, (group,), offload=config),
        ),
        names,
    )


def transfer(source, destination):
    return CacheTransfer(0, source, destination, "", 0)


@pytest.mark.parametrize("prefix", [8, 16])
@pytest.mark.parametrize("host_first", [False, True])
@pytest.mark.parametrize("queries", [1, 4])
def test_mixed_snapshot_restores_fresh_pages_and_invalidates_hot_slot(
    prefix, host_first, queries
):
    arena, names = make_arena(prefix, queries=queries)
    layout = layout_from_cache_arena(
        arena,
        consumers=tuple((name,) for name in names),
        group_ids=("history",),
        field_ids=frozenset(names),
    )
    # The complete authoritative regions are imaged; the hot payload is absent.
    assert layout.buffers[1] is arena.regions[names[1]]
    assert all(b is not arena.compute_field(names[1]) for b in layout.buffers)
    if host_first:
        layout = replace(
            layout,
            buffers=layout.buffers[::-1],
            groups=tuple(
                replace(
                    g,
                    fields=tuple(
                        replace(f, device_buffer_index=1 - f.device_buffer_index)
                        for f in g.fields
                    ),
                )
                for g in layout.groups
            ),
        )
    pool = SimpleNamespace(arena=arena, cache_transfer_layout=lambda: layout)
    router = CacheGroupRouter.__new__(CacheGroupRouter)
    router.device = "cuda"
    router._offload_adapter = SimpleNamespace(engine=arena.offload)
    executor = HostCacheExecutor(
        pool,
        draft_pool=None,
        l2_tier=False,
        host_ratio=0,
        host_size_gb=0,
        snapshot_pool=RetractionPoolRequest(
            host_gb=0,
            ratio=1,
            max_retracted_requests=2,
            tail_lcm_blocks_per_request=0,
        ),
        slot_state_exporters=(router,),
        io_backend="kernel",
        attn_tp_rank=0,
        dcp_rank=0,
    )
    originals = {}
    history = {name: arena.field(name).view(-1, 1, 32) for name in names}
    for index, name in enumerate(names):
        payload = (
            torch.arange(history[name].numel(), dtype=torch.int32)
            .reshape(history[name].shape)
            .to(torch.bfloat16)
            + index * 512
        )
        history[name].copy_(payload)
        originals[name] = payload[8:24].clone()
    stream = torch.cuda.current_stream()
    cache = arena.offload
    cache.begin(
        torch.tensor([1], device="cuda", dtype=torch.int32),
        num_extends=0,
        stream=stream,
    )
    selected = torch.arange(8, 24, device="cuda", dtype=torch.int32).repeat(queries, 1)
    _, writes = cache.resolve(
        names[1],
        selected,
        torch.arange(128, 128 + queries, device="cuda"),
        torch.arange(8, 8 + queries, device="cuda", dtype=torch.int32),
    )
    cache.fields[names[1]].device[writes.long()] = 99
    accepted = queries - 1 if queries > 1 else 1
    cache.commit(torch.tensor([accepted], device="cuda", dtype=torch.int32))
    originals[names[1]][:accepted].fill_(99)
    store = SnapshotOp(1, "victim", 1, 2, (transfer(1, 1), transfer(2, 2)))
    executor.submit_write_backs(
        [store], prerequisite_stream=stream, fence_stream=stream
    )
    # This overwrite must run after both the Host and Device image copies.
    arena.clear()
    torch.cuda.synchronize()
    assert [type(e).__name__ for e in executor.poll_results()] == ["SnapshotDoneEvent"]
    state = cache.fields[names[1]]
    state.keys.view(4, -1)[2].fill_(123)
    state.seeded[2] = 1
    restore = RestoreOp(
        2,
        "victim",
        2,
        2,
        (transfer(1, 3), transfer(2, 4)),
        (HostTier.SNAPSHOT_POOL,) * 2,
    )
    executor.submit_load_backs([restore], prerequisite_stream=stream)
    torch.cuda.synchronize()
    assert [type(e).__name__ for e in executor.poll_results()] == ["RestoreDoneEvent"]
    for name in names:
        assert torch.equal(history[name][24:40].cpu(), originals[name])
    assert (state.keys.view(4, -1)[2] == -1).all()
    assert state.seeded[2].item() == 0
    assert torch.equal(state.lru_slots.view(4, -1)[2], torch.arange(32, device="cuda"))
    cache.begin(
        torch.tensor([2], device="cuda", dtype=torch.int32),
        num_extends=0,
        stream=stream,
    )
    mapped, _ = cache.resolve(
        names[1],
        selected + 16,
        torch.arange(128, 128 + queries, device="cuda"),
        torch.arange(40, 40 + queries, device="cuda", dtype=torch.int32),
    )
    assert torch.equal(
        state.device[mapped.flatten().long()].cpu(),
        originals[names[1]].repeat(queries, 1, 1),
    )


def test_mixed_snapshot_rejects_unordered_cpu_dma():
    arena, names = make_arena(8, queries=1)
    layout = layout_from_cache_arena(
        arena,
        consumers=tuple((name,) for name in names),
        group_ids=("history",),
        field_ids=frozenset(names),
    )
    with pytest.raises(ValueError, match="stream-ordered snapshots"):
        HostCacheExecutor(
            SimpleNamespace(arena=arena, cache_transfer_layout=lambda: layout),
            draft_pool=None,
            l2_tier=False,
            host_ratio=0,
            host_size_gb=0,
            snapshot_pool=RetractionPoolRequest(
                host_gb=0,
                ratio=1,
                max_retracted_requests=2,
                tail_lcm_blocks_per_request=0,
            ),
            slot_state_exporters=(),
            io_backend="direct",
            attn_tp_rank=0,
            dcp_rank=0,
        )


def test_offload_rejects_local_recovery_prefill():
    arena, _ = make_arena(8, queries=1)
    with pytest.raises(ValueError, match="restore KV by snapshot"):
        arena.offload.begin(
            torch.tensor([1], device="cuda"),
            num_extends=1,
            stream=torch.cuda.current_stream(),
        )
