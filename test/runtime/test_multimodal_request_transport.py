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

"""Cross-host request routing preserves pixels and republishes destination SHM."""

import pickle
from collections import deque
from multiprocessing import shared_memory
from types import SimpleNamespace

import pytest
import torch
import zmq

from tokenspeed.runtime.engine.async_llm import AsyncLLM
from tokenspeed.runtime.engine.io_struct import (
    MsgpackDecoder,
    MsgpackEncoder,
    TokenizedGenerateReqInput,
)
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle
from tokenspeed.runtime.sampling.sampling_params import SamplingParams


class RequestChannel:
    def __init__(self, codec):
        self.codec = codec
        self.messages = deque()

    def send_pyobj(self, obj):
        if self.codec == "msgpack":
            frames = [bytes(frame) for frame in MsgpackEncoder().encode(obj)]
            decoded = MsgpackDecoder(TokenizedGenerateReqInput).decode(frames)
        else:
            decoded = pickle.loads(pickle.dumps(obj))
        self.messages.append(decoded)

    def recv_pyobj(self, flags):
        if not self.messages:
            raise zmq.Again()
        return self.messages.popleft()


def frontend(channel, *, nnodes, dp_size):
    return SimpleNamespace(
        server_args=SimpleNamespace(
            nnodes=nnodes,
            mapping=SimpleNamespace(attn=SimpleNamespace(has_dp=dp_size > 1)),
        ),
        engine_core_client=SimpleNamespace(send_to_scheduler=channel),
        rid_to_state={},
    )


def request(pixels):
    return TokenizedGenerateReqInput(
        rid="image",
        input_ids=[1, 2],
        sampling_params=SamplingParams(),
        multimodal_inputs=MultimodalInputs(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=pixels,
                    hash=123,
                    offsets=[(1, 1)],
                    model_specific_data={"grid": torch.tensor([1, 2, 3])},
                )
            ]
        ),
    )


@pytest.mark.parametrize("codec", ["msgpack", "pickle"])
@pytest.mark.parametrize("source_shm", [False, True])
def test_remote_dp_republishes_exact_pixels_before_tp_broadcast(codec, source_shm):
    pixels = torch.arange(96, dtype=torch.float32).reshape(8, 12).to(torch.bfloat16)
    req = request(pixels)
    original_name = None
    if source_shm:
        req.multimodal_inputs.publish_shm_features()
        original_name = req.multimodal_inputs.mm_items[0].feature_shm.shm_name
    channel = RequestChannel(codec)
    ingress = frontend(channel, nnodes=8, dp_size=8)
    AsyncLLM._send_one_request(ingress, SimpleNamespace(rid="image"), req)
    routed = channel.messages[0]
    item = routed.multimodal_inputs.mm_items[0]
    assert item.feature_shm is None
    assert torch.equal(item.feature.view(torch.uint8), pixels.view(torch.uint8))
    if original_name:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=original_name)
    leader = SimpleNamespace(attn_tp_rank=0, recv_func=channel)
    received = RequestHandler._drain_reqs(leader)
    local_item = received[0].multimodal_inputs.mm_items[0]
    handle = local_item.feature_shm
    assert local_item.feature is None
    assert handle is not None
    assert handle.shm_name != original_name
    assert local_item.hash == 123 and local_item.offsets == [(1, 1)]
    assert torch.equal(local_item.model_specific_data["grid"], torch.tensor([1, 2, 3]))
    peer = ShmTensorHandle(
        shm_name=handle.shm_name, shape=handle.shape, dtype=handle.dtype
    )
    try:
        handle.attach()
        peer.attach()
        for reader in (handle, peer):
            copied = torch.empty_like(pixels)
            reader.copy_into(copied)
            assert torch.equal(copied.view(torch.uint8), pixels.view(torch.uint8))
    finally:
        handle.release()
        peer.release()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.shm_name)


@pytest.mark.parametrize("nnodes,dp_size", [(1, 1), (1, 2), (2, 1)])
def test_frontend_retains_shm_when_producer_is_in_receiving_group(nnodes, dp_size):
    channel = RequestChannel("msgpack")
    req = request(torch.arange(8, dtype=torch.float32))
    ingress = frontend(channel, nnodes=nnodes, dp_size=dp_size)
    AsyncLLM._send_one_request(ingress, SimpleNamespace(rid="image"), req)
    handle = req.multimodal_inputs.mm_items[0].feature_shm
    try:
        leader = SimpleNamespace(attn_tp_rank=0, recv_func=channel)
        received = RequestHandler._drain_reqs(leader)
        assert (
            received[0].multimodal_inputs.mm_items[0].feature_shm.shm_name
            == handle.shm_name
        )
        assert received[0].multimodal_inputs.mm_items[0].feature is None
    finally:
        handle.release()
