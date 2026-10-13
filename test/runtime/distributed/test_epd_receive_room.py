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

"""EPD receive rooms have one owner even when upstream retries overlap."""

import threading
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.epd.mooncake.prefill import MooncakeEmbeddingManagerPrefill
from tokenspeed.runtime.epd.mooncake.receiver import MooncakeEmbeddingReceiver
from tokenspeed.runtime.pd.base.status import TransferPoll


@pytest.fixture
def manager():
    mgr = MooncakeEmbeddingManagerPrefill.__new__(MooncakeEmbeddingManagerPrefill)
    mgr.engine = SimpleNamespace(get_session_id=lambda: "session")
    mgr.request_status = {}
    mgr.required_response_num = {}
    mgr.response_tracker = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._room_lock = threading.Lock()
    mgr.world_size = 1
    mgr.dp_size = 1
    mgr.embedding_args = SimpleNamespace(engine_rank=0)
    mgr.prefill_parallel_info = {
        "encoder:1": {"prefill_tp_size": 1, "prefill_dp_size": 1}
    }
    mgr.connection_pool = {"encoder:1_0_0": [{}]}
    return mgr


def test_overlapping_retry_does_not_clear_original_receive(manager):
    original = MooncakeEmbeddingReceiver(manager, "encoder:1", 7)
    duplicate = MooncakeEmbeddingReceiver(manager, "encoder:1", 7)
    assert duplicate.poll() == TransferPoll.Failed
    duplicate.clear()
    assert original.poll() == TransferPoll.Bootstrapped
    manager._handle_status(7, TransferPoll.Success, 0)
    assert original.poll() == TransferPoll.Success
    original.clear()
    assert 7 not in manager.request_status
    assert original.poll() == TransferPoll.Success
    assert duplicate.poll() == TransferPoll.Failed
    duplicate.clear()
    original.clear()


@pytest.mark.parametrize("status", [TransferPoll.Success, TransferPoll.Failed])
def test_late_completion_does_not_recreate_cleared_room(manager, status):
    receiver = MooncakeEmbeddingReceiver(manager, "encoder:1", 7)
    receiver.clear()
    manager._handle_status(7, status, 0)
    assert receiver.poll() == TransferPoll.Failed
    assert not manager.request_status
    assert not manager.required_response_num
    assert not manager.response_tracker
    assert not manager.failure_records


def test_missing_room_fails_receiver_without_escaping_scheduler(manager):
    receiver = MooncakeEmbeddingReceiver(manager, "encoder:1", 7)
    manager.clear_room(7)
    assert receiver.poll() == TransferPoll.Failed
    receiver.clear()


def test_failure_remains_visible_until_receive_is_cleared(manager):
    receiver = MooncakeEmbeddingReceiver(manager, "encoder:1", 7)
    manager._handle_status(7, TransferPoll.Success, 0)
    assert receiver.poll() == TransferPoll.Success
    manager._handle_status(7, TransferPoll.Failed, 0)
    assert receiver.poll() == TransferPoll.Failed
    manager._handle_status(7, TransferPoll.Success, 0)
    assert receiver.poll() == TransferPoll.Failed
    receiver.clear()
    assert receiver.poll() == TransferPoll.Failed
