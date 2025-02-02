# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from unittest.mock import MagicMock, create_autospec, patch

import torch
from torch.distributed import TCPStore

from torchft.manager import MANAGER_ADDR_KEY, Manager, WorldSizeMode
from torchft.process_group import ProcessGroup, _DummyWork
from torchft.torchft import ManagerClient


class TestManager(TestCase):
    store: TCPStore  # pyre-fixme[13]: never initialized
    load_state_dict: MagicMock  # pyre-fixme[13]: never initialized

    def _create_manager(
        self,
        use_async_quorum: bool = True,
        min_replica_size: int = 2,
        world_size_mode: WorldSizeMode = WorldSizeMode.DYNAMIC,
    ) -> Manager:
        pg = create_autospec(ProcessGroup)
        self.store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        self.store.set(MANAGER_ADDR_KEY, "dummy")
        with patch(
            "os.environ",
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": self.store.port,
                "RANK": "1",
                "WORLD_SIZE": "2",
            },
        ):
            self.load_state_dict = MagicMock()
            manager = Manager(
                pg=pg,
                min_replica_size=min_replica_size,
                load_state_dict=self.load_state_dict,
                state_dict=lambda: {},
                use_async_quorum=use_async_quorum,
                world_size_mode=world_size_mode,
            )
        return manager

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        self.assertEqual(client_mock.call_count, 1)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_state_dict(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()

        state_dict = manager.state_dict()
        self.assertEqual(
            state_dict,
            {
                "step": 0,
                "batches_committed": 0,
            },
        )

        manager.load_state_dict(
            {
                "step": 1234,
                "batches_committed": 2345,
            }
        )
        self.assertEqual(manager.current_step(), 1234)
        self.assertEqual(manager.batches_committed(), 2345)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_happy(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        client_mock().should_commit = lambda rank, step, should_commit: should_commit

        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world
            "manager address",
            f"localhost:{self.store.port}",
            1,  # max_step
            1,  # max_rank
            2,  # max_world_size
            False,  # heal
        )

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager._step, 0)
        self.assertEqual(manager.batches_committed(), 0)

        manager.step()
        manager.allreduce_grad(torch.tensor([1.0])).wait()
        self.assertEqual(len(manager._pending_work), 1)
        self.assertTrue(manager.should_commit())
        self.assertEqual(len(manager._pending_work), 0)

        self.assertEqual(manager._quorum_id, 123)
        self.assertEqual(manager._step, 1)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.call_count, 1)

        manager.step()
        self.assertEqual(manager.batches_committed(), 2)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_heal_sync(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(use_async_quorum=False)
        client_mock().should_commit = lambda rank, step, should_commit: should_commit

        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world
            "manager address",
            f"localhost:{self.store.port}",
            20,  # max_step
            None,  # max_rank
            2,  # max_world_size
            True,  # heal
        )
        # forceable increment checkpoint server to compute correct address
        manager._ckpt_server.allow_checkpoint(1)

        client_mock().checkpoint_address.return_value = manager._ckpt_server.address()

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager._step, 0)

        manager.step()
        manager.allreduce_grad(torch.tensor([1.0])).wait()
        self.assertFalse(manager._healing)
        self.assertTrue(manager.is_participating())
        self.assertEqual(manager.num_participants(), 2)
        self.assertTrue(manager.should_commit())

        self.assertEqual(manager._quorum_id, 123)
        self.assertEqual(manager._step, 20)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.call_count, 1)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.return_value.get_future.call_count, 1)

        self.assertEqual(self.load_state_dict.call_count, 1)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_heal_async_not_enough_participants(
        self, client_mock: MagicMock
    ) -> None:
        manager = self._create_manager(use_async_quorum=True, min_replica_size=2)
        client_mock().should_commit = lambda rank, step, should_commit: should_commit

        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world
            "manager address",
            f"localhost:{self.store.port}",
            20,  # max_step
            None,  # max_rank
            1,  # max_world_size
            True,  # heal
        )
        # forceable increment checkpoint server to compute correct address
        manager._ckpt_server.allow_checkpoint(1)

        client_mock().checkpoint_address.return_value = manager._ckpt_server.address()

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager._step, 0)

        manager.step()
        assert manager._quorum_future is not None
        manager._quorum_future.result()
        self.assertTrue(manager._healing)
        self.assertFalse(manager.is_participating())
        self.assertEqual(manager.num_participants(), 1)

        grad = torch.tensor([1.0])
        manager.allreduce_grad(grad).wait()
        torch.testing.assert_close(grad, torch.zeros_like(grad))
        # don't commit since num_max < min_replica_size
        self.assertFalse(manager.should_commit())
        self.assertFalse(manager._should_step)

        self.assertEqual(manager._quorum_id, 123)
        self.assertEqual(manager._step, 20)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.call_count, 1)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.return_value.get_future.call_count, 1)

        self.assertEqual(self.load_state_dict.call_count, 1)

        # failed to commit so no step
        manager.step()
        self.assertEqual(manager._step, 20)
        self.assertEqual(manager.batches_committed(), 0)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_heal_async_zero_grad(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(use_async_quorum=True, min_replica_size=1)
        client_mock().should_commit = lambda rank, step, should_commit: should_commit

        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world
            "manager address",
            f"localhost:{self.store.port}",
            20,  # max_step
            None,  # max_rank
            1,  # max_world_size
            True,  # heal
        )
        # forceable increment checkpoint server to compute correct address
        manager._ckpt_server.allow_checkpoint(1)

        client_mock().checkpoint_address.return_value = manager._ckpt_server.address()

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager._step, 0)

        manager.step()
        assert manager._quorum_future is not None
        manager._quorum_future.result()
        self.assertTrue(manager._healing)

        grad = torch.tensor([1.0])
        manager.allreduce_grad(grad).wait()
        torch.testing.assert_close(grad, torch.zeros_like(grad))
        # don't commit since num_max < min_replica_size
        self.assertTrue(manager.should_commit())
        self.assertEqual(manager.num_participants(), 1)
        self.assertTrue(manager._should_step)

        self.assertEqual(manager._quorum_id, 123)
        self.assertEqual(manager._step, 20)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.call_count, 1)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.return_value.get_future.call_count, 1)

        self.assertEqual(self.load_state_dict.call_count, 1)

        manager.step()
        self.assertEqual(manager._step, 21)
        self.assertEqual(manager.batches_committed(), 1)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_allreduce_error(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        client_mock().should_commit = lambda rank, step, should_commit: should_commit

        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world
            "manager address",
            f"localhost:{self.store.port}",
            1,  # max_step
            1,  # max_rank
            2,  # max_world_size
            False,  # heal
        )

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager._step, 0)

        manager.step()
        manager.allreduce_grad(torch.tensor([1.0])).wait()
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.call_count, 1)

        # inject failure when work queued
        # pyre-ignore[16]: _pg is mocked
        manager._pg.allreduce.side_effect = RuntimeError("injected failure")
        manager.allreduce_grad(torch.tensor([1.0])).wait()
        self.assertTrue(manager._errored)
        # this should be skipped due to error
        manager.allreduce_grad(torch.tensor([1.0])).wait()
        self.assertEqual(manager._pg.allreduce.call_count, 2)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.return_value.get_future.call_count, 1)

        self.assertFalse(manager.should_commit())
        self.assertTrue(manager._errored)

        # cleanup
        manager._pg.allreduce.side_effect = None

        # inject failure when worked waited
        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world
            "manager address",
            f"localhost:{self.store.port}",
            2,  # max_step
            1,  # max_rank
            2,  # max_world_size
            False,  # heal
        )
        manager.step()

        bad_fut = torch.futures.Future()  # pyre-fixme[29]: not a function
        bad_fut.set_exception(RuntimeError("injected failure"))
        manager._pg.allreduce.return_value.get_future.return_value = bad_fut
        manager.allreduce_grad(torch.tensor([1.0])).wait()
        self.assertTrue(manager._errored)
        self.assertFalse(manager.should_commit())
        self.assertTrue(manager._errored)
        self.assertEqual(manager._pg.allreduce.return_value.get_future.call_count, 2)

        # cleanup
        manager._pg.allreduce.reset_mock(return_value=True)

        # recover on next step
        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world_size
            "manager address",
            f"localhost:{self.store.port}",
            3,  # max_step
            1,  # max_rank
            2,  # max_world_size
            False,  # heal
        )

        manager.step()
        manager.allreduce_grad(torch.tensor([1.0])).wait()
        self.assertTrue(manager.should_commit())

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_fixed_world_size(self, client_mock: MagicMock) -> None:
        # test active and spares
        for rank in [1, 2]:
            manager = self._create_manager(
                min_replica_size=2,
                world_size_mode=WorldSizeMode.FIXED_WITH_SPARES,
            )
            client_mock().should_commit = (
                lambda rank, step, should_commit: should_commit
            )

            client_mock().quorum.return_value = (
                123,  # quorum_id
                rank,  # replica_rank
                3,  # replica_world
                "manager address",
                f"localhost:{self.store.port}",
                1,  # max_step
                rank,  # max_rank
                3,  # max_world_size
                False,  # heal
            )

            self.assertEqual(manager._quorum_id, -1)
            self.assertEqual(manager._step, 0)
            self.assertEqual(manager.batches_committed(), 0)

            manager.step()
            manager.allreduce_grad(torch.tensor([1.0])).wait()

            self.assertEqual(manager.is_participating(), rank != 2)
            self.assertEqual(manager.num_participants(), 2)

            manager.step()
            self.assertEqual(manager.batches_committed(), 2)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager_report_error(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()

        self.assertFalse(manager.errored())
        manager.report_error()
        self.assertTrue(manager.errored())

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager_wrap_future(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()

        self.assertFalse(manager.errored())

        fut = torch.futures.Future()  # pyre-fixme[29]: not a function
        wrapped_fut = manager.wrap_future(fut, 2)

        fut.set_exception(RuntimeError("injected failure"))

        self.assertEqual(wrapped_fut.value(), 2)
        self.assertTrue(manager.errored())
        self.assertEqual(manager._pending_work, [wrapped_fut])

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager_numerics(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()

        manager._quorum_future = MagicMock()
        manager._participating_rank = 1
        manager._participating_world_size = 5
        self.assertEqual(manager.num_participants(), 5)
        # pyre-ignore[16]: _pg is mocked
        manager._pg.allreduce.return_value = _DummyWork(None)

        self.assertTrue(manager.is_participating())
        fut = torch.futures.Future()  # pyre-fixme[29]: not a function
        fut = manager.allreduce_grad(torch.tensor([1.0]))
        result = fut.value()
        torch.testing.assert_close(result, torch.tensor([1.0 / 5]))

        # check healing numerics
        manager._healing = True
        self.assertFalse(manager.is_participating())
        fut = torch.futures.Future()  # pyre-fixme[29]: not a function
        fut = manager.allreduce_grad(torch.tensor([1.0]))
        result = fut.value()
        torch.testing.assert_close(result, torch.tensor([0.0]))
