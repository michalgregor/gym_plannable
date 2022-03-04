import unittest
from gym_plannable.multi_agent import StopServerException, multi_agent_to_single_agent
from gym_plannable.env.tic_tac_toe import TicTacToeEnv
from multi_agent_mixins import (ServerTestMixin, ServerDeleteTestMixin,
                                ClientTestMixin, EnvTestMixin,
                                ClientScoreTestMixin, IllegalActionTestMixin)
from dummy_envs import DummyEnvTurnBased
from threading import Thread
import weakref
import gc

class ServerTestTicTacToe(ServerTestMixin, unittest.TestCase):
    env_constructor = TicTacToeEnv
    actions = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0]
    ]

class ServerTestDummyEnv(ServerTestMixin, unittest.TestCase):
    env_constructor = DummyEnvTurnBased
    actions = [0, 1, 2, 3, 0, 1, 2, 3]

class ServerDeleteTestTicTacToe(ServerDeleteTestMixin, unittest.TestCase):
    env_constructor = TicTacToeEnv
    actions = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0]
    ]

class ServerDeleteDummyEnv(ServerDeleteTestMixin, unittest.TestCase):
    env_constructor = DummyEnvTurnBased
    actions = [0, 1, 2, 3, 0, 1, 2, 3]

class ClientTestTicTacToe(ClientTestMixin, unittest.TestCase):
    env_constructor = TicTacToeEnv
    actions = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0]
    ]

class ClientTestDummyEnv(ClientTestMixin, unittest.TestCase):
    env_constructor = DummyEnvTurnBased
    actions = [0, 1, 2, 3, 0, 1, 2, 3]
    
class TestTicTacToeEnv(EnvTestMixin, unittest.TestCase):
    env_constructor = TicTacToeEnv

class TestDummyEnv(EnvTestMixin, unittest.TestCase):
    env_constructor = DummyEnvTurnBased

class ClientExceptionSafeTest(unittest.TestCase):
    actions = [0, 1, 2, 3, 0, 1, 2, 3]
    exception_at = 3

    def setUp(self):
        self.multiagent_env = DummyEnvTurnBased(exception_at=self.exception_at)
        self.clients, server = multi_agent_to_single_agent(self.multiagent_env, return_server=True)
        self.finished_event = server.csi.finished_event

    def tearDown(self):
        del self.clients
        gc.collect()
        self.assertTrue(self.finished_event.is_set())

    def testExceptionSafe(self):
        self.agent0_done = False
        self.agent1_done = False

        def agent0():
            with self.assertRaises(RuntimeError):
                env = weakref.proxy(self.clients[0])
                obs = env.reset()
                any_obs_none = obs is None

                for a in self.actions[::2]:
                    obs, rew, done, info = env.step(a)
                    any_obs_none = any_obs_none or obs is None
                    if done: break

                self.assertTrue(done)
                self.assertFalse(any_obs_none)

            self.agent0_done = True

        def agent1():
            with self.assertRaises(StopServerException):
                env = weakref.proxy(self.clients[1])
                obs = env.reset()
                any_obs_none = obs is None

                for a in self.actions[1::2]:
                    obs, rew, done, info = env.step(a)
                    any_obs_none = any_obs_none or obs is None
                    if done: break

                self.assertTrue(done)
                self.assertFalse(any_obs_none)

            self.agent1_done = True

        thread0 = Thread(target=weakref.proxy(agent0))
        thread1 = Thread(target=weakref.proxy(agent1))

        thread0.start()
        thread1.start()

        thread0.join()
        thread1.join()

        self.assertTrue(self.agent0_done)
        self.assertTrue(self.agent1_done)

class ClientScoreTestTicTacToe(ClientScoreTestMixin, unittest.TestCase):
    env_constructor = TicTacToeEnv
    actions = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0]
    ]
    
    scores = [1, -1]

class IllegalActionTestTicTacToe(IllegalActionTestMixin, unittest.TestCase):
    env_constructor = TicTacToeEnv

    agent0_actions0 = [[0, 0]]
    agent0_actions1 = [[0, 1], [0, 2]]
    agent1_actions = [[1, 0], [1, 1], [2, 0]]
