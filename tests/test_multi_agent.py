import unittest
from gym_plannable.env.tic_tac_toe import TicTacToeEnv
from multi_agent_mixins import (ServerTestMixin, ServerDeleteTestMixin,
                                ClientTestMixin, EnvTestMixin)
from dummy_envs import DummyEnvTurnBased

class ServerTestTicTacToe(ServerTestMixin, unittest.TestCase):
    env_constructor = TicTacToeEnv
    actions = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0]
    ]

class ServerTestDummyTurn(ServerTestMixin, unittest.TestCase):
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

class TestDummyTurnEnv(EnvTestMixin, unittest.TestCase):
    env_constructor = DummyEnvTurnBased








# class MultiAgentExceptionSafetyTest(unittest.TestCase):
#     pass

# make sure that exceptions don't cause deadlocks







    # def testExceptionSafe(self):
    #     def agent0():
    #         env = weakref.proxy(self.clients[0])
    #         obs = env.reset()
    #         obs, rew, done, info = env.step([0, 0])
    #         obs, rew, done, info = env.step([1, 0])

    #     def agent1():
    #         env = weakref.proxy(self.clients[1])
    #         obs = env.reset()
    #         obs, rew, done, info = env.step([0, 1])
    #         obs, rew, done, info = env.step([1, 1])
    #         self.assertTrue(done)

    #     thread0 = Thread(target=weakref.proxy(agent0))
    #     thread1 = Thread(target=weakref.proxy(agent1))

    #     thread0.start()
    #     thread1.start()

    #     thread0.join()
    #     thread1.join()
